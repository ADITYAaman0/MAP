import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import warnings
import pickle
from tqdm import tqdm
import gc
import os

warnings.filterwarnings('ignore')

# Force CPU usage and optimize for memory
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class LightweightNeuralModel(nn.Module):
    """Lightweight neural model for text classification"""
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_classes=65, dropout=0.5):
        super(LightweightNeuralModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(embed_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, dropout=dropout)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size//2, num_heads=4, batch_first=True)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc2 = nn.Linear(hidden_size//4, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM layers
        lstm_out1, _ = self.lstm1(embedded)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Attention
        attn_output, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        
        # Global max pooling
        pooled = torch.max(attn_output, dim=1)[0]
        
        # Classification
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if not pd.isna(self.texts[idx]) else ""
        
        # Simple tokenization for lightweight model
        tokens = text.lower().split()[:self.max_length]
        
        # Convert to indices (simple word-to-index mapping)
        indices = [hash(token) % 10000 for token in tokens]  # Simple hash-based tokenization
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return {
            'input_ids': torch.tensor(indices, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def calculate_map_at_k(y_true, y_pred_proba, k=3):
    """Calculate Mean Average Precision at K"""
    def apk(actual, predicted, k):
        if len(predicted) > k:
            predicted = predicted[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(predicted):
            if p in actual:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        if not actual:
            return 0.0
        
        return score / min(len(actual), k)
    
    def mapk(actual, predicted, k):
        return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
    
    # Get top-k predictions for each sample
    predictions_list = []
    for proba in y_pred_proba:
        top_k_idx = np.argsort(proba)[-k:][::-1]
        predictions_list.append(top_k_idx.tolist())
    
    # Convert true labels to list format
    actual_list = [[label] for label in y_true]
    
    return mapk(actual_list, predictions_list, k)

def train_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    """Train the lightweight model"""
    device = torch.device('cpu')  # Force CPU usage
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': total_loss / num_batches})
            
            # Memory cleanup
            del input_ids, labels, outputs, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation')
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids)
                probabilities = F.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
                
                # Memory cleanup
                del input_ids, labels, outputs, probabilities, predicted
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # Calculate MAP@3
        map3_score = calculate_map_at_k(all_labels, all_probs, k=3)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Accuracy: {val_acc:.2f}%')
        print(f'  Val MAP@3: {map3_score:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_lightweight_model.pth')
            print(f'  New best model saved (Accuracy: {best_val_acc:.2f}%)')
        
        scheduler.step()
        print()
    
    return train_losses, val_accuracies, best_val_acc

def main():
    """Main function to run the lightweight advanced NLP pipeline"""
    print("Starting Lightweight Advanced Math Misconception NLP Pipeline...")
    print("="*70)
    
    # Load data
    print("Loading data...")
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Create combined labels
    combined_labels = []
    for _, row in train_data.iterrows():
        category = row['Category']
        misconception = row['Misconception'] if not pd.isna(row['Misconception']) else 'NA'
        combined_labels.append(f"{category}:{misconception}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(combined_labels)
    
    print(f"Number of unique labels: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_data['StudentExplanation'].values,
        encoded_labels,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets and data loaders
    print("Creating datasets...")
    train_dataset = TextDataset(X_train, y_train, None, max_length=64)  # Reduced max_length
    val_dataset = TextDataset(X_val, y_val, None, max_length=64)
    test_dataset = TextDataset(test_data['StudentExplanation'].values, np.zeros(len(test_data)), None, max_length=64)
    
    # Use smaller batch sizes for CPU training
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    print("Initializing lightweight model...")
    vocab_size = 10000  # Fixed vocabulary size
    model = LightweightNeuralModel(
        vocab_size=vocab_size,
        embed_size=64,  # Reduced embedding size
        hidden_size=128,  # Reduced hidden size
        num_classes=len(label_encoder.classes_),
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    train_losses, val_accuracies, best_val_acc = train_model(
        model, train_loader, val_loader, 
        num_epochs=3,  # Reduced epochs for faster training
        learning_rate=0.001
    )
    
    # Load best model for predictions
    model.load_state_dict(torch.load('best_lightweight_model.pth'))
    model.eval()
    
    # Make predictions on test set
    print("Making predictions on test set...")
    test_predictions = []
    test_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids']
            outputs = model(input_ids)
            probabilities = F.softmax(outputs, dim=1)
            
            test_probabilities.extend(probabilities.cpu().numpy())
    
    # Format submission
    print("Formatting submission...")
    submission = pd.DataFrame()
    submission['row_id'] = test_data['row_id']
    
    predictions_formatted = []
    for proba in test_probabilities:
        # Get top 3 predictions
        top_3_idx = np.argsort(proba)[-3:][::-1]
        
        # Convert to label names
        top_3_labels = [label_encoder.classes_[idx] for idx in top_3_idx]
        
        # Format as space-separated string
        predictions_formatted.append(" ".join(top_3_labels))
    
    submission['Category:Misconception'] = predictions_formatted
    
    # Save submission
    submission.to_csv('lightweight_submission.csv', index=False)
    print("Submission saved as 'lightweight_submission.csv'")
    
    # Save model and label encoder
    with open('lightweight_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("Label encoder saved as 'lightweight_label_encoder.pkl'")
    
    # Display results summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("\nFiles generated:")
    print("- lightweight_submission.csv (predictions)")
    print("- best_lightweight_model.pth (trained model)")
    print("- lightweight_label_encoder.pkl (label encoder)")
    
    print("\nFirst few rows of submission:")
    print(submission.head())
    
    return submission, best_val_acc

if __name__ == "__main__":
    try:
        submission, best_acc = main()
        print(f"\nFinal Results:")
        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        print("Model training and prediction completed successfully!")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
