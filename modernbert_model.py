import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
import pickle
import os
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

class ModernBERTMAPModel:
    def __init__(self, model_name="answerdotai/ModernBERT-base"):
        """
        Initialize ModernBERT model for math misconception prediction
        
        Args:
            model_name (str): The ModernBERT model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Model: {self.model_name}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the competition data"""
        print("Loading data...")
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Clean text data
        train_data['StudentExplanation'] = train_data['StudentExplanation'].fillna("")
        test_data['StudentExplanation'] = test_data['StudentExplanation'].fillna("")
        
        # Create combined category:misconception labels
        combined_labels = []
        for _, row in train_data.iterrows():
            category = row['Category']
            misconception = row['Misconception'] if not pd.isna(row['Misconception']) else 'NA'
            combined_labels.append(f"{category}:{misconception}")
        
        train_data['combined_label'] = combined_labels
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(combined_labels)
        
        print(f"Number of unique labels: {len(self.label_encoder.classes_)}")
        print("Top 10 most common labels:")
        label_counts = pd.Series(combined_labels).value_counts()
        print(label_counts.head(10))
        
        return train_data, test_data, encoded_labels
    
    def prepare_tokenizer_and_model(self, num_labels):
        """Initialize ModernBERT tokenizer and model"""
        print(f"Loading ModernBERT tokenizer and model...")
        
        try:
            # Try to load ModernBERT - if not available, fallback to distilbert
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=num_labels,
                problem_type="single_label_classification",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1
            )
            print("✅ Successfully loaded ModernBERT model")
        except Exception as e:
            print(f"❌ Failed to load ModernBERT: {e}")
            print("🔄 Falling back to DistilBERT...")
            self.model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=num_labels,
                problem_type="single_label_classification"
            )
        
        # Move model to device
        self.model.to(self.device)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.tokenizer, self.model

class MAPDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if not pd.isna(self.texts[idx]) else ""
        
        # Tokenize with ModernBERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
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

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    # Calculate standard metrics
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    # Calculate MAP@3
    map3_score = calculate_map_at_k(labels, predictions, k=3)
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'map_at_3': map3_score
    }

def main():
    """Main function to run the ModernBERT NLP pipeline"""
    print("Starting ModernBERT Math Misconception NLP Pipeline...")
    print("="*70)
    
    # Initialize model handler
    nlp_model = ModernBERTMAPModel()
    
    # Load and preprocess data
    train_data, test_data, encoded_labels = nlp_model.load_and_preprocess_data()
    
    # Prepare model and tokenizer
    tokenizer, model = nlp_model.prepare_tokenizer_and_model(len(nlp_model.label_encoder.classes_))
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(
        train_data['StudentExplanation'].values,
        encoded_labels,
        test_size=0.2,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Create datasets with optimized parameters for CPU/limited GPU
    print("Creating datasets...")
    max_length = 128  # Reduced for efficiency
    train_dataset = MAPDataset(X_train, y_train, tokenizer, max_length)
    val_dataset = MAPDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = MAPDataset(
        test_data['StudentExplanation'].values, 
        np.zeros(len(test_data)),  # Dummy labels for test set
        tokenizer,
        max_length
    )
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Optimized training arguments for CPU/limited resources
    training_args = TrainingArguments(
        output_dir='./modernbert_results',
        num_train_epochs=2,  # Reduced for faster training
        per_device_train_batch_size=8,  # Small batch size for CPU
        per_device_eval_batch_size=16,
        warmup_steps=200,
        weight_decay=0.01,
        logging_dir='./modernbert_logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="map_at_3",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU available
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to=None,  # Disable wandb logging
        gradient_accumulation_steps=2,  # Effective batch size = 8*2 = 16
        learning_rate=2e-5,  # Standard BERT learning rate
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
    )
    
    # Initialize trainer
    print("Initializing ModernBERT trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("Starting ModernBERT training...")
    print("Note: This may take a while on CPU. Consider using GPU for faster training.")
    
    try:
        # Start training
        trainer.train()
        print("✅ Training completed successfully!")
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        print("Validation Results:")
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
        
        # Make predictions on test set
        print("Making predictions on test set...")
        test_predictions = trainer.predict(test_dataset)
        test_proba = F.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()
        
        # Format submission
        submission = format_submission(test_data, test_proba, nlp_model.label_encoder)
        
        # Save submission
        submission.to_csv('modernbert_submission.csv', index=False)
        print("✅ Submission saved as 'modernbert_submission.csv'")
        
        # Save model and tokenizer
        trainer.save_model('./modernbert_model')
        tokenizer.save_pretrained('./modernbert_model')
        
        # Save label encoder
        with open('./modernbert_model/label_encoder.pkl', 'wb') as f:
            pickle.dump(nlp_model.label_encoder, f)
        
        # Save training results
        results_summary = {
            'model_name': nlp_model.model_name,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'validation_results': eval_results,
            'training_args': training_args.to_dict(),
        }
        
        with open('./modernbert_model/training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=4, default=str)
        
        print("✅ Model, tokenizer, and results saved in './modernbert_model' directory")
        
        # Display results summary
        print("\n" + "="*70)
        print("MODERNBERT TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Model used: {nlp_model.model_name}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Best validation MAP@3 score: {eval_results.get('eval_map_at_3', 'N/A'):.4f}")
        print(f"Best validation accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Best validation F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print("\nFiles generated:")
        print("- modernbert_submission.csv (predictions)")
        print("- ./modernbert_model/ (trained model and tokenizer)")
        print("- ./modernbert_results/ (training logs)")
        print("- ./modernbert_model/training_results.json (detailed results)")
        
        return submission, eval_results
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("This might be due to memory limitations or model availability.")
        print("Consider:")
        print("1. Using a GPU for training")
        print("2. Reducing batch size further")
        print("3. Using a smaller model variant")
        return None, None

def format_submission(test_data, predictions_proba, label_encoder):
    """Format predictions according to competition requirements"""
    print("Formatting submission...")
    
    submission = pd.DataFrame()
    submission['row_id'] = test_data['row_id']
    
    predictions_formatted = []
    
    for proba in predictions_proba:
        # Get top 3 predictions
        top_3_idx = np.argsort(proba)[-3:][::-1]
        
        # Convert to label names
        top_3_labels = [label_encoder.classes_[idx] for idx in top_3_idx]
        
        # Format as space-separated string
        predictions_formatted.append(" ".join(top_3_labels))
    
    submission['Category:Misconception'] = predictions_formatted
    
    print("✅ Submission formatting completed!")
    return submission

if __name__ == "__main__":
    submission, results = main()
    
    if submission is not None:
        print("\nFirst few rows of submission:")
        print(submission.head())
        
        print("\n🎉 ModernBERT training completed successfully!")
        print("Check the generated files for detailed results.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        print("You may need to adjust the configuration for your system.")
