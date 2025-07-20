import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification, 
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from tqdm import tqdm
import json

warnings.filterwarnings('ignore')

class MAPNLPAdvanced:
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess the competition data"""
        print("Loading data...")
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
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
        print("Label distribution:")
        label_counts = pd.Series(combined_labels).value_counts()
        print(label_counts.head(10))
        
        return train_data, test_data, encoded_labels
    
    def prepare_tokenizer_and_model(self, num_labels):
        """Initialize tokenizer and model"""
        print(f"Loading tokenizer and model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Move model to device
        self.model.to(self.device)
        
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
    """Main function to run the advanced NLP pipeline"""
    print("Starting Advanced Math Misconception NLP Pipeline with BERT...")
    print("="*70)
    
    # Initialize model handler
    nlp_model = MAPNLPAdvanced(model_name="distilbert-base-uncased")  # Using DistilBERT for faster training
    
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
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = MAPDataset(X_train, y_train, tokenizer)
    val_dataset = MAPDataset(X_val, y_val, tokenizer)
    test_dataset = MAPDataset(
        test_data['StudentExplanation'].values, 
        np.zeros(len(test_data)),  # Dummy labels for test set
        tokenizer
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./bert_results',
        num_train_epochs=2,  # Reduced for faster training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./bert_logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="map_at_3",
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
        fp16=False,  # Set to True if using GPU with mixed precision support
        report_to=None  # Disable wandb logging
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("Starting training...")
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Evaluate on validation set
        print("Evaluating on validation set...")
        eval_results = trainer.evaluate()
        print("Validation Results:")
        for key, value in eval_results.items():
            print(f"{key}: {value:.4f}")
        
        # Make predictions on test set
        print("Making predictions on test set...")
        test_predictions = trainer.predict(test_dataset)
        test_proba = F.softmax(torch.tensor(test_predictions.predictions), dim=1).numpy()
        
        # Format submission
        submission = format_submission(test_data, test_proba, nlp_model.label_encoder)
        
        # Save submission
        submission.to_csv('bert_submission.csv', index=False)
        print("Submission saved as 'bert_submission.csv'")
        
        # Save model and tokenizer
        trainer.save_model('./bert_model')
        tokenizer.save_pretrained('./bert_model')
        
        # Save label encoder
        import pickle
        with open('./bert_model/label_encoder.pkl', 'wb') as f:
            pickle.dump(nlp_model.label_encoder, f)
        
        print("Model, tokenizer, and label encoder saved in './bert_model' directory")
        
        # Display results summary
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Best validation MAP@3 score: {eval_results.get('eval_map_at_3', 'N/A'):.4f}")
        print(f"Best validation accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
        print(f"Best validation F1: {eval_results.get('eval_f1', 'N/A'):.4f}")
        print("\nFiles generated:")
        print("- bert_submission.csv (predictions)")
        print("- ./bert_model/ (trained model)")
        print("- ./bert_results/ (training logs)")
        
        return submission, eval_results
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
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
    
    print("Submission formatting completed!")
    return submission

if __name__ == "__main__":
    submission, results = main()
    
    if submission is not None:
        print("\nFirst few rows of submission:")
        print(submission.head())
    else:
        print("Training failed. Please check the error messages above.")
