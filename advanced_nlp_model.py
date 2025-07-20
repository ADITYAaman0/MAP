import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import datasets

# Load data
train_data = pd.read_csv('train.csv')

# Preprocess
train_data['label'] = train_data['Category'].factorize()[0]
train_text, val_text, train_labels, val_labels = train_test_split(train_data['StudentExplanation'], train_data['label'], test_size=0.2)

# Load model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def encode_batch(text_list):
    return tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')

train_encodings = encode_batch(train_text.tolist())
val_encodings = encode_batch(val_text.tolist())

class MAPDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = MAPDataset(train_encodings, train_labels.tolist())
val_dataset = MAPDataset(val_encodings, val_labels.tolist())

# Load the model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(train_data['label'].unique()))

# Define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train
trainer.train()
