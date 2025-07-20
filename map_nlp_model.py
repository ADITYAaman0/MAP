import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

warnings.filterwarnings('ignore')

class MAPNLPModel:
    def __init__(self):
        self.tfidf = None
        self.label_encoder = None
        self.model = None
        self.misconception_encoder = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers in some contexts, keep mathematical notation
        text = re.sub(r'[^\w\s/\\()=+-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_category_misconception_labels(self, train_data):
        """Create combined category:misconception labels"""
        labels = []
        for _, row in train_data.iterrows():
            category = row['Category']
            misconception = row['Misconception']
            if pd.isna(misconception):
                misconception = 'NA'
            labels.append(f"{category}:{misconception}")
        return labels
    
    def load_and_explore_data(self):
        """Load and explore the dataset"""
        print("Loading data...")
        train_data = pd.read_csv('train.csv')
        test_data = pd.read_csv('test.csv')
        sample_submission = pd.read_csv('sample_submission.csv')
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        # Explore the data
        print("\nCategory distribution:")
        print(train_data['Category'].value_counts())
        
        print("\nMisconception distribution:")
        print(train_data['Misconception'].value_counts())
        
        return train_data, test_data, sample_submission
    
    def extract_features(self, train_explanations, test_explanations=None):
        """Extract TF-IDF features from text"""
        print("Extracting features...")
        
        # TF-IDF Vectorization
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        X_train = self.tfidf.fit_transform(train_explanations)
        
        if test_explanations is not None:
            X_test = self.tfidf.transform(test_explanations)
            return X_train, X_test
        
        return X_train
    
    def train_model(self, X_train, y_train):
        """Train the classification model"""
        print("Training model...")
        
        # Use Logistic Regression with One-vs-Rest for multi-class
        self.model = OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0
            )
        )
        
        self.model.fit(X_train, y_train)
        print("Model training completed!")
    
    def predict_with_probabilities(self, X_test):
        """Predict with probabilities for MAP@3 calculation"""
        # Get prediction probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_test)
        elif hasattr(self.model, "decision_function"):
            proba = self.model.decision_function(X_test)
        else:
            # Fallback to simple predictions
            return self.model.predict(X_test)
        
        return proba
    
    def format_submission(self, test_data, predictions_proba):
        """Format predictions according to submission requirements"""
        print("Formatting submission...")
        
        submission = pd.DataFrame()
        submission['row_id'] = test_data['row_id']
        
        predictions_formatted = []
        
        for i, proba in enumerate(predictions_proba):
            # Get top 3 predictions
            if len(proba.shape) > 1:
                proba = proba[0] if proba.shape[0] == 1 else proba
            
            # Get indices of top 3 predictions
            top_3_idx = np.argsort(proba)[-3:][::-1]
            
            # Get class names for top 3 predictions
            class_names = self.model.classes_
            
            predictions_list = []
            for idx in top_3_idx:
                if idx < len(class_names):
                    predictions_list.append(class_names[idx])
            
            # Ensure we have exactly 3 predictions
            while len(predictions_list) < 3:
                predictions_list.append("False_Misconception:Incomplete")
            
            predictions_formatted.append(" ".join(predictions_list[:3]))
        
        submission['Category:Misconception'] = predictions_formatted
        return submission
    
    def calculate_map_at_3(self, y_true, y_pred_proba):
        """Calculate Mean Average Precision @ 3"""
        def apk(actual, predicted, k=3):
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
        
        def mapk(actual, predicted, k=3):
            return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
        
        # Convert predictions to format needed for MAP@3
        predictions_list = []
        for proba in y_pred_proba:
            if len(proba.shape) > 1:
                proba = proba[0] if proba.shape[0] == 1 else proba
            top_3_idx = np.argsort(proba)[-3:][::-1]
            class_names = self.model.classes_
            pred_list = [class_names[idx] for idx in top_3_idx if idx < len(class_names)]
            predictions_list.append(pred_list)
        
        # Convert true labels to list format
        actual_list = [[label] for label in y_true]
        
        return mapk(actual_list, predictions_list)
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline"""
        print("Starting Math Misconception NLP Pipeline...")
        
        # Load data
        train_data, test_data, sample_submission = self.load_and_explore_data()
        
        # Preprocess text
        print("Preprocessing text data...")
        train_data['processed_explanation'] = train_data['StudentExplanation'].apply(self.preprocess_text)
        test_data['processed_explanation'] = test_data['StudentExplanation'].apply(self.preprocess_text)
        
        # Create combined labels
        combined_labels = self.create_category_misconception_labels(train_data)
        
        # Split data for validation
        X_train_text = train_data['processed_explanation']
        y_train = combined_labels
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_text, y_train, test_size=0.2, random_state=42
        )
        
        # Extract features
        X_train_features = self.extract_features(X_train_split)
        X_val_features = self.tfidf.transform(X_val_split)
        X_test_features = self.tfidf.transform(test_data['processed_explanation'])
        
        # Train model
        self.train_model(X_train_features, y_train_split)
        
        # Validate model
        print("Validating model...")
        val_pred_proba = self.predict_with_probabilities(X_val_features)
        map3_score = self.calculate_map_at_3(y_val_split, val_pred_proba)
        print(f"Validation MAP@3 Score: {map3_score:.4f}")
        
        # Make predictions on test set
        print("Making predictions on test set...")
        test_pred_proba = self.predict_with_probabilities(X_test_features)
        
        # Format submission
        submission = self.format_submission(test_data, test_pred_proba)
        
        # Save submission
        submission.to_csv('submission.csv', index=False)
        print("Submission saved as 'submission.csv'")
        
        # Save model
        with open('map_nlp_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'tfidf': self.tfidf
            }, f)
        print("Model saved as 'map_nlp_model.pkl'")
        
        return submission, map3_score

if __name__ == "__main__":
    # Initialize and run the model
    nlp_model = MAPNLPModel()
    submission, map3_score = nlp_model.run_complete_pipeline()
    
    print(f"\nFinal Validation MAP@3 Score: {map3_score:.4f}")
    print("Submission file created successfully!")
    print("\nFirst few rows of submission:")
    print(submission.head())
