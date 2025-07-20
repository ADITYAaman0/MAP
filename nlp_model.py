import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Preprocessing
def preprocess(text):
    # Tokenization and Lowercase
    tokens = nltk.word_tokenize(text.lower())
    # Remove stopwords and punctuation
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in tokens if word.isalpha() and word not in stopwords]
    return " ".join(words)

train_data['processed_explanation'] = train_data['StudentExplanation'].apply(preprocess)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(train_data['processed_explanation'])
y = train_data['Category']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and selection
pipeline = Pipeline([
    ('clf', LogisticRegression())
])

param_grid = {
    'clf__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate
val_predictions = best_model.predict(X_val)
print('Validation Mean Average Precision:', average_precision_score(y_val, val_predictions, average='macro'))

# Inference
processed_test_explanation = test_data['StudentExplanation'].apply(preprocess)
X_test = tfidf.transform(processed_test_explanation)
test_predictions = best_model.predict(X_test)

# Create Submission
test_data['Category'] = test_predictions
submission = test_data[['row_id']]
submission['Category:Misconception'] = test_data['Category'].apply(lambda x: f'{x}:NA False_Neither:NA False_Misconception:Incomplete')
submission.to_csv('submission.csv', index=False)
