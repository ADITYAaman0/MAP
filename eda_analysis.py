import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re

def load_data():
    """Load the competition data"""
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    sample_submission = pd.read_csv('sample_submission.csv')
    return train_data, test_data, sample_submission

def basic_statistics(train_data):
    """Generate basic statistics about the dataset"""
    print("=== BASIC DATASET STATISTICS ===")
    print(f"Training data shape: {train_data.shape}")
    print(f"Number of unique questions: {train_data['QuestionId'].nunique()}")
    print(f"Number of unique students explanations: {train_data['StudentExplanation'].nunique()}")
    
    print("\n=== CATEGORY DISTRIBUTION ===")
    category_counts = train_data['Category'].value_counts()
    print(category_counts)
    print(f"Category percentages:")
    print((category_counts / len(train_data) * 100).round(2))
    
    print("\n=== MISCONCEPTION DISTRIBUTION ===")
    misconception_counts = train_data['Misconception'].value_counts()
    print(misconception_counts.head(10))
    print(f"Number of unique misconceptions: {train_data['Misconception'].nunique()}")
    print(f"NA misconceptions: {train_data['Misconception'].isna().sum()}")

def text_analysis(train_data):
    """Analyze the text characteristics"""
    print("\n=== TEXT ANALYSIS ===")
    
    # Calculate text lengths
    train_data['explanation_length'] = train_data['StudentExplanation'].str.len()
    train_data['explanation_words'] = train_data['StudentExplanation'].str.split().str.len()
    
    print(f"Average explanation length (characters): {train_data['explanation_length'].mean():.2f}")
    print(f"Average explanation length (words): {train_data['explanation_words'].mean():.2f}")
    print(f"Median explanation length (characters): {train_data['explanation_length'].median():.2f}")
    print(f"Median explanation length (words): {train_data['explanation_words'].median():.2f}")
    
    return train_data

def create_visualizations(train_data):
    """Create visualizations for the data"""
    plt.ioff()  # Turn off interactive mode
    try:
        plt.style.use('default')  # Use default style instead
    except:
        pass
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Category distribution
    category_counts = train_data['Category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Category Distribution')
    
    # Explanation length distribution
    axes[0, 1].hist(train_data['explanation_length'], bins=50, alpha=0.7)
    axes[0, 1].set_xlabel('Explanation Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Explanation Lengths')
    
    # Word count distribution
    axes[1, 0].hist(train_data['explanation_words'], bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Explanation Length (words)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Explanation Word Counts')
    
    # Top misconceptions
    misconception_counts = train_data['Misconception'].value_counts().head(10)
    axes[1, 1].barh(range(len(misconception_counts)), misconception_counts.values)
    axes[1, 1].set_yticks(range(len(misconception_counts)))
    axes[1, 1].set_yticklabels(misconception_counts.index, fontsize=8)
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].set_title('Top 10 Misconceptions')
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print("Visualizations saved as 'eda_visualizations.png'")

def create_wordclouds(train_data):
    """Create word clouds for different categories"""
    categories = train_data['Category'].unique()
    
    fig, axes = plt.subplots(1, len(categories), figsize=(5*len(categories), 5))
    if len(categories) == 1:
        axes = [axes]
    
    for i, category in enumerate(categories):
        category_explanations = train_data[train_data['Category'] == category]['StudentExplanation'].dropna()
        text = ' '.join(category_explanations.astype(str))
        
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'WordCloud: {category}')
    
    plt.tight_layout()
    plt.savefig('wordclouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Word clouds saved as 'wordclouds.png'")

def analyze_question_types(train_data):
    """Analyze different types of questions"""
    print("\n=== QUESTION ANALYSIS ===")
    
    # Group by question to see variation in student responses
    question_analysis = train_data.groupby('QuestionId').agg({
        'Category': lambda x: x.value_counts().to_dict(),
        'StudentExplanation': 'count',
        'Misconception': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    print(f"Average number of student responses per question: {train_data.groupby('QuestionId').size().mean():.2f}")
    
    # Find questions with most diverse responses
    diversity_scores = []
    for _, row in question_analysis.iterrows():
        category_diversity = len(row['Category'])
        diversity_scores.append(category_diversity)
    
    question_analysis['category_diversity'] = diversity_scores
    most_diverse = question_analysis.nlargest(5, 'category_diversity')
    
    print("\nQuestions with most diverse student responses:")
    for _, row in most_diverse.iterrows():
        print(f"Question {row['QuestionId']}: {row['category_diversity']} different categories")

def export_analysis_summary(train_data, test_data):
    """Export a summary of the analysis"""
    summary = {
        'train_size': len(train_data),
        'test_size': len(test_data),
        'unique_questions_train': train_data['QuestionId'].nunique(),
        'unique_questions_test': test_data['QuestionId'].nunique(),
        'category_distribution': train_data['Category'].value_counts().to_dict(),
        'avg_explanation_length': train_data['explanation_length'].mean(),
        'avg_explanation_words': train_data['explanation_words'].mean(),
        'unique_misconceptions': train_data['Misconception'].nunique()
    }
    
    # Save as JSON
    import json
    with open('eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nAnalysis summary exported to 'eda_summary.json'")

def main():
    """Main function to run all analyses"""
    print("Starting Exploratory Data Analysis for MAP Competition")
    print("="*60)
    
    # Load data
    train_data, test_data, sample_submission = load_data()
    
    # Basic statistics
    basic_statistics(train_data)
    
    # Text analysis
    train_data = text_analysis(train_data)
    
    # Create visualizations
    create_visualizations(train_data)
    
    # Create word clouds
    create_wordclouds(train_data)
    
    # Analyze question types
    analyze_question_types(train_data)
    
    # Export summary
    export_analysis_summary(train_data, test_data)
    
    print("\n" + "="*60)
    print("EDA completed! Check generated files:")
    print("- eda_visualizations.png")
    print("- wordclouds.png") 
    print("- eda_summary.json")

if __name__ == "__main__":
    main()
