# Math Misconception Annotation Project

## Overview

This project focuses on building an NLP model to predict students' potential math misconceptions based on their explanations. It utilizes various models to achieve this goal and evaluates their performance using the Mean Average Precision @ 3 (MAP@3) metric.

## Repository Structure

- **data/**: Contains the training and test datasets.
  - `train.csv`: Training data with explanations and labels.
  - `test.csv`: Test data for generating predictions.

- **models/**: Python scripts for different models and analysis.
  - `eda_analysis.py`: Script for exploratory data analysis.
  - `map_nlp_model.py`: Implementation using TF-IDF and Logistic Regression.
  - `bert_map_model.py`: Attempted implementation using BERT (resource-heavy).
  - `lightweight_advanced_model.py`: Custom lightweight neural network model.

- **outputs/**: Results and visualizations from the analysis and models.
  - `submission.csv`: Predictions from the basic model.
  - `lightweight_submission.csv`: Predictions from the lightweight model.
  - `eda_visualizations.png`: Visualization charts from EDA.

- **saved_models/**: Stored models and encoders.
  - `map_nlp_model.pkl`: Basic model and vectorizer.
  - `best_lightweight_model.pth`: Neural network weights.

- **PROJECT_SUMMARY.md**: Comprehensive project summary, explaining each step, model, and outcome.
- **requirements.txt**: Required Python packages and libraries.

## Instructions

1. **Setup**: Install dependencies using `pip install -r requirements.txt`.
2. **EDA**: Run `python models/eda_analysis.py` to analyze the data.
3. **Basic Model**: Run `python models/map_nlp_model.py` for the TF-IDF based solution.
4. **Advanced Model**: For lightweight model, run `python models/lightweight_advanced_model.py`.

## Outcome

- **Basic Model**: Achieved MAP@3 score of 0.7213, suitable for deployment.
- **Lightweight Model**: Achieved MAP@3 score of 0.5356, optimized for CPUs.

## Future Work

- Explore GPU-accelerated training.
- Enhance feature engineering.
- Implement ensemble methods for better accuracy.

## Contact

For any questions or support, please feel free to reach out.

