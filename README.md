# Loan Default Prediction with Streamlit Deployment

## Overview
This project focuses on predicting whether a loan applicant is likely to default or not using machine learning techniques. It is based on a real
world scenario. The data is gotten from kaggle.

Financial institutions face significant risks due to loan defaults. 
By leveraging historical loan data, this model helps in making informed lending decisions, reducing financial losses, 
and improving risk management.

## Objectives
- Predict loan default status (0(Not Default)/1 (Default))
- Identify key factors influencing loan repayment
- Build and evaluate multiple classification models and selection of the best model based on the metrics used
- Deployment of said model in a web application using streamlit.

## Dataset
The dataset contains historical loan application records.

### Features include:
- Loan ID
- Age
- Income
- Loan Amount
- Credit Score
- Months Employed
- Number of Credit Lines
- Interest Rate
- Loan Term
- Debt-to-Income Ratio
- Education
- Employment Type
- Marital Status
- Has Mortgage (Boolean)
- Has Depedents (Boolean)
- Loan Purpose
- Has Cosigner (Boolean)

### Target Variable:
- Default (Target Variable)

Dataset Source: https://www.kaggle.com/code/adekunlesolomon/loan-default-classification-prediction/notebook

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn, Imblearn
- Matplotlib, Seaborn
- Streamlit (for deployment)

## Exploratory Data Analysis
- There were no missing values, outliers or skewed distribution. The distribution seemed normal.
- Dropped Loan ID column since it is unique
- Replaced irregular object variables to new ones 
- Detected heavy class imbalance in target variable

## Model Building
The following models were trained and evaluated using GridSearchCV:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- K-Neighbours Classifier

### Steps:
- Data preprocessing (encoding)
- Train-test split
- Scaling
- Model training using XGBClassifier and SMOTE (Due to the imbalance)
- Hyperparameter tuning

## Model Evaluation

Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score

| Model              | Accuracy | Precision | F1 Score | Recall |
|-------------------|----------|----------|----------|----------|
| XGBoost | 88.66%     |      |
| 0 | 88.66%     | 0.89     | 0.99     | 0.94     |
| 1 | 88.66%     | 0.51     | 0.10     | 0.16     |


## Deployment
The model is deployed using Streamlit using the render as host site.

🔗 Live App: https://dayfigurez-capstone-project-5.onrender.com/

### Features:
- User input form
- Real-time prediction
- Clean UI

## Installation

Clone the repository:
git clone https://github.com/KayD128/dayfigurez-capstone-project.git

Navigate to project folder:
cd root folder

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run loan.py

## Key Insights
- The data seems too random. It is hard to know which one is the strongest predictor. But using the feature importances, Age is gotten to be the most important.
- Class imbalance significantly affects model performance.
- It is recommended to get a better data for good model performance
- Can also adjust the decision threshold to balance the decision. This is a temporary solution.

## Contributing
Pull requests are welcome. For major changes, please open an issue first.

