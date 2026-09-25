![ML Level](https://img.shields.io/badge/ML_Level-1.2_%C2%B7_Applied_EDA_%26_Hypothesis_Testing-blue)

# Customer Churn Prediction

Work in progress — predicting customer churn from account/usage data. Structure and tooling are set up; EDA and modeling to follow.

Second project in an ongoing ML learning path (see [`ML-01-exam-scores-prediction`](https://github.com/jmb-python-developer/ML-01-exam-scores-prediction) for the first). Where that project built the full pipeline end-to-end on a small, clean feature set, this one goes deeper on the EDA side: univariate/bivariate analysis with actual hypothesis testing (chi-square, effect size) behind the visual reads, not just eyeballed plots.

## Models

Five classification models are trained on the same train/test split and evaluated with the same metrics: Random Forest, Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbours (KNN) and Gradient Boosting.

Random Forest and Logistic Regression are the main models of this project. SVM, KNN and Gradient Boosting are included for comparison purposes, and to practise a standard workflow of fitting several different models and choosing the best one. Each of these algorithms will be studied in greater depth in separate classification projects at a higher level.

## Project structure

```
customer_churn_prediction/
├── README.md
├── requirements.txt
├── setup.sh
├── data/
├── notebooks/
│   └── notebook.ipynb
├── model_exports/
└── app/
    └── app.py
```

## Setup

```bash
git clone git@github.com:jmb-python-developer/ML-02-customer-churn-prediction.git
cd ML-02-customer-churn-prediction
source setup.sh        # creates .venv (if missing), activates it, installs requirements.txt
```
