![ML Level](https://img.shields.io/badge/ML_Level-1.2_%C2%B7_Applied_EDA_%26_Hypothesis_Testing-blue)

# Customer Churn Prediction

Predicts whether a bank customer will leave, from account and demographic data, using scikit-learn, with a small Streamlit app for trying the exported model interactively.

Second project in an ongoing ML learning path (see [`ML-01-exam-scores-prediction`](https://github.com/jmb-python-developer/ML-01-exam-scores-prediction) for the first). Where that project built the full pipeline end-to-end on a small, clean feature set, this one goes deeper on the EDA side: univariate/bivariate analysis with actual hypothesis testing (chi-square, effect size) behind the visual reads, not just eyeballed plots.

## Dataset

`data/Churn_Modelling.csv` contains 10,000 customers and 14 columns. The target is `Exited`, which is 1 for the 20.37% of customers who left. The identifiers `RowNumber`, `CustomerId` and `Surname` are excluded from the features, and the remaining columns yield 12 features after encoding.

## Approach

1. **EDA.** The data was checked for nulls and duplicates and examined with univariate and bivariate analysis. Chi-square tests support the findings for `Geography` and `Gender`. Age, the number of products, active membership, Germany as a country and the presence of a balance showed the clearest relationship with churn.
2. **Feature engineering.** `Gender` is encoded with `OrdinalEncoder` and `Geography` with one-hot encoding. The data is split 80/20 with `random_state=42`.
3. **Models.** Five models are trained and evaluated with one reusable method that prints the confusion matrix, the accuracy and the precision, recall and f1-score of each group. Logistic Regression, SVM and KNN receive `Balance`, `EstimatedSalary`, `CreditScore` and `Age` scaled with `StandardScaler` inside a `Pipeline`. The two tree-based models receive the raw features.
4. **Selection.** The winner is chosen by the mean f1-score for leavers in 5-fold cross-validation on the training data. The test set is not used for the choice.
5. **Export and app.** The selected model is serialized with `joblib` and loaded by a small Streamlit app.

## Models

Five classification models are trained on the same train/test split and evaluated with the same metrics: Random Forest, Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbours (KNN) and Gradient Boosting.

Random Forest and Logistic Regression are the main models of this project. SVM, KNN and Gradient Boosting are included for comparison purposes, and to practise a standard workflow of fitting several different models and choosing the best one. Each of these algorithms will be studied in greater depth in separate classification projects at a higher level.

## Results

Mean f1-score for leavers in 5-fold cross-validation, training data only (this selected the exported model):

| Model | CV f1-score |
|---|---|
| **Gradient Boosting** | **0.586** |
| Random Forest | 0.576 |
| KNN | 0.437 |
| Logistic Regression | 0.319 |
| SVM | 0.219 |

Evaluation on the test set of 2,000 customers, of whom 393 left:

| Model | Accuracy | Precision (left) | Recall (left) | f1-score (left) |
|---|---|---|---|---|
| Random Forest | 86.8% | 0.77 | 0.47 | 0.58 |
| Gradient Boosting | 86.4% | 0.74 | 0.47 | 0.58 |
| KNN | 83.2% | 0.64 | 0.34 | 0.44 |
| SVM | 83.3% | 0.94 | 0.16 | 0.27 |
| Logistic Regression | 81.1% | 0.55 | 0.20 | 0.29 |

Gradient Boosting obtained the highest cross-validated score, narrowly ahead of the Random Forest, and is the model exported for the app. On the test set the two models are practically equivalent.

## Project structure

```
customer_churn_prediction/
├── README.md
├── requirements.txt
├── setup.sh
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   └── notebook.ipynb          # EDA, modeling, comparison and export
├── model_exports/
│   └── best_model.joblib       # exported model
└── app/
    └── app.py                  # Streamlit app
```

## Setup

```bash
git clone git@github.com:jmb-python-developer/ML-02-customer-churn-prediction.git
cd ML-02-customer-churn-prediction
source setup.sh        # creates .venv (if missing), activates it, installs requirements.txt
```

## Usage

**Notebook.** Open `notebooks/notebook.ipynb` in Jupyter or VS Code and run it from top to bottom. Running the last cell regenerates `model_exports/best_model.joblib`.

**App**
```bash
cd app
streamlit run app.py
```

## Notes

- Accuracy is reported, and the f1-score for leavers is used for selection, since a model that predicts "stays" for every customer reaches 80.4% accuracy on this data.
- Every model is trained with its default settings. Hyperparameter tuning is outside the scope of this project.
- The test set is used to report the results of each model, and the choice of the exported model relies on cross-validation over the training data.
- All five models find fewer than half of the leavers on the test set (recall between 0.16 and 0.47).

## Stack

Python 3.14 · pandas · scikit-learn · matplotlib/seaborn · Streamlit — see `requirements.txt` for exact versions.
