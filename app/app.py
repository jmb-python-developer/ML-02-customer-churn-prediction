from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# The exported model: the notebook's best estimator, fitted on the training data
model = joblib.load(Path(__file__).resolve().parent.parent / "model_exports" / "best_model.joblib")

# Page composing
st.title("Customer Churn Predictor")

credit_score = st.slider("Credit score", 350, 850, 650)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 18, 92, 40)
tenure = st.slider("Tenure (years as a customer)", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=0.0, step=1000.0)
num_of_products = st.selectbox("Number of products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has a credit card?", ["Yes", "No"])
is_active_member = st.selectbox("Is an active member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated salary", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)

if st.button("Predict"):
    # Same encoding as in the notebook, since it was applied before the model was trained:
    # Gender by OrdinalEncoder (alphabetical order) and Geography as one-hot columns
    input_data = pd.DataFrame(
        [[
            credit_score,
            1.0 if gender == "Male" else 0.0,
            age,
            tenure,
            balance,
            num_of_products,
            1 if has_cr_card == "Yes" else 0,
            1 if is_active_member == "Yes" else 0,
            estimated_salary,
            geography == "Germany",
            geography == "Spain",
            geography == "France",
        ]],
        # The exact names and order used in training
        columns=[
            "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
            "IsActiveMember", "EstimatedSalary", "Geography_Germany", "Geography_Spain", "Geography_France",
        ],
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("This customer is likely to leave.")
    else:
        st.success("This customer is likely to stay.")
