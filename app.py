import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import numpy as np

# URLs for the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/MLP/main/Logistic_Regression_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/MLP/main/scaler.pkl"

# Download and save model and scaler files locally
if not os.path.exists("Logistic_Regression_model.pkl"):
    model_response = requests.get(model_url)
    with open("Logistic_Regression_model.pkl", "wb") as file:
        file.write(model_response.content)

if not os.path.exists("scaler.pkl"):
    scaler_response = requests.get(scaler_url)
    with open("scaler.pkl", "wb") as file:
        file.write(scaler_response.content)

# Load the trained model
with open("Logistic_Regression_model.pkl", "rb") as model_file:
    classifier = pickle.load(model_file)

# Load the Min-Max scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gender TEXT,
        married TEXT,
        dependents INTEGER,
        self_employed TEXT,
        loan_amount REAL,
        property_area TEXT,
        credit_history TEXT,
        education TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount_term REAL,
        result TEXT
    )
    """)
    conn.commit()
    conn.close()

# Save prediction data to the database
def save_to_database(gender, married, dependents, self_employed, loan_amount, property_area, 
                     credit_history, education, applicant_income, coapplicant_income, 
                     loan_amount_term, result):
    conn = sqlite3.connect("loan_data.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO loan_predictions (
        gender, married, dependents, self_employed, loan_amount, property_area, 
        credit_history, education, applicant_income, coapplicant_income, loan_amount_term, result
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (gender, married, dependents, self_employed, loan_amount, property_area, 
          credit_history, education, applicant_income, coapplicant_income, 
          loan_amount_term, result))
    conn.commit()
    conn.close()

# Prediction function
@st.cache_data
def prediction(Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Feature mapping
    Education = 0 if Education == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data with the correct feature order
    input_data = pd.DataFrame(
        [[Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Scale specified features using Min-Max scaler
    columns_to_scale = ["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    scaled_input = input_data.copy()
    scaled_input[columns_to_scale] = scaler.transform(scaled_input[columns_to_scale])

    # Ensure feature names match model training
    scaled_input = scaled_input[classifier.feature_names_in_]

    # Model prediction
    prediction = classifier.predict(scaled_input)
    probabilities = classifier.predict_proba(scaled_input)

    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data, scaled_input, probabilities

# Feature importance function using coefficients
def plot_feature_importance():
    feature_names = classifier.feature_names_in_
    coefficients = classifier.coef_[0]

    # Absolute value of coefficients for feature importance
    importance = np.abs(coefficients)

    # Create a DataFrame to show feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance,
        'Coefficient': coefficients  # Adding coefficients to the plot for more details
    }).sort_values(by='Importance', ascending=False)

    # Plotting the feature importance
    st.subheader("Feature Importance")
    st.write(feature_importance_df)

    # Creating the horizontal bar plot
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the size of the plot as per your need
    colors = ["green" if coef > 0 else "red" for coef in feature_importance_df['Coefficient']]  # Color bars based on coefficient sign
    bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)

    # Annotating the bars with coefficient values
    for bar, coef in zip(bars, feature_importance_df['Coefficient']):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{coef:.4f}", va='center')

    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance based on Logistic Regression Coefficients')

    # Show the plot in Streamlit
    st.pyplot(fig)

# Main Streamlit app
def main():
    init_db()

    st.title("Loan Prediction ML App")

    Gender = st.selectbox("Gender", ("Male", "Female"))
    Married = st.selectbox("Married", ("Yes", "No"))
    Dependents = st.number_input("Dependents (0-5)", min_value=0, max_value=5, step=1)
    Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0)
    Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semi-urban"))
    Credit_History = st.selectbox("Credit History", ("Unclear Debts", "Clear Debts"))
    Education = st.selectbox("Education", ("Undergraduate", "Graduate"))
    ApplicantIncome = st.number_input("Applicant's Yearly Income", min_value=0.0)
    CoapplicantIncome = st.number_input("Co-applicant's Yearly Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=0.0)

    if st.button("Predict"):
        result, original_data, scaled_data, probabilities = prediction(
            Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term
        )

        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        if result == "Approved":
            st.success(f"Your loan is Approved! (Probability: {probabilities[0][1]:.2f})")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0][0]:.2f})")

        plot_feature_importance()

    if st.button("Download Database"):
        if os.path.exists("loan_data.db"):
            with open("loan_data.db", "rb") as f:
                st.download_button(
                    label="Download SQLite Database",
                    data=f,
                    file_name="loan_data.db",
                    mime="application/octet-stream"
                )
        else:
            st.error("Database file not found.")

if __name__ == '__main__':
    main()
