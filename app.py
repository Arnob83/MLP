import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import shap

# URLs for the model and scaler files in your GitHub repository
model_url = "https://raw.githubusercontent.com/Arnob83/MLP/main/MLP_model.pkl"
scaler_url = "https://raw.githubusercontent.com/Arnob83/MLP/main/scaler.pkl"

# Download and save model and scaler files locally
if not os.path.exists("MLP_model.pkl"):
    model_response = requests.get(model_url)
    with open("MLP_model.pkl", "wb") as file:
        file.write(model_response.content)

if not os.path.exists("scaler.pkl"):
    scaler_response = requests.get(scaler_url)
    with open("scaler.pkl", "wb") as file:
        file.write(scaler_response.content)

# Load the trained model
with open("MLP_model.pkl", "rb") as model_file:
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
    # Map user inputs to numeric values
    Education = 0 if Education == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data for the model
    input_data = pd.DataFrame(
        [[Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term]],
        columns=["Credit_History", "Education_1", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    )

    # Scale specified features using Min-Max scaler
    columns_to_scale = ["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

    # Model prediction
    prediction = classifier.predict(input_data)
    probabilities = classifier.predict_proba(input_data)
    
    pred_label = 'Approved' if prediction[0] == 1 else 'Rejected'
    return pred_label, input_data, probabilities

# Explanation function
def explain_prediction(input_data, final_result):
    # Define a function that will return predictions (probabilities) for SHAP
    def model_predict(input_data):
        # Scale the input data as the model expects scaled values
        columns_to_scale = ["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]
        input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
        return classifier.predict_proba(input_data)

    # Create an explainer object using the model_predict wrapper
    explainer = shap.KernelExplainer(model_predict, shap.sample(input_data, 100))
    
    # Generate SHAP values for the input data
    shap_values = explainer.shap_values(input_data)

    # Get SHAP values for the first input instance
    feature_names = input_data.columns
    shap_values_for_input = shap_values[0][0]  # SHAP values for the first row

    explanation_text = f"**Why your loan is {final_result}:**\n\n"
    for feature, shap_value in zip(feature_names, shap_values_for_input):
        explanation_text += (
            f"- **{feature}**: {'Positive' if shap_value > 0 else 'Negative'} contribution with a SHAP value of {shap_value:.2f}\n"
        )

    if final_result == 'Rejected':
        explanation_text += "\nThe loan was rejected because the negative contributions outweighed the positive ones."
    else:
        explanation_text += "\nThe loan was approved because the positive contributions outweighed the negative ones."

    # Generate SHAP summary plot
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)
    plt.tight_layout()

    return explanation_text, plt

# Main Streamlit app
def main():
    # Initialize database
    init_db()

    # App layout
    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f4f6f9;
            border: 2px solid #e6e8eb;
            padding: 20px;
            border-radius: 10px;
        }
        .header {
            background-color: #4caf50;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .header h1 {
            color: white;
        }
        </style>
        <div class="main-container">
        <div class="header">
        <h1>Loan Prediction ML App</h1>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # User inputs
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

    # Prediction and database saving
    if st.button("Predict"):
        result, input_data, probabilities = prediction(
            Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term
        )

        # Save data to database
        save_to_database(Gender, Married, Dependents, Self_Employed, Loan_Amount, Property_Area, 
                         Credit_History, Education, ApplicantIncome, CoapplicantIncome, 
                         Loan_Amount_Term, result)

        # Display the prediction
        if result == "Approved":
            st.success(f"Your loan is Approved! (Probability: {probabilities[0][1]:.2f})")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0][0]:.2f})")

        st.subheader("Input Data (Scaled)")
        st.write(input_data)

        # Generate and display SHAP explanation
        explanation_text, shap_plot = explain_prediction(input_data, result)
        st.markdown(explanation_text)
        st.pyplot(shap_plot)

    # Download database button
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
