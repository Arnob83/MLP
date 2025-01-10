import sqlite3
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import requests
import os
import shap  # Import SHAP

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

# Prediction function
@st.cache_data
def prediction(Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term):
    # Map user inputs to numeric values
    Education = 0 if Education == "Graduate" else 1
    Credit_History = 0 if Credit_History == "Unclear Debts" else 1

    # Create input data for the model (Before scaling)
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

# Function for SHAP explanation and visualization
def explain_with_shap(input_data):
    # Create a model prediction function for SHAP (MLPClassifier)
    def model_predict(data):
        return classifier.predict_proba(data)

    # SHAP Explainer requires a properly scaled input, so we'll pass the entire input data after scaling
    explainer = shap.Explainer(model_predict, input_data)  # Using the model's predict function

    # Calculate SHAP values for the input data
    shap_values = explainer(input_data)

    # Create a SHAP bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_data, plot_type="bar", show=False)  # 'show=False' prevents automatic rendering

    # Display SHAP plot in Streamlit
    st.pyplot(fig)

# Main Streamlit app
def main():
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

    # Prediction and SHAP visualization
    if st.button("Predict"):
        result, input_data, probabilities = prediction(
            Credit_History, Education, ApplicantIncome, CoapplicantIncome, Loan_Amount_Term
        )

        # Display the prediction
        if result == "Approved":
            st.success(f"Your loan is Approved! (Probability: {probabilities[0][1]:.2f})")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0][0]:.2f})")

        st.subheader("Input Data (Scaled)")
        st.write(input_data)

        # Show SHAP bar plot for feature importance
        explain_with_shap(input_data)

if __name__ == '__main__':
    main()
