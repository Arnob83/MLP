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

# Explanation function using SHAP
def explain_prediction(input_data, final_result):
    # Define a model prediction function for SHAP (this function is required)
    def model_predict(data):
        return classifier.predict_proba(data)

    # Ensure input_data is in the correct 2D shape for SHAP
    input_data_for_shap = input_data.values.reshape(1, -1)

    # Initialize SHAP explainer with the model prediction function
    explainer = shap.Explainer(model_predict, input_data_for_shap)

    # Get the SHAP values for the input data
    shap_values = explainer(input_data_for_shap)
    shap_values_for_input = shap_values[0].values

    # Get feature names for the explanation
    feature_names = input_data.columns

    # Explanation text for the user
    explanation_text = f"**Why your loan is {final_result}:**\n\n"
    for feature, shap_value in zip(feature_names, shap_values_for_input):
        explanation_text += (
            f"- **{feature}**: {'Positive' if shap_value > 0 else 'Negative'} contribution with a SHAP value of {shap_value:.2f}\n"
        )

    if final_result == 'Rejected':
        explanation_text += "\nThe loan was rejected because the negative contributions outweighed the positive ones."
    else:
        explanation_text += "\nThe loan was approved because the positive contributions outweighed the negative ones."

    # Plotting the SHAP bar plot for feature contributions
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, shap_values_for_input, color=["green" if val > 0 else "red" for val in shap_values_for_input])
    plt.xlabel("SHAP Value (Impact on Prediction)")
    plt.ylabel("Features")
    plt.title("Feature Contributions to Prediction")
    plt.tight_layout()

    return explanation_text, plt

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

        # Display the prediction result
        if result == "Approved":
            st.success(f"Your loan is Approved! (Probability: {probabilities[0][1]:.2f})")
        else:
            st.error(f"Your loan is Rejected! (Probability: {probabilities[0][0]:.2f})")

        st.subheader("Input Data (Scaled)")
        st.write(input_data)

        # Show SHAP explanation and plot for feature importance
        explanation_text, shap_plot = explain_prediction(input_data, result)
        st.subheader("Explanation of Prediction")
        st.write(explanation_text)
        st.pyplot(shap_plot)  # Display the SHAP bar plot

if __name__ == '__main__':
    main()
