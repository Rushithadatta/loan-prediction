import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details to predict loan approval status.")

# User inputs
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (months)", min_value=0)
Credit_History = st.selectbox("Credit History", [0, 1])

Gender_Male = st.selectbox("Gender", ["Male", "Female"]) == "Male"
Married_Yes = st.selectbox("Married", ["Yes", "No"]) == "Yes"
Education_Not_Graduate = st.selectbox("Education", ["Graduate", "Not Graduate"]) == "Not Graduate"
Self_Employed_Yes = st.selectbox("Self Employed", ["Yes", "No"]) == "Yes"
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Create input dataframe (must match training columns)
input_data = pd.DataFrame([{
    'ApplicantIncome': ApplicantIncome,
    'CoapplicantIncome': CoapplicantIncome,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Gender_Male': int(Gender_Male),
    'Married_Yes': int(Married_Yes),
    'Dependents_1': 0,
    'Dependents_2': 0,
    'Dependents_3+': 0,
    'Education_Not Graduate': int(Education_Not_Graduate),
    'Self_Employed_Yes': int(Self_Employed_Yes),
    'Property_Area_Semiurban': int(Property_Area == "Semiurban"),
    'Property_Area_Urban': int(Property_Area == "Urban")
}])

# Predict
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
