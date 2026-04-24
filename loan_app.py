import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


@st.cache_data
def load_accuracy():
    return joblib.load("model_accuracy.pkl")

# Streamlit UI
st.header('Loan Approval Prediction App')

no_of_dep = st.slider('Choose No. of Dependents', 0, 5, 1)
grad = st.selectbox('Choose Education', ['Graduated', 'Not Graduated'])
self_emp = st.selectbox('Self Employed ?', ['Yes', 'No'])
annual_income = st.number_input('Choose Annual Income', min_value=0, max_value=5000000, step=1000, value=50000)
loan_amount = st.number_input('Choose Loan Amount', min_value=0, max_value=5000000, step=1000, value=200000)
loan_duration = st.slider('Choose Loan Duration', 1, 20, 10)
cibil_score = st.slider('Choose Cibil Score', 300, 900, 750)
assets = st.number_input('Choose Assets', min_value=0, max_value=5000000, step=1000, value=500000)

# Convert inputs
grad_s = 1 if grad == 'Graduated' else 0
emp_s = 0 if self_emp == 'No' else 1

# Load model and predict
if st.button("Predict"):
    model = load_model()
    accuracy = load_accuracy()
    pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, annual_income, loan_amount, loan_duration, cibil_score, assets]], 
                             columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
    
    prediction = model.predict(pred_data)
    if prediction[0] == 1:
        st.success('Loan Is Approved')
    else:
        st.error('Loan is Rejected')

    # Display prediction info without Streamlit's dataframe renderer.
    st.subheader("Input Summary")
    st.text(f"Loan Amount: {loan_amount}")
    st.text(f"Loan Duration: {loan_duration}")
    st.text(f"Cibil Score: {cibil_score}")
    st.text(f"Assets: {assets}")
    st.text(f"Annual Income: {annual_income}")
    st.text(f"No of Dependents: {no_of_dep}")
    st.text(f"Education: {grad}")
    st.text(f"Self Employed: {self_emp}")
    st.text(f"Loan Status: {prediction[0]}")
    st.text(f"Model Accuracy: {accuracy}")
    st.text("Model Name: Random Forest Classifier")
    st.text("Model Type: Classification")
