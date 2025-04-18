
# Loan Approval Prediction App

A Machine Learning web application built with Streamlit to predict whether a loan will be approved or rejected based on user input.

## 🔍 Project Overview

This project performs the following steps:
- Data cleaning and preprocessing
- Feature engineering (asset aggregation)
- Label encoding of categorical variables
- Train-test split
- Hyperparameter tuning using GridSearchCV
- Model training using Random Forest Classifier
- Deployment using Streamlit

## 🚀 Tech Stack
- Python
- Pandas, Scikit-learn, Joblib
- Streamlit
- GridSearchCV for hyperparameter tuning

## 📂 How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run loan_app.py
   ```

## ✅ Model Details

- Model: Random Forest Classifier
- Accuracy: Tuned using GridSearchCV with 5-fold Cross Validation

## 📄 Output

The app provides predictions for loan approval along with input summary and model performance.

---

**Author:** Rohit  
