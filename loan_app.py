import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and clean data
data = pd.read_csv(r"C:\Users\Rohit\Documents\Rohit\Projects\MLOps Project\loan_approval_dataset.csv")
data.drop(columns = ['loan_id'], inplace = True)
data.columns = data.columns.str.strip()

# Feature engineering
data['Assets'] = data.residential_assets_value + data.commercial_assets_value + data.luxury_assets_value + data.bank_asset_value
data.drop(columns = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'], inplace = True)

# Clean categorical columns
def clean_data(st):
    return st.strip()

data.education = data.education.apply(clean_data).replace(['Graduate', 'Not Graduate'], [1,0])
data.self_employed = data.self_employed.apply(clean_data).replace(['No', 'Yes'], [0,1])
data.loan_status = data.loan_status.apply(clean_data).replace(['Approved', 'Rejected'], [1,0])

# Split data
input_data = data.drop(columns = ['loan_status'])
output_data = data['loan_status']
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Tuned Accuracy:", accuracy)

# Save model
joblib.dump(best_model, 'model.pkl')

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
    model = joblib.load('model.pkl')
    pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, annual_income, loan_amount, loan_duration, cibil_score, assets]], 
                             columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
    
    prediction = model.predict(pred_data)
    if prediction[0] == 1:
        st.success('Loan Is Approved')
    else:
        st.error('Loan is Rejected')

    # Display prediction info
    st.write(pred_data)
    st.write('Loan Amount:', loan_amount)
    st.write('Loan Duration:', loan_duration)
    st.write('Cibil Score:', cibil_score)
    st.write('Assets:', assets)
    st.write('Annual Income:', annual_income)
    st.write('No of Dependents:', no_of_dep)
    st.write('Education:', grad)
    st.write('Self Employed:', self_emp)
    st.write('Loan Status:', prediction[0])
    st.write('Model Accuracy:', accuracy)
    st.write('Model Name: Random Forest Classifier')
    st.write('Model Type: Classification')
