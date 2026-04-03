import joblib
import streamlit as st
import os
from PIL import Image
import numpy as np

st.set_page_config(page_title="Loan Classification", layout="wide")

pipeline_path = 'pipeline2.pkl'
model = joblib.load(pipeline_path)

image_sidebar = Image.open('images.png')
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Loan Classification')

def get_user_input():
    age = st.sidebar.number_input("Enter Age: ", min_value=0, max_value=120, value=30)
    income = st.sidebar.number_input("Enter Income: ", min_value=0, value=50000)
    loan_amount = st.sidebar.number_input("Enter Loan Amount: ", min_value=0, value=10000)
    credit_score = st.sidebar.number_input("Enter Credit Score: ", min_value=300, value=650)
    months_employed = st.sidebar.number_input("Enter Months Employed: ", min_value=0, value=12)
    num_credit_lines = st.sidebar.number_input("Enter Number of Credit Lines: ", min_value=0, value=3)
    interest_rate = st.sidebar.number_input("Enter Interest Rate: ", min_value=0.0, value=10.0)
    loan_term = st.sidebar.number_input("Enter Loan Term (in months): ", min_value=0, value=3)
    dti_ratio = st.sidebar.number_input("Enter Debt-to-Income Ratio: ", min_value=0.0, value=3.0)
    
    education = st.sidebar.selectbox('Enter Education',['Bachelors', 'Masters', 'PhD', 'High School'])
    marital_status = st.sidebar.selectbox('Enter Marital Status',['Single', 'Married', 'Divorced'])
    has_mortgage = st.sidebar.selectbox('Has Mortgage?',['Yes', 'No'])
    has_mortgage = 1 if has_mortgage == "Yes" else 0
    has_dependents = st.sidebar.selectbox('Has Dependents?',['Yes', 'No'])
    has_dependents = 1 if has_dependents == "Yes" else 0
    loan_purpose = st.sidebar.selectbox('Enter Loan Purpose',['Home', 'Car', 'Education', 'Personal'])
    has_cosigner = st.sidebar.selectbox('Has Co-Signer?',['Yes', 'No'])
    has_cosigner = 1 if has_cosigner == "Yes" else 0
    
    user_data = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,

        f'Education_{education}': 1,
        f'MaritalStatus_{marital_status}': 1,
        'HasMortgage': has_mortgage,
        'HasDependents': has_dependents,
        f'LoanPurpose_{loan_purpose}': 1,
        'HasCoSigner': has_cosigner
    }
    return user_data

st.markdown("<h1 style='text-align: center;'>Loan Default Prediction App</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.5, 2, 0.5])
with col2:
    image_banner = Image.open('loan.png').resize((500, 200))  # Replace with your image file
    st.image(image_banner)

left_col, space, right_col = st.columns([1, 0.2, 1])

with right_col:
    st.subheader("Predict Loan Default")

    user_data = get_user_input()

    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    features = [
        'Age',
        'Income',
        'LoanAmount',
        'CreditScore',
        'MonthsEmployed',
        'NumCreditLines',
        'InterestRate',
        'LoanTerm',
        'DTIRatio',
        'HasMortgage',
        'HasDependents',
        'HasCoSigner',
        'Education_Bachelors',
        'Education_High School',
        'Education_Masters',
        'Education_PhD',
        'EmploymentType_Full time', 
        'EmploymentType_Part-time',
        'EmploymentType_Self employed', 
        'EmploymentType_Unemployed',
        'MaritalStatus_Divorced',
        'MaritalStatus_Married',
        'MaritalStatus_Single',
        'LoanPurpose_Auto',
        'LoanPurpose_Business',
        'LoanPurpose_Education',
        'LoanPurpose_Home',
        'LoanPurpose_Other'
    ]

    if st.button("Predict Result"):
            input_array = prepare_input(user_data, features)
            prediction = model.predict(input_array)
            st.subheader("Predicted Result")
            st.write(prediction)

