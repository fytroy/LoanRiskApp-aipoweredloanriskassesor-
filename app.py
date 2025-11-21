import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
# We load the dictionary containing both the model and the encoders
model_data = joblib.load('loan_risk_model.pkl')
model = model_data['model']
encoders = model_data['encoders']

# --- Page Config ---
st.set_page_config(page_title="Loan Risk AI", page_icon="🏦")

st.title("🏦 AI-Powered Loan Risk Assessor")
st.markdown("Enter applicant details below to predict credit default risk.")

# 2. Create Input Fields
# We split the layout into 3 columns for a cleaner look
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=1000, value=50000)
    emp_length = st.number_input("Employment Length (Years)", min_value=0.0, value=5.0)
    
with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=100, value=10000)
    interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, value=10.0)
    credit_hist = st.number_input("Credit History Length (Years)", min_value=0, value=4)

with col3:
    # For categorical features, we must use the exact same options the model saw during training
    home_ownership = st.selectbox("Home Ownership", options=encoders['person_home_ownership'].classes_)
    loan_intent = st.selectbox("Loan Intent", options=encoders['loan_intent'].classes_)
    loan_grade = st.selectbox("Loan Grade", options=encoders['loan_grade'].classes_)
    default_history = st.selectbox("Previous Default?", options=encoders['cb_person_default_on_file'].classes_)

# 3. Process Logic
# Calculate the 'loan_percent_income' automatically (Loan / Income)
# The model expects this feature, but it's better to calculate it than ask the user.
loan_percent_income = loan_amount / income if income > 0 else 0

# 4. Predict Button
if st.button("Analyze Risk", type="primary"):
    
    # Prepare the raw input data
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [interest_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [default_history],
        'cb_person_cred_hist_length': [credit_hist]
    })

    # Encode categorical text -> numbers using our saved encoders
    for col, le in encoders.items():
        input_data[col] = le.transform(input_data[col])

    # Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] # Probability of being '1' (Default)

    # 5. Display Results
    st.divider()
    
    if prediction == 0:
        st.success(f"✅ **APPROVED** (Low Risk)")
        st.markdown(f"Probability of Default: **{probability:.1%}**")
        st.markdown("The applicant shows a strong repayment profile.")
    else:
        st.error(f"🚨 **REJECTED** (High Risk)")
        st.markdown(f"Probability of Default: **{probability:.1%}**")
        st.markdown("The model has flagged this application as high risk.")