import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the single pipeline file
try:
    full_pipeline = joblib.load('full_pipeline.pkl')
except FileNotFoundError:
    st.error("Pipeline file not found. Please ensure 'full_pipeline.pkl' is in the same directory.")
    st.stop()


# --- Streamlit Frontend ---
st.set_page_config(page_title="Employee Attrition Predictor")

st.title("Employee Attrition Prediction 📊")
st.markdown("This version uses a robust Scikit-learn Pipeline.")

# Sidebar for Input Features
st.sidebar.header("Employee Characteristics")

# --- Streamlit Sliders & Selectboxes for User Input ---

# Demographics & Environment
age = st.sidebar.slider("Age", 18, 60, 30)
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
marital_status = st.sidebar.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
env_satisfaction = st.sidebar.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement (1=Low, 4=High)", 1, 4, 3)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work Life Balance (1=Bad, 4=Good)", 1, 4, 3)
education = st.sidebar.slider("Education Level (1=Below College, 5=Doctoral)", 1, 5, 3)

# Job & Compensation
department = st.sidebar.selectbox("Department", ['Research & Development', 'Sales', 'Human Resources'])
job_role = st.sidebar.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 
                                            'Manufacturing Director', 'Healthcare Representative', 
                                            'Manager', 'Sales Representative', 'Research Director', 
                                            'Human Resources'])
job_level = st.sidebar.slider("Job Level (1=Entry, 5=Executive)", 1, 5, 2)
monthly_income = st.sidebar.slider("Monthly Income (USD)", 1000, 20000, 6500)
percent_salary_hike = st.sidebar.slider("Percent Salary Hike", 11, 25, 14)

# Time & Travel
over_time = st.sidebar.selectbox("Works Overtime?", ['No', 'Yes'])
business_travel = st.sidebar.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)

# --- Assemble ALL necessary features for prediction (including defaults) ---
input_data = pd.DataFrame({
    'Age': [age],
    'BusinessTravel': [business_travel],
    'DailyRate': [800],
    'Department': [department],
    'DistanceFromHome': [10],
    'Education': [education], 
    'EducationField': ['Life Sciences'],
    'EnvironmentSatisfaction': [env_satisfaction],
    'Gender': [gender],
    'HourlyRate': [65],
    'JobInvolvement': [job_involvement],
    'JobLevel': [job_level],
    'JobRole': [job_role],
    'JobSatisfaction': [job_satisfaction],
    'MaritalStatus': [marital_status],
    'MonthlyIncome': [monthly_income],
    'MonthlyRate': [15000],
    'NumCompaniesWorked': [1],
    'OverTime': [over_time],
    'PercentSalaryHike': [percent_salary_hike],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [3],
    'StockOptionLevel': [1],
    'TotalWorkingYears': [10],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [work_life_balance],
    'YearsAtCompany': [years_at_company],
    'YearsInCurrentRole': [4],
    'YearsSinceLastPromotion': [1],
    'YearsWithCurrManager': [3],
})


# --- Prediction Logic ---
if st.button("Predict Attrition"):
    
    # The pipeline handles all preprocessing internally
    try:
        prediction_proba = full_pipeline.predict_proba(input_data)
        attrition_prob = prediction_proba[0][1] # Probability of 'Yes' (class 1)
        prediction = full_pipeline.predict(input_data)[0]
    except Exception as e:
        # If any error persists, provide a clear message.
        st.error(f"Prediction error. Please check your input values. The underlying error is: {e}")
        st.stop()
        
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error(f"**High Attrition Risk (Employee Likely to Leave)**")
    else:
        st.success(f"**Low Attrition Risk (Employee Likely to Stay)**")
        
    st.metric(label="Probability of Attrition (Yes)", value=f"{attrition_prob*100:.2f}%")
    
    st.balloons()