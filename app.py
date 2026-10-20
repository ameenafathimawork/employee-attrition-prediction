import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load Model and Features ---
# Ensure 'random_forest_model.joblib' and 'model_features.joblib' are in the same directory.
try:
    rf_model = joblib.load('random_forest_model.joblib')
    MODEL_FEATURES = joblib.load('model_features.joblib')
    st.sidebar.success("Model and features loaded successfully.")
except FileNotFoundError:
    st.error("❌ Error: Model or features file not found. Ensure 'train_model.py' was run.")
    st.stop()

# --- 2. Define Feature Mappings and Defaults ---

# Mappings for user-friendly select boxes (using numerical codes from your data)
TRAVEL_MAP = {0: 'No Travel', 1: 'Travel Rarely', 2: 'Travel Frequently'}
DEPT_MAP = {0: 'Human Resources', 1: 'Research & Development', 2: 'Sales'}
EDU_MAP = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
SATISFACTION_MAP = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
GENDER_MAP = {0: 'Male', 1: 'Female'}
MARITAL_MAP = {0: 'Divorced', 1: 'Single', 2: 'Married'}
OVERTIME_MAP = {0: 'No', 1: 'Yes'}
WLB_MAP = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}

# Utility function to convert user choice back to model's numerical code
def reverse_map(m):
    return {v: k for k, v in m.items()}

# 🚨 CRITICAL UPDATE: This dictionary must contain ALL 30 features.
# It holds the default mean/mode values for features not collected via the UI.
DEFAULT_DATA = {
    'Age': 37, 
    'BusinessTravel': 2, 
    'DailyRate': 802, 
    'Department': 1,
    'DistanceFromHome': 9, 
    'Education': 3, 
    'EducationField': 2, # Default for features not in UI
    'EnvironmentSatisfaction': 3, 
    'Gender': 0, 
    'HourlyRate': 66, # Default for features not in UI
    'JobInvolvement': 3, # Default for features not in UI
    'JobLevel': 2, 
    'JobRole': 6, # Default for features not in UI
    'JobSatisfaction': 3, # Default for features not in UI
    'MaritalStatus': 1, 
    'MonthlyIncome': 6502, 
    'MonthlyRate': 14313, # Default for features not in UI
    'NumCompaniesWorked': 3, # Default for features not in UI
    'OverTime': 0, 
    'PercentSalaryHike': 15, # Default for features not in UI
    'PerformanceRating': 3, # Default for features not in UI
    'RelationshipSatisfaction': 3, # Default for features not in UI
    'StockOptionLevel': 1, # Default for features not in UI
    'TotalWorkingYears': 11,
    'TrainingTimesLastYear': 3, # Default for features not in UI
    'WorkLifeBalance': 3, 
    'YearsAtCompany': 7,
    'YearsInCurrentRole': 4, # Default for features not in UI
    'YearsSinceLastPromotion': 2, # Default for features not in UI
    'YearsWithCurrManager': 4 # Default for features not in UI
}

# Ensure the default dictionary only contains the features the model expects (30 total)
DEFAULT_DATA = {k: DEFAULT_DATA[k] for k in MODEL_FEATURES}

# --- 3. Streamlit UI Design ---
st.title("👨‍💼 Employee Attrition Prediction App")
st.markdown("Use the controls below to input employee data and predict the likelihood of attrition.")

# Create a sidebar for user input
st.sidebar.header("Employee Input Features")
st.sidebar.markdown("Set the key attributes for prediction:")

# Create input widgets for the key features
with st.sidebar.form("input_form"):
    
    # --- Categorical/Ordinal Inputs ---
    bt_selection = st.selectbox("Business Travel", list(TRAVEL_MAP.values()))
    dept_selection = st.selectbox("Department", list(DEPT_MAP.values()))
    gender_selection = st.selectbox("Gender", list(GENDER_MAP.values()))
    marital_selection = st.selectbox("Marital Status", list(MARITAL_MAP.values()))
    overtime_selection = st.selectbox("OverTime", list(OVERTIME_MAP.values()))
    
    env_sat_selection = st.select_slider("Environment Satisfaction", options=list(SATISFACTION_MAP.values()))
    wlb_selection = st.select_slider("Work-Life Balance", options=list(WLB_MAP.values()))
    
    # --- Numerical/Slider Inputs ---
    age = st.slider("Age", 18, 60, 35)
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=6500)
    years_at_company = st.slider("Years At Company", 0, 40, 5)
    total_working_years = st.slider("Total Working Years", 0, 40, 10)

    # Submit Button
    submitted = st.form_submit_button("Predict Attrition")

# --- 4. Prediction Logic ---

if submitted:
    # 1. Create a dictionary for user-supplied inputs (mapped to numerical codes)
    user_inputs = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'YearsAtCompany': years_at_company,
        'TotalWorkingYears': total_working_years,
        'BusinessTravel': reverse_map(TRAVEL_MAP)[bt_selection],
        'Department': reverse_map(DEPT_MAP)[dept_selection],
        'Gender': reverse_map(GENDER_MAP)[gender_selection],
        'MaritalStatus': reverse_map(MARITAL_MAP)[marital_selection],
        'OverTime': reverse_map(OVERTIME_MAP)[overtime_selection],
        'EnvironmentSatisfaction': reverse_map(SATISFACTION_MAP)[env_sat_selection],
        'WorkLifeBalance': reverse_map(WLB_MAP)[wlb_selection]
    }
    
    # 2. Populate the full 30-feature vector using defaults and then overwrite with user inputs
    input_data = DEFAULT_DATA.copy()
    input_data.update(user_inputs)
    
    # 3. Create DataFrame and ensure column order matches training data using MODEL_FEATURES
    # This is the line that fixes the feature mismatch error!
    features_df = pd.DataFrame([input_data])[MODEL_FEATURES] 
    
    # 4. Make prediction
    prediction = rf_model.predict(features_df)[0]
    
    # 5. Display results
    st.subheader("Prediction Result")
    
    if prediction == 1:
        st.error("⚠️ Prediction: **High Risk of Attrition**")
        st.markdown("Based on the input features, the model predicts the employee is likely to leave the company.")
    else:
        st.success("✅ Prediction: **Low Risk of Attrition**")
        st.markdown("Based on the input features, the model predicts the employee is likely to stay with the company.")
    
    # Display the input data for reference
    st.subheader("Input Data Used for Prediction (30 Features)")
    st.dataframe(features_df.T.rename(columns={0: 'Feature Value'}))