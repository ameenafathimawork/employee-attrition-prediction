import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load Data
file_path = "Employee_Attrition.csv.csv"
df = pd.read_csv(file_path)

# 2. Preprocessing & Feature Definition
# Drop non-predictive/constant/ID columns
df_cleaned = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'])

X = df_cleaned.drop('Attrition', axis=1)
y = df_cleaned['Attrition'] 

# --- FINAL CORRECT FEATURE LISTS ---
# 23 Features to be scaled (Continuous + Ordinal Integers)
numerical_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 
    'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 
    'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
    'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 
    'StockOptionLevel', 'WorkLifeBalance'
]

# 7 Categorical string columns (to be One-Hot Encoded)
categorical_string_cols = [
    'BusinessTravel', 'Department', 'EducationField', 'Gender', 
    'JobRole', 'MaritalStatus', 'OverTime'
]

# 3. Create Preprocessing Pipeline (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        # Transformer 1: Scale all 23 numerical/ordinal integer features.
        ('num', StandardScaler(), numerical_cols),
        
        # Transformer 2: One-hot encode ONLY the 7 string columns.
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_string_cols)
    ],
    remainder='drop' # Explicitly drop any feature not listed
)

# 4. Create Full ML Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Split Data and Train Model (using the raw X data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
full_pipeline.fit(X_train, y_train)

# 6. Save the Full Pipeline
joblib.dump(full_pipeline, 'full_pipeline.pkl')
print("--- SUCCESS: NEW full_pipeline.pkl SAVED ---")