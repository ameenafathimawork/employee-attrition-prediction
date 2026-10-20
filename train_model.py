import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- Configuration ---
FILE_PATH = "Employee_Attrition.csv.csv"
MODEL_FILENAME = 'random_forest_model.joblib'
FEATURES_FILENAME = 'model_features.joblib'
RANDOM_STATE = 42

# --- 1. Load Data ---
df = pd.read_csv(FILE_PATH)

# --- 2. Corrected Preprocessing ---
# Drop non-predictive/constant/ID columns
columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
X = df.drop(columns=columns_to_drop + ['Attrition'], errors='ignore')
y = df['Attrition']

# All features are already numerical (int64). No encoding or scaling is needed.

# Store the final column names for the Streamlit app
encoded_feature_names = X.columns.tolist()

# --- 3. Split Data ---
# Use stratify=y to maintain the proportion of attrition cases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# --- 4. Train Model ---
print("Training Random Forest Classifier (using class_weight='balanced')...")
# Using class_weight='balanced' to help mitigate the class imbalance issue
model = RandomForestClassifier(
    n_estimators=200, 
    random_state=RANDOM_STATE, 
    class_weight='balanced',
    max_depth=10 
)
model.fit(X_train, y_train)
print("Training complete.")

# --- 5. Model Evaluation ---
y_pred = model.predict(X_test)
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred))

# --- 6. Save Model and Feature Names ---
# NOTE: Removed saving the scaler, as it's no longer used.
joblib.dump(model, MODEL_FILENAME)
joblib.dump(encoded_feature_names, FEATURES_FILENAME)

print(f"\nModel saved successfully as: {MODEL_FILENAME}")
print(f"Feature names saved successfully as: {FEATURES_FILENAME}")