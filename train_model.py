import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- Starting Model Training ---")

# 1. Load Data
# We read the CSV file into a Pandas DataFrame
df = pd.read_csv('data/credit_risk_dataset.csv')
print(f"Data Loaded: {df.shape[0]} rows")

# 2. Data Cleaning & Preprocessing
# Fill missing income/employment with 0 (Simple imputation)
df['person_emp_length'] = df['person_emp_length'].fillna(0)
df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].mean())

# Define our features (inputs) and target (output)
# We want to predict 'loan_status' (0 = Paid, 1 = Default)
target = 'loan_status'
features = ['person_age', 'person_income', 'person_home_ownership', 
            'person_emp_length', 'loan_intent', 'loan_grade', 
            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
            'cb_person_default_on_file', 'cb_person_cred_hist_length']

X = df[features].copy()
y = df[target]

# Convert Text Columns to Numbers (Encoding)
# ML models only understand numbers, so we convert "RENT", "OWN" -> 0, 1
encoders = {} # We save these to use in our app later!
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# 3. Split Data
# 80% for training, 20% for testing to see how good the model is
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model (Random Forest)
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save the Brain
# We save the Model AND the Encoders so the App can use them
model_data = {
    "model": model,
    "encoders": encoders
}
joblib.dump(model_data, 'loan_risk_model.pkl')
print("Model saved to 'loan_risk_model.pkl'")