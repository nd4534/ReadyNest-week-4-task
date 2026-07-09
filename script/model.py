import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("[*] Starting Stage 3: Prediction Model Training...")

# --- NEW: Read the data first so 'df' exists inside this file ---
file_name = 'Loan_default.csv'

if not os.path.exists(file_name):
    raise FileNotFoundError(f"[-] '{file_name}' not found! Place model.py in the same folder as {file_name}.")

print("[*] Reading dataset into model memory...")
df = pd.read_csv(file_name)

# --- QUICK PRE-PROCESSING FOR MODEL BACKEND ---
# Drop unique ID right away to prevent artificial performance inflation
if 'LoanID' in df.columns:
    df.drop(columns=['LoanID'], inplace=True)

# Handle potential missing entries on the fly using column medians/modes
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 1. Prepare Categorical Encodings (Turn string text categories into numeric flags)
df_encoded = df.copy()
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# 2. Separate Features (X) and Target Label (y)
X = df_encoded.drop(columns=['Default'])
y = df_encoded['Default']

# 3. Train-Test Split (80% Training for building patterns, 20% Testing for strict verification)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[+] Active training matrix shape: {X_train.shape}")
print(f"[+] Active verification matrix shape: {X_test.shape}")

# 4. Initialize and Train Random Forest
print("[*] Training Random Forest Classifier (processing data patterns now)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 5. Model Evaluation Metrics
y_pred = rf_model.predict(X_test)

print("\n================ MODEL PERFORMANCE REPORT ================")
print(f"Overall Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Matrix:")
print(classification_report(y_test, y_pred, target_names=['Fully Paid (0)', 'Defaulted (1)']))
print("==========================================================")