import pandas as pd
import numpy as np
import os

# --- STEP 1: INITIALIZE AND READ DATA ---
file_name = 'Loan_default.csv'

if not os.path.exists(file_name):
    raise FileNotFoundError(f"[-] '{file_name}' not found in your current VS Code folder! Check your directory structure.")

print("[*] Reading dataset into VS Code memory...")
df = pd.read_csv(file_name)
print(f"[+] Initial Raw Matrix Shape: {df.shape}\n")


# --- STEP 2: HANDLE NULL VALUES (No fake data!) ---
print("[*] Stage 1: Imputing missing profiles with column medians/modes...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Handle numerical nulls using median
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Handle categorical nulls using mode
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])


# --- STEP 3: DROP ID COLUMNS (Zero Predictive Value) ---
if 'LoanID' in df.columns:
    df.drop(columns=['LoanID'], inplace=True)
    print("[-] Stage 2: Dropped 'LoanID' column to prevent model overfitting.")


# --- STEP 4: REMOVE ROW DUPLICATES ---
duplicate_count = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"[+] Stage 3: Identified and removed {duplicate_count} exact line duplicates.")


# --- STEP 5: OUTLIER MITIGATION VIA CLIPPING ---
print("[*] Stage 4: Handling extreme feature outliers via statistical clipping...")
outlier_targets = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'DTIRatio', 'MonthsEmployed']
for col in outlier_targets:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip outliers to preserve row data and target class balance
        df[col] = np.clip(df[col], lower_bound, upper_bound)

print("\n[+] Cleaning Complete! Baseline Matrix Profile:")
print(f"   - Cleaned Rows: {df.shape[0]} | Cleaned Columns: {df.shape[1]}")
print(f"   - Target Class Summary:\n{df['Default'].value_counts()}")