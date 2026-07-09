import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION & THEME (UI/UX Layer)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ReadyNest Credit Risk Analyzer",
    page_icon="💳",
    layout="centered"
)

# Custom structural styling for a clean design
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    h1 { color: #0f4c81; font-weight: 700; }
    .stButton>button { background-color: #0f4c81; color: white; border-radius: 6px; width: 100%; }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Credit Risk & Loan Default Analyzer")
st.write("Adjust the borrower metrics below to test the predictive model in real-time.")
st.markdown("---")

# ------------------------------------------------------------------
# 2. CACHED MODEL INITIALIZATION (Performance Layer)
# ------------------------------------------------------------------
@st.cache_resource
def load_and_train_base_model():
    # Load and clean a subset of the data quickly to train the backend model for the UI
    df = pd.read_csv('Loan_default.csv')
    if 'LoanID' in df.columns: df.drop(columns=['LoanID'], inplace=True)
    
    # Quick clean
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in numeric_cols: df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols: df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encodings
    encoders = {}
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        
    X = df_encoded.drop(columns=['Default'])
    y = df_encoded['Default']
    
    # Optimized training setup for live deployment interface
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1,class_weight='balanced')
    model.fit(X, y)
    
    return model, encoders, X.columns

rf_model, encoders, model_features = load_and_train_base_model()

# ------------------------------------------------------------------
# 3. INTERACTIVE INPUT FORMS (UI Component Layer)
# ------------------------------------------------------------------
st.subheader("👤 Borrower Demographic & Stability")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 80, 35)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
with col2:
    income = st.number_input("Annual Income ($)", min_value=0, value=50000, step=1000)
    employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
with col3:
    months_employed = st.slider("Months Employed", 0, 120, 36)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

st.markdown("---")
st.subheader("💰 Loan Logistics & Liability parameters")
col4, col5, col6 = st.columns(3)

with col4:
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=15000, step=500)
    loan_term = st.slider("Loan Term (Months)", 12, 60, 36)
with col5:
    interest_rate = st.slider("Interest Rate (%)", 1.0, 25.0, 8.5, step=0.1)
    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", 0.0, 1.0, 0.3, step=0.01)
with col6:
    credit_score = st.slider("Credit Score", 300, 850, 650)
    num_credit_lines = st.slider("Open Credit Lines", 1, 15, 3)

col7, col8, col9 = st.columns(3)
with col7:
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
with col8:
    has_mortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
    has_cosigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])
with col9:
    has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

# ------------------------------------------------------------------
# 4. DATA TRANSFORMATION & LIVE PREDICTION
# ------------------------------------------------------------------
if st.button("Run Risk Prediction Diagnostics"):
    # Structure inputs into a dictionary matching original features
    input_data = {
        'Age': age, 'Income': income, 'LoanAmount': loan_amount, 'CreditScore': credit_score,
        'MonthsEmployed': months_employed, 'NumCreditLines': num_credit_lines, 'InterestRate': interest_rate,
        'LoanTerm': loan_term, 'DTIRatio': dti_ratio, 'Education': education, 'EmploymentType': employment_type,
        'MaritalStatus': marital_status, 'HasMortgage': has_mortgage, 'HasDependents': has_dependents,
        'LoanPurpose': loan_purpose, 'HasCoSigner': has_cosigner
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encode values using the backend dictionary maps
    for col in encoders:
        if col in input_df.columns:
            try:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
            except Exception:
                input_df[col] = 0 # Default fallback safe structural index
                
    # Align order exactly with training features layout
    input_df = input_df[model_features]
    
    # Generate direct classification inference vector
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Analytical Diagnostic Results")
    
    if prediction == 0:
        st.success(f"🟩 **Decision: APPROVE LOAN** \n\nThe model predicts a low structural default probability of **{probability*100:.2f}%**.")
    else:
        st.error(f"🟥 **Decision: DENY LOAN (High Risk)** \n\nThe model flags this account as risky with a default probability of **{probability*100:.2f}%**.")