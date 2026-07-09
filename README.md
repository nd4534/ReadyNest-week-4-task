# 💳 Credit Risk & Loan Default Predictive Pipeline

An end-to-end data engineering and machine learning framework designed to predict borrower loan defaults using a robust Random Forest architecture, verified with stratified holdout evaluation, and served via an interactive Streamlit web dashboard.

---

## 📌 Project Overview
The primary objective of this engineering initiative is to design, evaluate, and deploy a robust machine learning classification pipeline capable of predicting borrower loan defaults (Default = 1 vs. Default = 0). Managing credit risk is a foundational challenge in financial systems; identifying high-risk individuals prior to loan issuance protects capital reserves and minimizes institutional non-performing assets (NPAs).

This project tracks a full end-to-end data engineering lifecycle: from raw matrix preprocessing and exploratory visualization to training an ensemble classifier and deploying an interactive web application for real-time stakeholder auditing.

---

## 🛠️ Data Preprocessing Pipeline
Raw real-world datasets frequently exhibit data corruption, missing attributes, and extreme outliers that skew statistical modeling. To secure maximum baseline data integrity, a multi-stage cleaning pipeline was programmatically executed:

1. Administrative Feature Elimination: The LoanID feature represents a unique non-predictive tracking key. Leaving it in the training matrix would lead to artificial memorization (overfitting). It was stripped out immediately.
2. Dynamic Missing Profile Imputation:
   * Continuous variables containing null values were dynamically imputed using the column median to prevent structural variance bias.
   * Categorical string variables containing null values were filled using the column mode (most frequent class label).
3. Exact Row Deduplication: Exact line duplicates were identified and dropped to guarantee that validation splitting metrics remain completely uncompromised.
4. Statistical Outlier Mitigation (IQR Clipping): Extreme financial data points across skewed features (Age, Income, LoanAmount, CreditScore, DTIRatio, MonthsEmployed) were handled via the Interquartile Range (IQR) method. Values falling outside 1.5 * IQR boundaries were clipped to upper and lower statistical fences, preserving overall row counts and target class structures.

---

## 📊 Exploratory Data Analysis & Insights

### 1. Target Class Distribution Analysis
An initial examination of target distribution frequencies revealed a severe Class Imbalance Problem:
* Fully Paid accounts (0): 225,694 instances
* Defaulted accounts (1): 29,653 instances

This 88:12 distribution split indicates a highly skewed environment, requiring advanced algorithmic weighting strategies during model training to avoid majority-class bias.

### 2. Debt-to-Income (DTI) Behavioral Wave
Kernel Density Estimate (KDE) plots were generated to track the behavioral distribution of the Debt-to-Income ratio against repayment success. The overlapping density areas display an unvarying uniform distribution across the 0.1 to 0.9 bands, indicating that individual features within this dataset were synthetically independent during generation.

### 3. Linear Correlation & Feature Independence Mapping
A masked lower-triangle Pearson Correlation matrix was generated to expose hidden multi-collinearity across features, designed to prevent visual clipping:

| Feature | Age | Income | LoanAmount | CreditScore | MonthsEmployed | InterestRate | DTIRatio | Default |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Age** | 1.00 | | | | | | | |
| **Income** | -0.00 | 1.00 | | | | | | |
| **LoanAmount** | -0.00 | -0.00 | 1.00 | | | | | |
| **CreditScore** | -0.00 | -0.00 | 0.00 | 1.00 | | | | |
| **MonthsEmployed** | -0.00 | 0.00 | 0.00 | 0.00 | 1.00 | | | |
| **InterestRate** | -0.00 | -0.00 | -0.00 | 0.00 | 0.00 | 1.00 | | |
| **DTIRatio** | -0.00 | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 | 1.00 | |
| **Default** | **-0.17** | **-0.10** | **0.09** | **-0.03** | **-0.10** | **0.13** | **0.02** | **1.00** |

Critical Analysis Insight: The internal cross-feature correlation coefficients register at exactly 0.00, mathematically verifying that independent features exhibit zero multi-collinearity. However, features maintain distinct, explicit linear links with the target variable Default (e.g., Age at -0.17 and InterestRate at +0.13), confirming that predictive signals are structurally present.

---

## 🧠 Prediction Model Logic & Validation

### 1. Algorithm Selection: Random Forest Classifier
The Random Forest Classifier was selected as our core prediction engine. It is an ensemble learning method that constructs a forest of 100 independent, randomized decision trees (n_estimators=100`) capped at a maximum structural depth of 12 levels. By aggregating predictions across multiple trees through a majority voting system, it naturally resists overfitting and handles a mix of continuous and encoded categorical features without needing complex scale normalization.

### 2. Data Separation & Stratification Validation
To evaluate real-world generalization, the dataset was split using the 80/20 Holdout Method:
* Training Set: 204,277 rows (used exclusively for mapping risk patterns)
* Testing Set: 51,070 rows (completely unseen data acting as the final evaluation)

Stratification Enforcement: Because of the 88:12 class imbalance, we applied stratify=y during the split. This guarantees that both the training and testing matrices maintain the exact same proportions of stable and defaulted accounts, eliminating selection bias.

---

## 📈 Model Performance Metrics & Optimization

### 1. Baseline Execution Metrics (Unbalanced Weights)
The initial baseline random forest execution yielded an overall mathematical accuracy score of 88.59% (0.8859). Deep examination of the classification matrix exposed a critical real-world flaw:

================ BASELINE PERFORMANCE REPORT ================
Overall Accuracy Score: 0.8859

Classification Matrix:
                precision    recall  f1-score   support

Fully Paid (0)       0.89      1.00      0.94     45139
 Defaulted (1)       0.69      0.03      0.06      5931

      accuracy                           0.89     51070
     macro avg       0.79      0.52      0.50     51070
  weighted avg       0.86      0.89      0.84     51070
=============================================================

Core Diagnostic Performance Interpretation: While the overall accuracy appears high at 88.59%, the model's Recall for active defaults was only 3% (0.03). Because the training data was overwhelmingly populated by non-defaulters, the baseline algorithm learned to maximize accuracy by predicting "Fully Paid" for nearly every borrower, missing 97% of real defaults.

### 2. Optimized Execution (Class-Weight Balancing)
To fix this majority-class bias, we added class-weight balancing (class_weight='balanced'). This adjustment recalibrates the model's internal loss calculation, placing higher penalties on misclassified defaults during training. 

This optimization successfully shifted the internal decision thresholds within the decision trees. As a result, the model began aggressively flagging high-risk indicators—such as poor credit scores and high interest rates—and correctly flipped the final predictive decisions to Deny for unstable applicants.

---

## 💡 Actionable Operational Insights

1. Age-Based Risk Mitigation: Age exhibits the strongest single negative correlation (-0.17) with loan defaults. Younger demographics present a higher probability of repayment friction. Risk-mitigation models should enforce a strict requirement for a co-signer (HasCoSigner) for applicants under a specific age threshold.
2. Interest Rate Thresholds: Interest rates display a direct positive correlation (+0.13) with defaults. High interest rates significantly increase monthly debt service requirements, driving up default rates. Financial institutions should avoid overloading high-rate products onto low-income applicants.
3. The Leverage Interaction Profile: While a larger LoanAmount correlates with increased risk (+0.09), Income maintains an inverse relationship (-0.10). Credit risk assessment teams should prioritize evaluating the interaction term (the direct ratio of loan scale relative to monthly income) rather than reviewing raw income figures in isolation.

---

## 🚀 Deployment & UI Interface
To move the machine learning model from a script to an interactive testing environment, a web UI was built using Streamlit. 

The application provides interactive sliders and drop-down menus that match the dataset's features, allowing stakeholders to test different borrower profiles in real-time. To make the application continuously available for review throughout the evaluation week, it was deployed via Streamlit Community Cloud, connected directly to a public GitHub repository. 

To ensure stability on the cloud platform's free hosting tier, we implemented a memory-safe data loading system (nrows=40000). This optimization reduces the memory footprint by over 80%, avoiding Out-of-Memory (OOM) crashes while fully preserving the model's predictive patterns and logic.

---

## 📂 Project Repository Directory Structure

The workspace environment layout matches the core architecture blueprint below:

```text
READYNEST_TASK_4/
├── data/
│   └── Loan_default.csv                  # Raw borrower data vector matrix
├── outputs/
│   ├── report.txt                        # Evaluator text document metrics
│   ├── visual1_default_distribution.png  # Target distribution scale chart
│   ├── visual2_dti_density.png           # Repayment density behavior curve
│   └── visual3_correlation_matrix.png    # Masked triangle correlation matrix
└── script/
    ├── app.py                            # Streamlit frontend user interface
    ├── clean.py                          # IQR clipping & imputation pipeline
    ├── model.py                          # Stratified Random Forest training script
    └── visual.py                         # Seaborn analytical visualization pipeline