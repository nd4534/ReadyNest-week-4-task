import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("[*] Starting Stage 2: Generating UI-Grade Visualizations...")

# --- NEW: Read the data first so 'df' exists in this file ---
file_name = 'Loan_default.csv'
df = pd.read_csv(file_name)

# Establish global clean UI/UX styling configurations
sns.set_theme(style="whitegrid")
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#475569'

# Define our professional high-contrast color scheme
ui_palette = {0: '#0f4c81', 1: '#f26a36'}

# ------------------------------------------------------------------
# VISUAL 1: Target Class Distribution (Checking for Imbalance)
# ------------------------------------------------------------------
plt.figure(figsize=(6, 5))
ax1 = sns.countplot(data=df, x='Default', palette=ui_palette, hue='Default', legend=False)

plt.title('Target Metric: Loan Default Distribution', fontsize=13, fontweight='bold', pad=15)
plt.xticks([0, 1], ['Fully Paid (0)', 'Defaulted (1)'], fontsize=11)
plt.xlabel('')
plt.ylabel('Count of Borrowers', fontsize=11)
sns.despine(left=True, bottom=True)

# Add clear value labels on top of the bars
for p in ax1.patches:
    ax1.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('visual1_default_distribution.png', dpi=300)
plt.close()
print("[+] Visual 1 saved successfully as 'visual1_default_distribution.png'")

# ------------------------------------------------------------------
# REFINED VISUAL 2: Debt-to-Income (DTI) Ratio Density (No Grid Clutter)
# ------------------------------------------------------------------
plt.figure(figsize=(9, 5))
sns.set_style("white") # Removes the background grid lines completely

sns.kdeplot(data=df, x='DTIRatio', hue='Default', palette=ui_palette, 
            fill=True, common_norm=False, alpha=0.4, linewidth=2)

plt.title('Risk Profile: Debt-to-Income (DTI) Ratio Density', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Debt-to-Income (DTI) Ratio', fontsize=11)
plt.ylabel('Density Distribution', fontsize=11)

plt.legend(title='Status', labels=['Defaulted (1)', 'Fully Paid (0)'], frameon=False)
sns.despine(top=True, right=True, left=False, bottom=False)

plt.tight_layout()
plt.savefig('visual2_dti_density.png', dpi=300)
plt.close()


# ------------------------------------------------------------------
# REFINED VISUAL 3: Core Features Correlation Matrix (No Artifact Gridlines)
# ------------------------------------------------------------------
plt.figure(figsize=(10, 7))
sns.set_style("white") # Enforces clean white canvas under the heatmap

numerical_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'InterestRate', 'DTIRatio', 'Default']
corr_matrix = df[numerical_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Plot without background line bleeding
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0, mask=mask,
            square=True, cbar_kws={"shrink": .7}, linewidths=0.5)

plt.title('Feature Interaction & Correlation Grid', fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('visual3_correlation_matrix.png', dpi=300)
plt.close()
print("[+] Visualizations fully polished and updated!")