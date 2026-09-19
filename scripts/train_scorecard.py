import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Load data
df = pd.read_csv('data/barclays_loan_portfolio.csv')

# 2. Select model features
features = ['debt_to_income', 'revolving_utilization', 'delinquencies_2yrs', 'credit_inquiries_6m', 'annual_income']
X = df[features]
y = df['loan_status']

# 3. Train-Test Split (80% Train, 20% Out-Of-Time/Test validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. Predict Probability of Default (PD) on test set
# proba[:, 1] is Probability of Default (Bad)
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 6. Model Discrimination Power: AUC-ROC & Gini Coefficient
auc = roc_auc_score(y_test, y_pred_prob)
gini = (2 * auc) - 1

# 7. Scale PD into Credit Score (300 - 850 scale)
# Avoid division by zero
p_good = np.clip(1 - y_pred_prob, 0.0001, 0.9999)
p_bad = np.clip(y_pred_prob, 0.0001, 0.9999)

# Standard Banking Scaling: Target Score = 600 at 50:1 odds, PDO = 20
pdo = 20
factor = pdo / np.log(2)
offset = 600 - factor * np.log(50)

# Calculate Credit Score
scores = offset + factor * np.log(p_good / p_bad)
scores = np.clip(scores, 300, 850).round().astype(int)

# 8. Print Executive Results
print("=" * 65)
print("BARCLAYS CREDIT RISK MODEL - PERFORMANCE & SCORING")
print("=" * 65)
print(f"Validation Sample: {len(y_test):,} customer loans")
print(f"Model AUC-ROC:     {auc:.4f}  (Target: > 0.75)")
print(f"Model Gini Score:  {gini:.4f}  (Target: > 0.50)")
print("-" * 65)
print(f"Generated Credit Scores:")
print(f"  Minimum Score:   {scores.min()}")
print(f"  Median Score:    {int(np.median(scores))}")
print(f"  Maximum Score:   {scores.max()}")
print("-" * 65)