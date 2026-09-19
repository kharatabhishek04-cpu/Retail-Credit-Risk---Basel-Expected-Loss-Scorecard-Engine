import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)
n_records = 50000

print("Generating realistic retail banking loan portfolio (50,000 records)...")

# 1. Customer Demographics & Financials
age = np.random.randint(21, 68, size=n_records)
annual_income = np.random.lognormal(mean=10.5, sigma=0.55, size=n_records).round(-2)
emp_length = np.random.choice([0, 1, 2, 3, 5, 8, 10, 15], size=n_records, p=[0.08, 0.10, 0.12, 0.15, 0.20, 0.15, 0.12, 0.08])
home_ownership = np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], size=n_records, p=[0.45, 0.43, 0.11, 0.01])

# 2. Loan Characteristics
loan_amount = np.random.choice([2500, 5000, 7500, 10000, 15000, 20000, 25000], size=n_records, p=[0.10, 0.20, 0.25, 0.20, 0.13, 0.08, 0.04])
loan_term = np.random.choice([36, 60], size=n_records, p=[0.70, 0.30])
loan_intent = np.random.choice(['DEBT_CONSOLIDATION', 'HOME_IMPROVEMENT', 'CREDIT_CARD_REFINANCE', 'MAJOR_PURCHASE', 'MEDICAL'], size=n_records, p=[0.45, 0.20, 0.18, 0.12, 0.05])

# 3. Credit Bureau Indicators
dti = np.clip(np.random.normal(loc=18.5, scale=7.5, size=n_records), 2.0, 55.0).round(2)
revolving_util = np.clip(np.random.normal(loc=42.0, scale=22.0, size=n_records), 0.0, 100.0).round(2)
delinq_2yrs = np.random.choice([0, 1, 2, 3], size=n_records, p=[0.82, 0.12, 0.04, 0.02])
inquiries_6m = np.random.choice([0, 1, 2, 3, 4], size=n_records, p=[0.55, 0.25, 0.12, 0.05, 0.03])

# 4. Realistic Banking Default Probability (Calibrated ~7.5% UK portfolio default rate)
log_odds = (
    -3.8
    + (loan_amount / 10000) * 0.25
    + (dti / 10) * 0.35
    + (revolving_util / 20) * 0.40
    + delinq_2yrs * 0.65
    + inquiries_6m * 0.30
    - (annual_income / 50000) * 0.45
    - (emp_length / 5) * 0.15
    + np.where(home_ownership == 'RENT', 0.25, 0.0)
)
prob_default = 1 / (1 + np.exp(-log_odds))
loan_status = np.random.binomial(1, prob_default)

# 5. Build DataFrame
df = pd.DataFrame({
    'customer_id': [f"BARC-{100000 + i}" for i in range(n_records)],
    'age': age,
    'annual_income': annual_income,
    'emp_length_years': emp_length,
    'home_ownership': home_ownership,
    'loan_amount': loan_amount,
    'loan_term_months': loan_term,
    'loan_intent': loan_intent,
    'debt_to_income': dti,
    'revolving_utilization': revolving_util,
    'delinquencies_2yrs': delinq_2yrs,
    'credit_inquiries_6m': inquiries_6m,
    'loan_status': loan_status
})

os.makedirs('data', exist_ok=True)
output_path = 'data/barclays_loan_portfolio.csv'
df.to_csv(output_path, index=False)

print(f"Success! Dataset saved to {output_path}")
print(f"Total Portfolio Records: {len(df):,}")
print(f"Default Rate: {df['loan_status'].mean():.2%}")