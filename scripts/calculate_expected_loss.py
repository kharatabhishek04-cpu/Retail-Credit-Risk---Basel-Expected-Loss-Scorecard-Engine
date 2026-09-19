import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Load full portfolio
df = pd.read_csv('data/barclays_loan_portfolio.csv')

# 2. Train model on historical portfolio
features = ['debt_to_income', 'revolving_utilization', 'delinquencies_2yrs', 'credit_inquiries_6m', 'annual_income']
X = df[features]
y = df['loan_status']

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

# 3. Score every customer in the loan book
df['prob_default_pd'] = model.predict_proba(X)[:, 1]

# 4. Scale to Credit Score (300 - 850 scale)
p_good = np.clip(1 - df['prob_default_pd'], 0.0001, 0.9999)
p_bad = np.clip(df['prob_default_pd'], 0.0001, 0.9999)
factor = 20 / np.log(2)
offset = 600 - factor * np.log(50)
df['credit_score'] = (offset + factor * np.log(p_good / p_bad)).clip(300, 850).round().astype(int)

# 5. Assign Regulatory Risk Tier
conditions = [
    (df['credit_score'] >= 580),
    (df['credit_score'] >= 530) & (df['credit_score'] < 580),
    (df['credit_score'] < 530)
]
choices = ['1. Prime (Low Risk)', '2. Near Prime (Moderate Risk)', '3. Subprime (High Risk)']
df['credit_risk_tier'] = np.select(conditions, choices, default='Unknown')

# 6. Calculate Basel Expected Loss (EL = PD * LGD * EAD)
# Regulatory LGD = 45% for unsecured retail lending
regulatory_lgd = 0.45
df['loss_given_default_lgd'] = regulatory_lgd
df['exposure_at_default_ead'] = df['loan_amount']

df['expected_loss_gbp'] = (df['prob_default_pd'] * df['loss_given_default_lgd'] * df['exposure_at_default_ead']).round(2)

# 7. Save Scored Portfolio for Power BI / Tableau Dashboard
output_file = 'data/barclays_scored_portfolio.csv'
df.to_csv(output_file, index=False)

# 8. Print Executive Portfolio Summary
total_book = df['loan_amount'].sum()
total_el = df['expected_loss_gbp'].sum()
el_ratio = total_el / total_book

print("=" * 70)
print("BARCLAYS RETAIL LENDING - BASEL EXPECTED LOSS SUMMARY")
print("=" * 70)
print(f"Total Loan Book Exposure (EAD):   £{total_book:,.2f}")
print(f"Total Expected Credit Loss (EL): £{total_el:,.2f}")
print(f"Portfolio Expected Loss Ratio:    {el_ratio:.2%}")
print("-" * 70)
print("\n--- PORTFOLIO BREAKDOWN BY RISK TIER ---")
tier_summary = df.groupby('credit_risk_tier').agg(
    accounts=('customer_id', 'count'),
    total_exposure_gbp=('loan_amount', 'sum'),
    expected_loss_gbp=('expected_loss_gbp', 'sum'),
    avg_score=('credit_score', 'mean')
).reset_index()

tier_summary['avg_score'] = tier_summary['avg_score'].round(1)
tier_summary['portfolio_share'] = (tier_summary['total_exposure_gbp'] / total_book).map('{:.1%}'.format)
tier_summary['expected_loss_gbp'] = tier_summary['expected_loss_gbp'].map('£{:,.2f}'.format)
tier_summary['total_exposure_gbp'] = tier_summary['total_exposure_gbp'].map('£{:,.2f}'.format)

print(tier_summary.to_string(index=False))
print(f"\n[Saved production scored dataset to {output_file}]")