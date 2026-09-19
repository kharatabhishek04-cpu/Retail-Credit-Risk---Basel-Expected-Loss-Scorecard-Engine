import pandas as pd

# Load our loan portfolio
df = pd.read_csv('data/barclays_loan_portfolio.csv')

print("=" * 60)
print("BARCLAYS RETAIL LOAN PORTFOLIO - RISK PROFILING")
print("=" * 60)

# 1. Dataset Shape & Missing Values
print(f"Total Applications: {len(df):,}")
print(f"Missing Values: {df.isnull().sum().sum()}")

# 2. Risk Profiling by Home Ownership
print("\n--- DEFAULT RATE BY HOME OWNERSHIP ---")
home_risk = df.groupby('home_ownership').agg(
    total_loans=('customer_id', 'count'),
    defaults=('loan_status', 'sum'),
    default_rate=('loan_status', 'mean')
).sort_values(by='default_rate', ascending=False)

home_risk['default_rate'] = home_risk['default_rate'].map('{:.2%}'.format)
print(home_risk)

# 3. Risk Profiling by Loan Intent (Purpose of Loan)
print("\n--- DEFAULT RATE BY LOAN INTENT ---")
intent_risk = df.groupby('loan_intent').agg(
    total_loans=('customer_id', 'count'),
    defaults=('loan_status', 'sum'),
    default_rate=('loan_status', 'mean')
).sort_values(by='default_rate', ascending=False)

intent_risk['default_rate'] = intent_risk['default_rate'].map('{:.2%}'.format)
print(intent_risk)