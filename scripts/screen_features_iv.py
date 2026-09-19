import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv('data/barclays_loan_portfolio.csv')
total_good = (df['loan_status'] == 0).sum()
total_bad = (df['loan_status'] == 1).sum()

def calculate_iv(series, target, bins, labels):
    """Calculates Information Value for a continuous variable."""
    binned = pd.cut(series, bins=bins, labels=labels)
    t = pd.crosstab(binned, target)
    dist_good = t[0] / total_good
    dist_bad = t[1] / total_bad
    woe = np.log(dist_good / dist_bad)
    # replace potential infs with 0
    woe = woe.replace([np.inf, -np.inf], 0)
    iv = ((dist_good - dist_bad) * woe).sum()
    return round(iv, 4)

# 2. Compute IV across all core risk drivers
iv_summary = []

# DTI
iv_dti = calculate_iv(df['debt_to_income'], df['loan_status'], 
                      bins=[-np.inf, 12, 20, 30, np.inf], 
                      labels=['<12%', '12-20%', '20-30%', '>30%'])
iv_summary.append({'Feature': 'debt_to_income', 'IV': iv_dti})

# Revolving Utilization
iv_util = calculate_iv(df['revolving_utilization'], df['loan_status'], 
                       bins=[-np.inf, 30, 60, 80, np.inf], 
                       labels=['<30% (Low)', '30-60% (Moderate)', '60-80% (High)', '>80% (Severely Stressed)'])
iv_summary.append({'Feature': 'revolving_utilization', 'IV': iv_util})

# Delinquencies in last 2 years
iv_delinq = calculate_iv(df['delinquencies_2yrs'], df['loan_status'], 
                         bins=[-np.inf, 0, 1, np.inf], 
                         labels=['0 Missed', '1 Missed', '2+ Missed'])
iv_summary.append({'Feature': 'delinquencies_2yrs', 'IV': iv_delinq})

# Credit Inquiries last 6 months (Credit Seeking)
iv_inq = calculate_iv(df['credit_inquiries_6m'], df['loan_status'], 
                      bins=[-np.inf, 0, 1, 2, np.inf], 
                      labels=['0 Inquiries', '1 Inquiry', '2 Inquiries', '3+ Inquiries'])
iv_summary.append({'Feature': 'credit_inquiries_6m', 'IV': iv_inq})

# Annual Income
iv_inc = calculate_iv(df['annual_income'], df['loan_status'], 
                      bins=[-np.inf, 25000, 45000, 75000, np.inf], 
                      labels=['<£25k', '£25k-£45k', '£45k-£75k', '>£75k'])
iv_summary.append({'Feature': 'annual_income', 'IV': iv_inc})

# 3. Create Ranked Summary DataFrame
iv_df = pd.DataFrame(iv_summary).sort_values(by='IV', ascending=False).reset_index(drop=True)

def rating(iv):
    if iv >= 0.3: return 'Strong Predictor'
    elif iv >= 0.1: return 'Medium Predictor'
    elif iv >= 0.02: return 'Weak Predictor'
    else: return 'Unusable'

iv_df['Regulatory_Power'] = iv_df['IV'].apply(rating)

print("=" * 65)
print("BARCLAYS CREDIT SCORECARD: INFORMATION VALUE (IV) SCREENING")
print("=" * 65)
print(iv_df.to_string(index=False))
print("-" * 65)