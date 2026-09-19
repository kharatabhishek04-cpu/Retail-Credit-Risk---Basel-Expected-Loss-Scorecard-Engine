import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv('data/barclays_loan_portfolio.csv')

# 2. Total Good (0) and Total Bad (1) across the entire portfolio
total_good = (df['loan_status'] == 0).sum()
total_bad = (df['loan_status'] == 1).sum()

print(f"Portfolio Totals -> Good: {total_good:,} | Bad (Defaults): {total_bad:,}")

# 3. Bin 'debt_to_income' into 4 industry-standard regulatory brackets
df['dti_bin'] = pd.cut(
    df['debt_to_income'],
    bins=[-np.inf, 12.0, 20.0, 30.0, np.inf],
    labels=['< 12% (Prime)', '12% - 20% (Near Prime)', '20% - 30% (Mid Risk)', '> 30% (High Risk)']
)

# 4. Group by bin and calculate WoE
woe_table = df.groupby('dti_bin', observed=False).agg(
    total_customers=('customer_id', 'count'),
    good=('loan_status', lambda s: (s == 0).sum()),
    bad=('loan_status', lambda s: (s == 1).sum())
).reset_index()

# Calculate % of all good and % of all bad
woe_table['dist_good'] = woe_table['good'] / total_good
woe_table['dist_bad'] = woe_table['bad'] / total_bad

# Calculate Weight of Evidence (WoE)
woe_table['woe'] = np.log(woe_table['dist_good'] / woe_table['dist_bad'])

# Calculate Information Value (IV) contribution per bin
woe_table['iv_component'] = (woe_table['dist_good'] - woe_table['dist_bad']) * woe_table['woe']

total_iv = woe_table['iv_component'].sum()

print("\n" + "=" * 80)
print("WEIGHT OF EVIDENCE (WoE) TABLE: DEBT-TO-INCOME (DTI)")
print("=" * 80)
display_cols = ['dti_bin', 'total_customers', 'good', 'bad', 'woe', 'iv_component']
print(woe_table[display_cols].to_string(index=False))

print("-" * 80)
print(f"Total Information Value (IV) for Debt-to-Income: {total_iv:.4f}")

if total_iv >= 0.3:
    print("Regulatory Rating: STRONG PREDICTOR (Include in Barclays Scorecard)")
elif total_iv >= 0.1:
    print("Regulatory Rating: MEDIUM PREDICTOR (Acceptable)")
else:
    print("Regulatory Rating: WEAK PREDICTOR (Review or Drop)")