# Retail Credit Risk & Basel Expected Loss Scorecard Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tableau](https://img.shields.io/badge/Tableau-Public-E97627?logo=tableau&logoColor=white)](https://public.tableau.com/app/profile/abhishek.kharat/viz/RetailCreditRiskBaselExpectedLossScorecardEngine/Dashboard1)
[![Regulatory Standards](https://img.shields.io/badge/Standards-Basel%20II%2FIII%20%7C%20IFRS%209-green)](https://www.bis.org/bcbs/)

An end-to-end, regulatory-compliant retail credit risk application scorecard and expected loss calculation engine designed for commercial banking portfolios. The project implements Weight of Evidence (WoE) and Information Value (IV) screening, logistic regression calibration scaled into an industry-standard 300–850 credit scoring system, Basel Expected Credit Loss ($\text{EL} = \text{PD} \times \text{LGD} \times \text{EAD}$) quantification across a **£484.5M portfolio**, and an interactive executive cockpit published on Tableau Public.

🔗 **[View Live Interactive Dashboard on Tableau Public](https://public.tableau.com/app/profile/abhishek.kharat/viz/RetailCreditRiskBaselExpectedLossScorecardEngine/Dashboard1)**

---

## 🏛️ Business Context & Regulatory Framework

Under UK and European prudential regulatory frameworks (**Basel II/III** and **IFRS 9**), financial institutions cannot deploy uninterpretable "black-box" machine learning algorithms for consumer credit decisioning. The **Prudential Regulation Authority (PRA)** mandates that credit underwriting models must satisfy three core criteria:

1. **Interpretability & Auditability:** Every credit decision must trace back to transparent score point deductions for individual applicant characteristics.
2. **Monotonicity:** Risk scores must exhibit consistent directional relationships with default rates across attribute bins (e.g., higher leverage must monotonically translate to lower credit scores).
3. **Statistical Robustness:** Outlier sensitivity must be mitigated through coarse risk banding (binning) to maintain stability across macroeconomic cycles.

This engine models an unsecured retail loan portfolio (personal loans, credit consolidation, debt refinancing) to evaluate borrower default risk, calibrate loss reserves, and simulate credit cutoff adjustments.

---

## 🏗️ System Architecture & Analytical Workflow

```mermaid
graph TD
    A[Raw Loan Portfolio Ingestion<br/>50,000 Records | 12 Attributes] --> B[Univariate Risk Profiling & EDA<br/>Bad Rate Analysis by Segment]
    B --> C[Feature Engineering: WoE & IV Screening<br/>Monotonic Binning & Predictor Ranking]
    C --> D[Logistic Regression Model Training<br/>Probability of Default PD Calibration]
    D --> E[Scorecard Scaling Engine<br/>Base Odds 50:1 @ 600 pts | PDO = 20 pts<br/>300 - 850 Range]
    E --> F[Basel Regulatory Engine<br/>EL = PD x LGD x EAD | LGD = 45%]
    F --> G[Interactive Executive Risk Cockpit<br/>Tableau Public Deployment]
```

---

## 📐 Mathematical Methodology

### 1. Weight of Evidence (WoE) & Information Value (IV)
Continuous financial attributes were transformed into monotonic risk bins to linearize log-odds and suppress outlier variance. 

For each attribute bin $i$:
$$\text{WoE}_i = \ln\left(\frac{\% \text{ of All Non-Defaulters (Good)}_i}{\% \text{ of All Defaulters (Bad)}_i}\right)$$

Overall predictive strength was evaluated using Information Value (IV):
$$\text{IV} = \sum_{i=1}^{k} \left(\% \text{ Good}_i - \% \text{ Bad}_i\right) \times \text{WoE}_i$$

#### Feature Screening Results:
| Feature | Information Value (IV) | Regulatory Predictive Rating |
| :--- | :--- | :--- |
| `delinquencies_2yrs` | **0.1835** | **Medium-Strong Predictor** (Highest Power) |
| `debt_to_income` | **0.0562** | **Medium Predictor** |
| `revolving_utilization` | **0.0489** | **Medium Predictor** |
| `credit_inquiries_6m` | **0.0412** | **Medium Predictor** |
| `annual_income` | **0.0351** | **Weak Predictor** |

> **Key Banking Insight:** Historical payment delinquency (`delinquencies_2yrs`) proved **5x more predictive** than raw customer income (`annual_income`), validating that repayment discipline outweighs earning capacity in retail default prediction.

---

### 2. Scorecard Scaling (300 to 850 Scale)
The logistic regression output was transformed into credit score points using standard banking calibration parameters:
* **Target Score:** 600 points at **Base Odds of 50:1** (Good:Bad).
* **Points to Double the Odds (PDO):** 20 points.

$$\text{Factor} = \frac{\text{PDO}}{\ln(2)} = \frac{20}{\ln(2)} \approx 28.8539$$
$$\text{Offset} = \text{Target Score} - \text{Factor} \times \ln(\text{Base Odds}) = 600 - 28.8539 \times \ln(50) \approx 487.12$$
$$\text{Credit Score} = \text{Offset} + \text{Factor} \times \ln\left(\frac{1 - \text{PD}}{\text{PD}}\right)$$

#### Model Discrimination Performance:
* **AUC-ROC:** `0.6996` (Standard baseline discrimination for retail application scorecards without bureau trade-line depth).
* **Gini Coefficient:** `0.3993` ($2 \times \text{AUC} - 1$).
* **Score Distribution:** Minimum `441`, Median `549`, Maximum `621`.

---

### 3. Basel Expected Credit Loss (ECL) Quantification
Financial loss reserves were calculated across every account:
$$\text{Expected Loss (EL)} = \text{PD} \times \text{LGD} \times \text{EAD}$$
* **PD:** Modeled 12-month Probability of Default.
* **LGD:** Fixed at **45.0%** in alignment with Basel Foundation Internal Ratings-Based (F-IRB) parameters for senior unsecured retail exposures.
* **EAD:** Current outstanding principal exposure (`loan_amount`).

---

## 📊 Portfolio Summary & Financial Exposure

Analysis of the 50,000-account retail lending book produced the following portfolio metrics:

| Metric | Portfolio Value | Commercial Context |
| :--- | :--- | :--- |
| **Total Loan Book Exposure (EAD)** | **£484,467,500.00** | Full retail lending book principal |
| **Total Expected Credit Loss (EL)** | **£28,681,325.68** | Required IFRS 9 regulatory loss reserve |
| **Portfolio Expected Loss Ratio** | **5.92%** | Unsecured retail loss baseline |
| **Total Active Borrowers** | **50,000** | Retail customer accounts |

### Risk Tier Breakdown
| Credit Risk Tier | Score Band | Accounts | Total Exposure (£) | Expected Loss (£) | Loss Ratio (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Prime (Low Risk)** | $\ge 580$ | 2,468 | £24,217,500 | £751,210 | 3.10% |
| **2. Near Prime (Moderate Risk)** | $530 - 579$ | 38,214 | £370,122,500 | £21,124,198 | 5.71% |
| **3. Subprime (High Risk)** | $< 530$ | 9,318 | £90,127,500 | £6,805,917 | 7.55% |

---

## 💡 Strategic Risk Recommendations for Senior Leadership

1. **Volume Concentration vs. Unit Default Rate:**
   * While **Medical loans** demonstrated the highest unit default rate (13.8%), **Debt Consolidation** represents the primary financial exposure (**£13.0M expected loss**, ~45% of total losses) due to portfolio volume concentration.
2. **Cut-off Threshold Optimization:**
   * Shifting the automated underwriting cutoff score from **530 to 550** would eliminate **£8.2M** in vulnerable subprime expected losses while preserving **82.4%** of commercial lending volume.
3. **Housing Segment Sensitivity:**
   * Renters represent **£218M** of lending exposure with an elevated loss ratio of **5.93%**, suggesting that rental verification can serve as an effective secondary risk filter.

---

## 💻 Repository Structure & Reproducibility

```text
credit_risk_scorecard/
├── README.md                              # Technical & commercial documentation
├── barclays_credit_risk_dashboard.twbx   # Packaged Tableau workbook
├── data/
│   ├── barclays_loan_portfolio.csv        # Raw synthesized loan applications (50k records)
│   └── barclays_scored_portfolio.csv      # Production scored dataset with PD, scores & EL
└── scripts/
    ├── generate_loan_data.py              # Portfolio simulation script
    ├── eda_risk_profile.py                # Univariate bad rate analysis
    ├── calculate_woe_iv.py                # Single-feature WoE/IV calculator
    ├── screen_features_iv.py              # Portfolio-wide IV ranking engine
    ├── train_scorecard.py                 # Logistic regression & score scaling (300-850)
    └── calculate_expected_loss.py         # Basel Expected Loss quantification pipeline
```

### Quickstart Execution

```bash
# 1. Clone or navigate to the project directory
cd projects/credit_risk_scorecard

# 2. Install dependencies
pip install pandas numpy scikit-learn

# 3. Execute data pipeline & score the portfolio
python3 scripts/generate_loan_data.py
python3 scripts/screen_features_iv.py
python3 scripts/calculate_expected_loss.py
```

---

## 👤 Author & Tableau Public Portfolio

* **Author:** Abhishek Kharat
* **Role Focus:** Technology Analyst / Data Analytics
* **Live Dashboard:** [Tableau Public - Retail Credit Risk & Expected Loss Engine](https://public.tableau.com/app/profile/abhishek.kharat/viz/RetailCreditRiskBaselExpectedLossScorecardEngine/Dashboard1)
