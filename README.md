# Credit Risk Modeling — Loan Applicant Risk Grading

A multi-class classification system that grades loan applicants into four risk categories (**P1** = lowest risk through **P4** = highest risk) using credit bureau and applicant financial data. Deployed as an interactive web app for non-technical business users.

## Overview

Banks need to triage loan applicants quickly and consistently. This project takes raw credit bureau data across two source datasets, runs a full statistical feature-selection pipeline, trains a gradient-boosted classifier, and serves predictions through a Streamlit interface that a risk analyst can use without touching code.

## Dataset

Two Excel datasets joined on `PROSPECTID`:

- **Dataset 1** — applicant-level attributes (age, income, employment tenure, credit score)
- **Dataset 2** — credit bureau trade line data (account counts, delinquency history, enquiry patterns)

The target variable `Approved_Flag` has four classes: P1, P2, P3, P4.

## Pipeline

### 1. Data Cleaning
- Sentinel values encoded as `-99999` treated as nulls
- Columns with more than 10,000 null records dropped entirely
- Remaining null rows removed
- Inner join across both datasets to guarantee no missing records

### 2. Feature Selection

Three statistical tests were applied to reduce the feature space:

| Test | Applied to | Criterion |
|---|---|---|
| **Chi-square** | Categorical features | p ≤ 0.05 (association with target) |
| **VIF** (Variance Inflation Factor) | Numerical features | VIF ≤ 6, checked sequentially to remove multicollinearity |
| **ANOVA** (one-way F-test) | VIF-surviving numerical features | p ≤ 0.05 (means differ across P1–P4) |

This reduced the raw feature set to the final predictors used for training.

### 3. Encoding
- **`EDUCATION`** — ordinal mapping that preserves rank order (SSC → 1, 12TH → 2, GRADUATE/UNDER GRADUATE/PROFESSIONAL → 3, POST-GRADUATE → 4)
- **`MARITALSTATUS`, `GENDER`, `last_prod_enq2`, `first_prod_enq2`** — one-hot encoded
- **`Approved_Flag`** — label encoded for XGBoost compatibility

### 4. Model Comparison

Three classifiers were trained and evaluated on an 80/20 split:

- Random Forest (200 estimators)
- **XGBoost** ← best performer
- Decision Tree (max depth 20)

XGBoost was selected and tuned via `GridSearchCV` (3-fold CV) across `n_estimators`, `max_depth`, and `learning_rate`.

Evaluation used accuracy plus per-class precision, recall, and F1 — important here because the four risk classes are imbalanced, and overall accuracy alone would hide poor performance on minority classes.

## Deployment

The trained model is served through a **Streamlit** web app deployed on Streamlit Community Cloud.

Key design decisions for usability:
- Raw bureau column names (`pct_tl_open_L6M`) mapped to plain-English labels ("% Trade Lines Opened (Last 6 Months)") with tooltips
- Inputs grouped into five tabs — Applicant Profile, Trade Line Summary, Loan Types Held, Payment History, Credit Enquiries — rather than one long form
- Binary flags rendered as Yes/No dropdowns instead of raw 0/1 fields
- Output shows the predicted grade plus a probability distribution across all four classes, so users see model confidence rather than just a label

## Project Structure

```
├── credit_risk_modeling.ipynb   # Full EDA, feature selection, model training
├── app.py                       # Streamlit web application
├── test_predictions.py          # Sanity-check script with two dummy profiles
├── requirements.txt             # Python dependencies
├── credit_risk_model.pkl        # Trained XGBoost model
├── label_encoder.pkl            # Fitted LabelEncoder for P1–P4
└── model_columns.pkl            # Training column order for input alignment
```

## Running Locally

```bash
git clone https://github.com/trimurtikumar/credit-risk-modeling.git
cd credit-risk-modeling
pip install -r requirements.txt
streamlit run app.py
```

To verify the model loads and predicts correctly before launching the UI:

```bash
python test_predictions.py
```

This runs two contrasting dummy applicants (a clean-history profile and a delinquency-heavy profile) and prints predicted grades with class probabilities.

## Deployment Notes

When deploying to Streamlit Community Cloud, set the **Python version to 3.11** under *Advanced settings* before clicking Deploy. The platform defaults to a newer Python release that lacks prebuilt wheels for several pinned dependencies, causing source builds to fail. A `runtime.txt` file will not override this — the version must be selected in the deploy UI.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `SciPy` · `statsmodels` · `Streamlit`

## Limitations

- Trained on a fixed historical dataset with no monitoring for data drift
- Class imbalance across P1–P4 affects per-class recall
- Not validated for regulatory compliance or fairness across protected attributes — the model includes `GENDER` as a feature, which would require fair-lending review before any real-world lending use
