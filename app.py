
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Risk Grade Predictor", page_icon="\U0001F4CA", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("credit_risk_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    model_columns = joblib.load("model_columns.pkl")
    return model, label_encoder, model_columns


model, label_encoder, model_columns = load_artifacts()

#making human readable
FIELD_META = {
    # --- Applicant Profile ---
    "AGE": ("Applicant Age (years)", "Applicant's current age.", "Applicant Profile", "number"),
    "NETMONTHLYINCOME": ("Net Monthly Income (\u20b9)", "Take-home income per month.", "Applicant Profile", "number"),
    "Time_With_Curr_Empr": ("Time With Current Employer (months)", "How long at the current job.", "Applicant Profile", "number"),
    "Credit_Score": ("Credit Bureau Score", "Score from the credit bureau (e.g. 300-900).", "Applicant Profile", "number"),

    # --- Trade Line Summary ---
    "Total_TL_opened_L12M": ("Trade Lines Opened (Last 12 Months)", "Number of new credit accounts opened in the last year.", "Trade Line Summary", "number"),
    "Tot_TL_closed_L12M": ("Trade Lines Closed (Last 12 Months)", "Number of credit accounts closed in the last year.", "Trade Line Summary", "number"),
    "pct_tl_open_L6M": ("% Trade Lines Opened (Last 6 Months)", "Share of all accounts that were opened in the last 6 months.", "Trade Line Summary", "percent"),
    "pct_tl_closed_L6M": ("% Trade Lines Closed (Last 6 Months)", "Share of all accounts that were closed in the last 6 months.", "Trade Line Summary", "percent"),
    "pct_tl_open_L12M": ("% Trade Lines Opened (Last 12 Months)", "Share of all accounts that were opened in the last 12 months.", "Trade Line Summary", "percent"),
    "pct_tl_closed_L12M": ("% Trade Lines Closed (Last 12 Months)", "Share of all accounts that were closed in the last 12 months.", "Trade Line Summary", "percent"),
    "pct_of_active_TLs_ever": ("% Active Trade Lines (All Time)", "Share of all-time accounts that are still active.", "Trade Line Summary", "percent"),
    "pct_opened_TLs_L6m_of_L12m": ("% of Yearly New Accounts Opened in Last 6M", "Of the accounts opened in the last 12 months, what share were in the last 6.", "Trade Line Summary", "percent"),
    "Age_Oldest_TL": ("Age of Oldest Account (months)", "How long ago the applicant's oldest credit account was opened.", "Trade Line Summary", "number"),
    "Age_Newest_TL": ("Age of Newest Account (months)", "How long ago the applicant's newest credit account was opened.", "Trade Line Summary", "number"),

    # --- Loan Type Counts ---
    "CC_TL": ("Number of Credit Card Accounts", None, "Loan Types Held", "number"),
    "Home_TL": ("Number of Home Loan Accounts", None, "Loan Types Held", "number"),
    "PL_TL": ("Number of Personal Loan Accounts", None, "Loan Types Held", "number"),
    "Secured_TL": ("Number of Secured Loan Accounts", "Loans backed by collateral (e.g. home, gold).", "Loan Types Held", "number"),
    "Unsecured_TL": ("Number of Unsecured Loan Accounts", "Loans with no collateral (e.g. personal loans, credit cards).", "Loan Types Held", "number"),
    "Other_TL": ("Number of Other Loan Accounts", None, "Loan Types Held", "number"),
    "CC_Flag": ("Has a Credit Card?", None, "Loan Types Held", "flag"),
    "PL_Flag": ("Has a Personal Loan?", None, "Loan Types Held", "flag"),
    "HL_Flag": ("Has a Home Loan?", None, "Loan Types Held", "flag"),
    "GL_Flag": ("Has a Gold Loan?", None, "Loan Types Held", "flag"),

    # --- Payment & Delinquency History ---
    "Tot_Missed_Pmnt": ("Total Missed Payments", None, "Payment History", "number"),
    "time_since_recent_payment": ("Days Since Most Recent Payment", None, "Payment History", "number"),
    "max_recent_level_of_deliq": ("Worst Recent Delinquency Level", "Highest severity of missed payment recently (0 = none).", "Payment History", "number"),
    "recent_level_of_deliq": ("Current Delinquency Level", "Severity of the applicant's most recent missed payment (0 = none).", "Payment History", "number"),
    "num_deliq_6_12mts": ("Delinquencies (6-12 Months Ago)", "Number of missed payments 6 to 12 months ago.", "Payment History", "number"),
    "num_times_60p_dpd": ("Times 60+ Days Past Due", "Number of times a payment was 60 or more days late.", "Payment History", "number"),
    "num_std_12mts": ("Standard (On-Time) Accounts (Last 12 Months)", None, "Payment History", "number"),
    "num_sub": ("Substandard Accounts (Total)", "Accounts with a history of repeated late payments.", "Payment History", "number"),
    "num_sub_6mts": ("Substandard Accounts (Last 6 Months)", None, "Payment History", "number"),
    "num_sub_12mts": ("Substandard Accounts (Last 12 Months)", None, "Payment History", "number"),
    "num_dbt": ("Doubtful Accounts (Total)", "Accounts considered very unlikely to be repaid in full.", "Payment History", "number"),
    "num_dbt_12mts": ("Doubtful Accounts (Last 12 Months)", None, "Payment History", "number"),
    "num_lss": ("Loss Accounts (Written Off)", "Accounts the lender has written off as a loss.", "Payment History", "number"),

    # --- Credit Enquiries ---
    "CC_enq": ("Credit Card Enquiries (Total)", "Number of times the applicant applied for a credit card.", "Credit Enquiries", "number"),
    "CC_enq_L12m": ("Credit Card Enquiries (Last 12 Months)", None, "Credit Enquiries", "number"),
    "PL_enq_L12m": ("Personal Loan Enquiries (Last 12 Months)", None, "Credit Enquiries", "number"),
    "time_since_recent_enq": ("Days Since Most Recent Enquiry", None, "Credit Enquiries", "number"),
    "enq_L3m": ("Total Enquiries (Last 3 Months)", "Number of credit applications in the last 3 months.", "Credit Enquiries", "number"),
    "pct_PL_enq_L6m_of_ever": ("% of Personal Loan Enquiries in Last 6 Months", "Of all-time personal loan enquiries, share made in the last 6 months.", "Credit Enquiries", "percent"),
    "pct_CC_enq_L6m_of_ever": ("% of Credit Card Enquiries in Last 6 Months", "Of all-time credit card enquiries, share made in the last 6 months.", "Credit Enquiries", "percent"),
}

GROUP_ORDER = ["Applicant Profile", "Trade Line Summary", "Loan Types Held", "Payment History", "Credit Enquiries"]

CATEGORICAL_PREFIXES = ["MARITALSTATUS", "GENDER", "last_prod_enq2", "first_prod_enq2"]
CATEGORICAL_LABELS = {
    "MARITALSTATUS": "Marital Status",
    "GENDER": "Gender",
    "last_prod_enq2": "Most Recent Product Enquired For",
    "first_prod_enq2": "First Product Ever Enquired For",
}
PRODUCT_LABELS = {
    "AL": "Auto Loan", "CC": "Credit Card", "ConsumerLoan": "Consumer Loan",
    "HL": "Home Loan", "PL": "Personal Loan", "others": "Other",
}

EDUCATION_MAPPING = {
    "SSC": 1, "12TH": 2, "GRADUATE": 3, "UNDER GRADUATE": 3,
    "POST-GRADUATE": 4, "OTHERS": 1, "PROFESSIONAL": 3,
}


def humanize(col: str) -> str:
    """Fallback label for any column not in FIELD_META: snake_case -> Title Case."""
    return col.replace("_", " ").strip().title()


# ----------------------------------------------------------------------
# Split model_columns into numeric fields, EDUCATION, and one-hot groups
# ----------------------------------------------------------------------
dummy_groups = {prefix: [] for prefix in CATEGORICAL_PREFIXES}
numeric_columns = []

for col in model_columns:
    matched = False
    for prefix in CATEGORICAL_PREFIXES:
        if col.startswith(prefix + "_"):
            dummy_groups[prefix].append(col)
            matched = True
            break
    if not matched and col != "EDUCATION":
        numeric_columns.append(col)

# Group numeric columns by their FIELD_META group (unlisted columns go in "Other")
grouped_numeric = {g: [] for g in GROUP_ORDER}
grouped_numeric["Other"] = []
for col in numeric_columns:
    group = FIELD_META.get(col, (None, None, "Other", "number"))[2]
    grouped_numeric.setdefault(group, []).append(col)

st.title("Credit Risk Grade Predictor")
st.caption("Predicts approval risk category (P1 = best, P4 = riskiest) from applicant data.")

with st.form("prediction_form"):
    st.subheader("Applicant Category")
    c1, c2 = st.columns(2)
    with c1:
        education_label = st.selectbox("Education", list(EDUCATION_MAPPING.keys()))
        marital_choice = st.selectbox(CATEGORICAL_LABELS["MARITALSTATUS"], [c[len("MARITALSTATUS_"):] for c in dummy_groups["MARITALSTATUS"]])
    with c2:
        gender_choice = st.selectbox(CATEGORICAL_LABELS["GENDER"], [c[len("GENDER_"):] for c in dummy_groups["GENDER"]])

    prod_c1, prod_c2 = st.columns(2)
    with prod_c1:
        last_prod_options = [c[len("last_prod_enq2_"):] for c in dummy_groups["last_prod_enq2"]]
        last_prod_choice = st.selectbox(
            CATEGORICAL_LABELS["last_prod_enq2"], last_prod_options,
            format_func=lambda x: PRODUCT_LABELS.get(x, x),
        )
    with prod_c2:
        first_prod_options = [c[len("first_prod_enq2_"):] for c in dummy_groups["first_prod_enq2"]]
        first_prod_choice = st.selectbox(
            CATEGORICAL_LABELS["first_prod_enq2"], first_prod_options,
            format_func=lambda x: PRODUCT_LABELS.get(x, x),
        )

    categorical_choices = {
        "MARITALSTATUS": marital_choice, "GENDER": gender_choice,
        "last_prod_enq2": last_prod_choice, "first_prod_enq2": first_prod_choice,
    }

    st.subheader("Applicant Numbers")
    tabs = st.tabs([g for g in GROUP_ORDER if grouped_numeric[g]])
    numeric_inputs = {}
    flag_choices = {}

    tab_idx = 0
    for group in GROUP_ORDER:
        cols_in_group = grouped_numeric[group]
        if not cols_in_group:
            continue
        with tabs[tab_idx]:
            for i in range(0, len(cols_in_group), 2):
                pair = cols_in_group[i:i + 2]
                row_cols = st.columns(len(pair))
                for rc, colname in zip(row_cols, pair):
                    label, help_text, _, kind = FIELD_META.get(colname, (humanize(colname), None, group, "number"))
                    with rc:
                        if kind == "flag":
                            choice = st.selectbox(label, ["No", "Yes"], help=help_text, key=colname)
                            flag_choices[colname] = 1 if choice == "Yes" else 0
                        elif kind == "percent":
                            numeric_inputs[colname] = st.number_input(
                                label, min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                                help=help_text, key=colname,
                            )
                        else:
                            numeric_inputs[colname] = st.number_input(
                                label, value=0.0, step=1.0, help=help_text, key=colname,
                            )
        tab_idx += 1

    # Any leftover numeric columns not in a known group (safety net)
    if grouped_numeric["Other"]:
        st.subheader("Other Fields")
        for colname in grouped_numeric["Other"]:
            numeric_inputs[colname] = st.number_input(humanize(colname), value=0.0, step=1.0, key=colname)

    submitted = st.form_submit_button("Predict Risk Category")

if submitted:
    row = {col: 0 for col in model_columns}
    row.update(numeric_inputs)
    row.update(flag_choices)
    row["EDUCATION"] = EDUCATION_MAPPING[education_label]

    for prefix, choice in categorical_choices.items():
        row[f"{prefix}_{choice}"] = 1

    input_df = pd.DataFrame([row])[model_columns]  # enforce exact training column order

    pred_encoded = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(input_df)[0]
    proba_df = pd.DataFrame(
        {"Category": label_encoder.inverse_transform(range(len(proba))), "Probability": proba}
    ).sort_values("Probability", ascending=False)

    st.success(f"Predicted risk category: **{pred_label}**")
    st.bar_chart(proba_df.set_index("Category"))