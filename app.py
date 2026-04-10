import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Classic Capital Credit App v4", page_icon="💳", layout="wide")

DATA_FILE = Path("borrower_assessments.csv")
MODEL_FILE = Path("credit_model.pkl")
LOGO_FILE = Path("logo.png")

PRIMARY_BLUE = "#1518C7"
ACCENT_GOLD = "#F2C230"
LIGHT_BG = "#F7F9FC"
CARD_BG = "#FFFFFF"
TEXT_DARK = "#1F2937"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {LIGHT_BG};
    }}
    .main-title {{
        color: {PRIMARY_BLUE};
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }}
    .subtitle {{
        color: {TEXT_DARK};
        font-size: 1.05rem;
        margin-bottom: 1rem;
    }}
    .section-card {{
        background: {CARD_BG};
        border: 1px solid rgba(21, 24, 199, 0.12);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background: {CARD_BG};
        border-left: 7px solid {ACCENT_GOLD};
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 0.8rem;
    }}
    .metric-label {{
        color: {TEXT_DARK};
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }}
    .metric-value {{
        color: {PRIMARY_BLUE};
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
    }}
    .small-note {{
        color: #5B6472;
        font-size: 0.95rem;
    }}
    .risk-low {{color: #0F9D58; font-weight: 700;}}
    .risk-mod {{color: #B7791F; font-weight: 700;}}
    .risk-high {{color: #C53030; font-weight: 700;}}
    .risk-vhigh {{color: #8B0000; font-weight: 800;}}
    div.stButton > button {{
        background-color: {PRIMARY_BLUE};
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 700;
    }}
    div.stDownloadButton > button {{
        background-color: {ACCENT_GOLD};
        color: #1f2937;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 0.9rem;
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def classify_risk(pd_prob: float):
    if pd_prob < 0.20:
        return "Low Risk", "Approve"
    elif pd_prob < 0.40:
        return "Moderate Risk", "Approve with conditions"
    elif pd_prob < 0.60:
        return "High Risk", "Manual review"
    return "Very High Risk", "Reject or request strong guarantee"


def risk_class_name(risk_label: str):
    mapping = {
        "Low Risk": "risk-low",
        "Moderate Risk": "risk-mod",
        "High Risk": "risk-high",
        "Very High Risk": "risk-vhigh",
    }
    return mapping.get(risk_label, "")


def build_explanations(row: pd.DataFrame, pd_prob: float):
    reasons = []

    monthly_income = float(row.loc[0, "monthly_income"])
    monthly_expenses = float(row.loc[0, "monthly_expenses"])
    monthly_savings = float(row.loc[0, "monthly_savings"])
    loan_amount = float(row.loc[0, "loan_amount"])
    loan_term_months = float(row.loc[0, "loan_term_months"])
    prior_default = row.loc[0, "guarantor"]
    employment_type = row.loc[0, "employment_type"]
    other_debt_amount = float(row.loc[0, "other_debt_amount"])
    existing_loans = float(row.loc[0, "existing_loans"])
    guarantor = row.loc[0, "guarantor"]

    repayment_burden = loan_amount / max(monthly_income * loan_term_months, 1)
    savings_ratio = monthly_savings / max(monthly_income, 1)
    expense_ratio = monthly_expenses / max(monthly_income, 1)

    if repayment_burden > 0.20:
        reasons.append("Loan burden is high relative to income.")
    else:
        reasons.append("Loan burden is manageable relative to income.")

    if savings_ratio < 0.10:
        reasons.append("Savings are low compared with income.")
    else:
        reasons.append("Savings behavior supports repayment capacity.")

    if expense_ratio > 0.75:
        reasons.append("Monthly expenses consume a large share of income.")

    if other_debt_amount > 0 or existing_loans > 1:
        reasons.append("Existing debt obligations may weaken repayment capacity.")

    if employment_type == "Formal salaried":
        reasons.append("Formal salaried work supports more stable cash flow.")
    elif employment_type == "Self-employed":
        reasons.append("Self-employment can be viable but income may fluctuate.")
    elif employment_type == "Informal worker":
        reasons.append("Informal work may imply unstable earnings.")
    elif employment_type == "Unemployed":
        reasons.append("No current employment materially increases risk.")

    if guarantor == "Yes":
        reasons.append("A guarantor strengthens the application.")

    if pd_prob >= 0.60:
        reasons.append("Overall model-estimated default probability is very high.")
    elif pd_prob >= 0.40:
        reasons.append("Overall model-estimated default probability is elevated.")

    return reasons


def save_assessment(record: dict):
    df_new = pd.DataFrame([record])
    if DATA_FILE.exists():
        df_old = pd.read_csv(DATA_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(DATA_FILE, index=False)


def load_saved_data():
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()


def prepare_download(df: pd.DataFrame):
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


if not MODEL_FILE.exists():
    st.error("credit_model.pkl was not found in this folder. Put the trained model in the same folder as app.py.")
    st.stop()

model = load_model()
saved_df = load_saved_data()

header_col1, header_col2 = st.columns([1, 4])
with header_col1:
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=180)
    else:
        st.info("Save the company logo in this folder as logo.png")

with header_col2:
    st.markdown('<div class="main-title">Classic Capital Co-operative Society</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI Credit Scoring and Borrower Assessment Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">This version uses a trained machine learning model stored as credit_model.pkl.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

overview1, overview2, overview3, overview4 = st.columns(4)

with overview1:
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Total assessments</div><div class="metric-value">{}</div></div>'.format(len(saved_df)),
        unsafe_allow_html=True,
    )
with overview2:
    avg_pd = f"{saved_df['default_probability'].mean():.1%}" if not saved_df.empty else "0.0%"
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Average default probability</div><div class="metric-value">{}</div></div>'.format(avg_pd),
        unsafe_allow_html=True,
    )
with overview3:
    low_share = f"{(saved_df['risk_category'].eq('Low Risk').mean()):.1%}" if not saved_df.empty else "0.0%"
    st.markdown(
        '<div class="metric-card"><div class="metric-label">Low risk share</div><div class="metric-value">{}</div></div>'.format(low_share),
        unsafe_allow_html=True,
    )
with overview4:
    high_share = f"{(saved_df['risk_category'].isin(['High Risk', 'Very High Risk']).mean()):.1%}" if not saved_df.empty else "0.0%"
    st.markdown(
        '<div class="metric-card"><div class="metric-label">High risk share</div><div class="metric-value">{}</div></div>'.format(high_share),
        unsafe_allow_html=True,
    )

left_col, right_col = st.columns([1.25, 1])

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Borrower profile")

    with st.form("borrower_form"):
        borrower_name = st.text_input("Borrower name", placeholder="Enter borrower name")
        member_id = st.text_input("Member ID", placeholder="Enter member ID")
        application_date = st.date_input("Application date")
        loan_id = st.text_input("Loan ID", placeholder="Enter loan ID")

        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Widowed"])
        household_size = st.number_input("Household size", min_value=1, max_value=15, value=3)
        education_level = st.selectbox("Education level", ["Primary", "Secondary", "Tertiary"])

        employment_type = st.selectbox(
            "Employment type",
            ["Formal salaried", "Self-employed", "Informal worker", "Unemployed"],
        )
        years_employed = st.number_input("Years employed", min_value=0.0, max_value=40.0, value=3.0, step=0.5)
        business_owner = st.selectbox("Business owner", ["Yes", "No"])

        monthly_income = st.number_input("Monthly income", min_value=1.0, value=1500.0, step=50.0)
        monthly_expenses = st.number_input("Monthly expenses", min_value=0.0, value=900.0, step=50.0)
        monthly_savings = st.number_input("Monthly savings", min_value=0.0, value=200.0, step=10.0)
        other_debt_amount = st.number_input("Other debt amount", min_value=0.0, value=0.0, step=50.0)
        existing_loans = st.number_input("Existing loans", min_value=0, max_value=10, value=1)
        collateral_value = st.number_input("Collateral value", min_value=0.0, value=5000.0, step=100.0)
        guarantor = st.selectbox("Has guarantor?", ["Yes", "No"])
        member_years = st.number_input("Years as cooperative member", min_value=0.0, max_value=40.0, value=2.0, step=0.5)

        loan_amount = st.number_input("Requested loan amount", min_value=0.0, value=3000.0, step=50.0)
        loan_term_months = st.number_input("Loan term in months", min_value=1, max_value=60, value=12)
        loan_purpose = st.selectbox("Loan purpose", ["Business", "Trading", "Education", "Housing", "Agriculture", "Consumption"])
        interest_rate = st.number_input("Interest rate", min_value=0.0, max_value=1.0, value=0.18, step=0.01, format="%.2f")
        repayment_frequency = st.selectbox("Repayment frequency", ["Monthly", "Weekly", "Quarterly"])

        submitted = st.form_submit_button("Evaluate borrower")

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### Assessment result")

    if submitted:
        model_input = pd.DataFrame(
            [{
                "age": age,
                "gender": gender,
                "marital_status": marital_status,
                "household_size": household_size,
                "education_level": education_level,
                "employment_type": employment_type,
                "years_employed": years_employed,
                "business_owner": business_owner,
                "monthly_income": monthly_income,
                "monthly_expenses": monthly_expenses,
                "monthly_savings": monthly_savings,
                "other_debt_amount": other_debt_amount,
                "existing_loans": existing_loans,
                "collateral_value": collateral_value,
                "guarantor": guarantor,
                "member_years": member_years,
                "loan_amount": loan_amount,
                "loan_term_months": loan_term_months,
                "loan_purpose": loan_purpose,
                "interest_rate": interest_rate,
                "repayment_frequency": repayment_frequency,
            }]
        )

        pd_prob = float(model.predict_proba(model_input)[0, 1])
        risk_label, recommendation = classify_risk(pd_prob)
        explanations = build_explanations(model_input, pd_prob)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "borrower_name": borrower_name,
            "member_id": member_id,
            "application_date": str(application_date),
            "loan_id": loan_id,
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "household_size": household_size,
            "education_level": education_level,
            "employment_type": employment_type,
            "years_employed": years_employed,
            "business_owner": business_owner,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_expenses,
            "monthly_savings": monthly_savings,
            "other_debt_amount": other_debt_amount,
            "existing_loans": existing_loans,
            "collateral_value": collateral_value,
            "guarantor": guarantor,
            "member_years": member_years,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "loan_purpose": loan_purpose,
            "interest_rate": interest_rate,
            "repayment_frequency": repayment_frequency,
            "default_probability": round(pd_prob, 4),
            "risk_category": risk_label,
            "recommendation": recommendation,
        }
        save_assessment(record)
        saved_df = load_saved_data()

        st.success("Borrower assessment completed with trained model")
        st.markdown(
            '<div class="metric-label">Probability of default</div><div class="metric-value">{}</div>'.format(f"{pd_prob:.1%}"),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-label">Risk category</div><div class="metric-value {}">{}</div>'.format(risk_class_name(risk_label), risk_label),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-label">Recommendation</div><div class="metric-value">{}</div>'.format(recommendation),
            unsafe_allow_html=True,
        )

        st.markdown("### Key reasons")
        for item in explanations:
            st.write(f"• {item}")

        summary_df = pd.DataFrame(
            {
                "Field": list(record.keys()),
                "Value": list(record.values()),
            }
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.write("Complete the borrower form and click Evaluate borrower.")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])

with filter_col1:
    risk_filter = st.selectbox(
        "Filter by risk category",
        ["All", "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"],
    )
with filter_col2:
    search_name = st.text_input("Search borrower name", placeholder="Type a borrower name")
with filter_col3:
    st.download_button(
        "Download saved assessments",
        data=prepare_download(saved_df),
        file_name="borrower_assessments.csv",
        mime="text/csv",
        disabled=saved_df.empty,
    )

filtered_df = saved_df.copy()
if not filtered_df.empty:
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df["risk_category"] == risk_filter]
    if search_name.strip():
        filtered_df = filtered_df[
            filtered_df["borrower_name"].fillna("").str.contains(search_name.strip(), case=False, na=False)
        ]

st.markdown("### Saved records")
if filtered_df.empty:
    st.write("No matching records found.")
else:
    st.dataframe(filtered_df.sort_values("timestamp", ascending=False), use_container_width=True)