import streamlit as st
import pandas as pd
from src.predict import predict_churn

st.set_page_config(
    page_title="Banking Customer Retention Engine",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Customer Churn & Retention Intelligence")
st.markdown(
    "Proactively evaluate client attrition risk, assess balance-at-risk exposure, "
    "and identify targeted retention interventions."
)

st.markdown("---")

# Layout: Form on the left, Predictions & ROI on the right
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Customer Profile Parameters")
    
    geography = st.selectbox("Country / Region", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", min_value=18, max_value=90, value=42)
    creditscore = st.slider("Credit Score", min_value=300, max_value=850, value=650)
    tenure = st.slider("Tenure (Years with Bank)", min_value=0, max_value=10, value=4)
    balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=500000.0, value=85000.0, step=1000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=0)
    estimated_salary = st.number_input("Estimated Annual Salary (€)", min_value=0.0, max_value=500000.0, value=65000.0, step=1000.0)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        has_credit_card = st.radio("Holds Active Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with sub_col2:
        is_active_member = st.radio("Active Banking Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

    submit_button = st.button("Run Retention Assessment", use_container_width=True)

with col2:
    st.subheader("📊 Model Assessment & Risk Exposure")
    
    if submit_button:
        customer_payload = {
            "creditscore": creditscore,
            "geography": geography,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "num_of_products": num_of_products,
            "has_credit_card": has_credit_card,
            "is_active_member": is_active_member,
            "estimated_salary": estimated_salary
        }
        
        result = predict_churn(customer_payload)
        prob = result["churn_probability"]
        prediction = result["churn_prediction"]
        tier = result["retention_tier"]
        balance_exp = result["balance_exposure_eur"]

        # Metric Cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Attrition Probability", f"{prob * 100:.2f}%")
        metric_col2.metric("Predicted Outcome", "Churn Risk" if prediction == 1 else "Retained")
        metric_col3.metric("Capital at Stake", f"€{balance_exp:,.2f}")

        st.progress(prob)

        # Operational Tiers & Recommendations
        if tier == "TIER_1_CRITICAL_CAPITAL_RISK":
            st.error(f"🚨 **Action Tier: {tier}**")
            st.warning(
                "**Preservation Protocol**: Balance exposure exceeds €100,000. "
                "Trigger immediate notification to dedicated Senior Wealth Relationship Manager within 24 hours. "
                "Conduct full fee review and consider high-yield deposit restructuring."
            )
        elif tier == "TIER_2_HIGH_ATTRITION_RISK":
            st.warning(f"⚠️ **Action Tier: {tier}**")
            st.info(
                "**Digital Retention Protocol**: Customer displays elevated probability of attrition. "
                "Enroll in automated credit card cashback campaign and issue localized branch engagement survey."
            )
        elif tier == "TIER_3_WATCHLIST_MONITOR":
            st.info(f"🔍 **Action Tier: {tier}**")
            st.write(
                "**Watchlist Alert**: Moderate attrition indicators detected. "
                "Monitor for transaction volume drops or dormant balance shifts over the next 30 days."
            )
        else:
            st.success(f"✅ **Action Tier: {tier}**")
            st.write(
                "**Stable Client**: Low attrition probability. "
                "Eligible for cross-sell recommendations (wealth/investment products) and standard loyalty initiatives."
            )
    else:
        st.info("Adjust customer attributes on the left and click **Run Retention Assessment** to evaluate churn risk.")