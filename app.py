import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.predict import predict_churn, engineer_features, load_model

st.set_page_config(
    page_title="Banking Capital Risk & Retention Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate CSS Theme
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .stMetric { background-color: #1a1f2c; border: 1px solid #2d3748; border-radius: 8px; padding: 12px; }
    div[data-testid="stExpander"] { border: 1px solid #2d3748; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic Portfolio Retention & Capital Intelligence")
st.caption("Enterprise-grade ML decision support system for retail banking deposit preservation.")

# Navigation Tabs
tab_diagnostics, tab_simulator, tab_portfolio, tab_audit = st.tabs([
    "🎯 Individual Client Diagnostic & Explainability",
    "💼 Portfolio Capital ROI Simulator",
    "📂 Batch File Scorer & Triage Queue",
    "📈 Model Calibration & Risk Surfaces"
])

# ==========================================
# TAB 1: INDIVIDUAL DIAGNOSTICS & WATERFALL
# ==========================================
with tab_diagnostics:
    col_input, col_view = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown("### 👤 Client Profile Attributes")
        
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Region", ["Germany", "France", "Spain"], index=0)
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            age = st.slider("Customer Age", 18, 90, 49)
            tenure = st.slider("Tenure (Years)", 0, 10, 4)
        with c2:
            num_of_products = st.selectbox("Active Products", [1, 2, 3, 4], index=2)
            has_credit_card = st.radio("Holds Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
            is_active_member = st.radio("Active Member Status", [1, 0], format_func=lambda x: "Active" if x == 1 else "Dormant", horizontal=True, index=1)
            creditscore = st.slider("Credit Score", 300, 850, 620)

        balance = st.number_input("Current Balance (€)", 0.0, 500000.0, 138000.0, step=5000.0)
        salary = st.number_input("Estimated Salary (€)", 0.0, 500000.0, 82000.0, step=5000.0)

    with col_view:
        customer_payload = {
            "creditscore": creditscore, "geography": geography, "gender": gender,
            "age": age, "tenure": tenure, "balance": balance,
            "num_of_products": num_of_products, "has_credit_card": has_credit_card,
            "is_active_member": is_active_member, "estimated_salary": salary
        }

        result = predict_churn(customer_payload)
        prob = result["churn_probability"]
        tier = result["retention_tier"]
        balance_exp = result["balance_exposure_eur"]

        st.markdown("### 📊 Decision Diagnostic")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Attrition Probability", f"{prob*100:.2f}%")
        m2.metric("Portfolio Tier", tier.replace("TIER_", "T").replace("_", " "))
        m3.metric("Balance at Risk", f"€{balance_exp:,.0f}")
        expected_loss = balance_exp * prob
        m4.metric("Risk-Weighted Exposure", f"€{expected_loss:,.0f}")

        # Plotly Half-Donut Gauge
        gauge_color = "#ef4444" if prob >= 0.50 else ("#f59e0b" if prob >= 0.35 else "#10b981")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'font': {'size': 28, 'color': "white"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 35], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [35, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {'line': {'color': "white", 'width': 3}, 'thickness': 0.8, 'value': 50}
            }
        ))
        fig_gauge.update_layout(height=210, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature Impact Breakdown (Waterfall Proxy)
        st.markdown("#### ⚖️ Local Risk Factor Attribution")
        # Approximate local feature contributions based on customer attributes
        drivers = []
        if num_of_products >= 3:
            drivers.append(("Product Saturation (>=3 Products)", +0.35))
        elif num_of_products == 2:
            drivers.append(("Optimal Products (2 Products)", -0.20))
        if age >= 45:
            drivers.append(("Age Factor (>=45 yrs)", +0.18))
        if is_active_member == 0:
            drivers.append(("Dormant Activity Flag", +0.15))
        else:
            drivers.append(("Active Member Loyalty", -0.12))
        if geography == "Germany":
            drivers.append(("Germany Regional Risk", +0.14))
        if balance > 100000:
            drivers.append(("High Capital Balance", +0.08))

        df_drivers = pd.DataFrame(drivers, columns=["Factor", "Impact"]).sort_values(by="Impact", ascending=True)
        colors = ["#ef4444" if x > 0 else "#10b981" for x in df_drivers["Impact"]]
        
        fig_waterfall = go.Figure(go.Bar(
            x=df_drivers["Impact"] * 100,
            y=df_drivers["Factor"],
            orientation='h',
            marker_color=colors
        ))
        fig_waterfall.update_layout(
            title="Estimated Risk Push (+) vs. Pull (-) Factors",
            xaxis_title="Impact on Probability Points (%)",
            height=240,
            margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

# ==========================================
# TAB 2: PORTFOLIO CAPITAL ROI SIMULATOR
# ==========================================
with tab_simulator:
    st.subheader("💼 Executive Retention Campaign ROI Estimator")
    st.write("Simulate portfolio return on investment (ROI) based on proactive relationship outreach.")

    col_sim_ctrl, col_sim_metric = st.columns([1, 2], gap="large")

    with col_sim_ctrl:
        st.markdown("##### Campaign Assumptions")
        total_test_churners = 407
        total_at_risk_cap = 36708330.81
        model_captured_cap = 27829242.11

        retention_success_rate = st.slider("Target Outreach Success Rate (%)", 5, 50, 20) / 100
        cost_per_contact = st.number_input("Outreach Cost per Flagged Client (€)", 10.0, 500.0, 75.0, step=5.0)

    with col_sim_metric:
        total_targeted = 583 # XGBoost flagged accounts (TP + FP)
        campaign_cost = total_targeted * cost_per_contact
        preserved_capital = model_captured_cap * retention_success_rate
        net_retained_value = preserved_capital - campaign_cost
        roi_ratio = (preserved_capital / campaign_cost) if campaign_cost > 0 else 0

        st.markdown("##### Executive Financial Summary")
        r1, r2, r3 = st.columns(3)
        r1.metric("Campaign Budget Required", f"€{campaign_cost:,.2f}")
        r2.metric("Preserved Portfolio Capital", f"€{preserved_capital:,.2f}")
        r3.metric("Net Capital Preserved", f"€{net_retained_value:,.2f}", delta=f"{roi_ratio:.1f}x ROI")

        # Interactive Waterfall of Financial Impact
        fig_fin = go.Figure(go.Waterfall(
            name="Portfolio Flow",
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Total At-Risk Capital", "Missed by Model", "Preserved by Outreach", "Net Capital Preserved"],
            textposition="outside",
            y=[total_at_risk_cap, -(total_at_risk_cap - model_captured_cap), preserved_capital, net_retained_value],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig_fin.update_layout(title="Capital Recovery Breakdown (€)", height=320, margin=dict(l=20, r=20, t=30, b=10))
        st.plotly_chart(fig_fin, use_container_width=True)

# ==========================================
# TAB 3: BATCH FILE SCORER
# ==========================================
with tab_portfolio:
    st.subheader("📂 Batch Portfolio Scoring & Retention Dispatch")
    st.write("Upload raw bank accounts to batch-score attrition and download prioritized CRM queues.")

    uploaded_file = st.file_uploader("Upload Account Cohort (.csv)", type=["csv"])
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Cohort loaded: **{len(df_batch):,} customers**.")

        if st.button("⚡ Score Cohort Now", type="primary"):
            with st.spinner("Executing XGBoost inference pipeline..."):
                results = [predict_churn(row.to_dict()) for _, row in df_batch.iterrows()]
                res_df = pd.DataFrame(results)
                final_batch_df = pd.concat([df_batch.reset_index(drop=True), res_df], axis=1)

                col_b1, col_b2, col_b3 = st.columns(3)
                total_flagged = (final_batch_df['churn_prediction'] == 1).sum()
                flagged_cap = final_batch_df[final_batch_df['churn_prediction'] == 1]['balance_exposure_eur'].sum()
                
                col_b1.metric("Total Cohort Evaluated", f"{len(final_batch_df):,}")
                col_b2.metric("Flagged for Retention Action", f"{total_flagged:,} ({total_flagged/len(final_batch_df)*100:.1f}%)")
                col_b3.metric("Flagged Capital Exposure", f"€{flagged_cap:,.2f}")

                # Tier distribution chart
                tier_counts = final_batch_df['retention_tier'].value_counts().reset_index()
                fig_tiers = px.pie(tier_counts, names='retention_tier', values='count', 
                                   title="Account Distribution by Retention Tier",
                                   color_discrete_sequence=px.colors.qualitative.Safe)
                fig_tiers.update_layout(height=280)
                st.plotly_chart(fig_tiers, use_container_width=True)

                st.dataframe(final_batch_df, use_container_width=True)

                st.download_button(
                    label="📥 Export Triage Action List (CSV)",
                    data=final_batch_df.to_csv(index=False).encode('utf-8'),
                    file_name="crm_retention_action_queue.csv",
                    mime="text/csv"
                )

# ==========================================
# TAB 4: CALIBRATION & RISK SURFACES
# ==========================================
with tab_audit:
    st.subheader("📈 Diagnostic Risk Surfaces & Production Metrics")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if os.path.exists("assets/feature_importance.png"):
            st.image("assets/feature_importance.png", caption="Model Global Predictive Weights", use_container_width=True)
    with col_s2:
        if os.path.exists("assets/model_evaluation_metrics.png"):
            st.image("assets/model_evaluation_metrics.png", caption="ROC-AUC & Confusion Matrix", use_container_width=True)

    # Risk Surface simulation (Age vs Balance)
    st.markdown("#### 🌐 Attrition Probability Surface (Age vs. Balance)")
    ages = np.linspace(20, 70, 25)
    balances = np.linspace(0, 200000, 25)
    age_grid, bal_grid = np.meshgrid(ages, balances)

    # Simplified risk response surface based on model patterns
    risk_surface = 1 / (1 + np.exp(-(0.06 * (age_grid - 42) + 0.000008 * (bal_grid - 60000))))

    fig_surf = go.Figure(data=[go.Surface(z=risk_surface, x=ages, y=balances, colorscale='RdYlGn_r')])
    fig_surf.update_layout(
        title="Predictive Risk Gradient (Age x Balance)",
        scene=dict(xaxis_title="Age", yaxis_title="Balance (€)", zaxis_title="Churn Prob"),
        height=450, margin=dict(l=10, r=10, t=30, b=10)
    )
    st.plotly_chart(fig_surf, use_container_width=True)