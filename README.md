# 🏦 Banking Portfolio Intelligence & Customer Retention Engine

An end-to-end Machine Learning and Decision Intelligence project using **PostgreSQL, Python, Scikit-Learn, and XGBoost** to identify customer attrition risk, calculate balance-at-risk exposure, and automate targeted retention workflows.

---

## 📌 Business Problem

In retail banking, customer attrition directly impairs capital reserves and Net Interest Margin (NIM). Standard churn modeling frequently relies on naive accuracy, which masks critical balance exposure: missing high-balance customers costs millions, while untargeted retention campaigns waste operational budget.

This project delivers a **cost-sensitive machine learning engine** calibrated to maximize **Churn Recall** and **Capital Coverage Ratio**, triaging accounts into prioritized operational action queues before capital exits the bank.

---

## 🔑 Key Results

* **Total At-Risk Capital Flagged**: Proactively captured **€27,829,242.11 (75.81%)** out of **€36.71M** in total departing balances across the unseen test cohort.
* **Attrition Recall Rate**: Flagged **75.00%** of all departing accounts (304 of 407 churners caught).
* **Model Discrimination**: XGBoost achieved an **ROC-AUC of 0.8670**, outperforming Random Forest (0.8641) and baseline Logistic Regression (0.8355).
* **Precision Lift**: Elevated precision to **52%** (vs 45% in weighted linear baselines), significantly reducing false-alarm outreach costs.
* **Product Saturation Threshold**: Discovered that customers holding **3 or 4 products carry an extreme >80% churn risk**, identifying friction in multi-product cross-selling.

---

## 📊 Portfolio Risk & Segment Exposure

| Account Segment / Risk Tier | Churn Probability | Capital at Risk (€) | Primary Portfolio Risk Profile | Recommended Action |
| :--- | :---: | :---: | :--- | :--- |
| **Tier 1: Critical Capital Risk** | $\ge 50\%$ & Bal $\ge €100\text{k}$ | **€18.42M** | High-balance accounts, multi-product saturation, mature age | 1-on-1 Dedicated Wealth Manager outreach & fee review |
| **Tier 2: High Attrition Risk** | $\ge 50\%$ & Bal $< €100\text{k}$ | **€9.41M** | Inactive accounts, single product, regional German exposure | Targeted digital retention discounts & app re-engagement |
| **Tier 3: Watchlist / Monitor** | $35\% - 50\%$ | **€5.12M** | Moderate engagement, zero-balance secondary holders | Product bundling review and balance incentive offers |
| **Tier 4: Stable / Retained** | $< 35\%$ | **€3.76M** | High activity, 2 products, active credit card holders | Standard loyalty programs and upsell opportunities |

---

## 📈 Model Performance & Evaluation

![Model Benchmark & Confusion Matrix](assets/model_evaluation_metrics.png)

### Model Benchmark Matrix

| Model Architecture | Churn Recall (1) | Churn Precision (1) | Churn F1-Score | Overall Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression** | 0.36 | **0.72** | 0.48 | **0.84** | 0.8355 |
| **Weighted Logistic Regression** | **0.75** | 0.45 | 0.56 | 0.76 | 0.8362 |
| **Random Forest Classifier** | 0.73 | 0.51 | 0.60 | 0.80 | 0.8641 |
| **XGBoost Classifier (Production)** | **0.75** | 0.52 | **0.61** | 0.81 | **0.8670** |

---

## 🔍 Predictive Drivers & Risk Factors

![Feature Importance](assets/feature_importance.png)

1. **`num_of_products` (Primary Risk Determinant)**: While holding 2 products represents optimal retention (7% churn), moving to 3 or 4 products spikes churn past 80%, indicating customer fatigue with overlapping fees or poor bundled service.
2. **`is_zero_balance` (Secondary Indicator)**: Engineered zero-balance indicators capture secondary account behavior prone to silent attrition.
3. **`age` (Cohort Transition)**: Attrition climbs sharply between ages 45 and 60, reflecting wealth migration during life-stage transitions.
4. **`geography_Germany`**: German accounts exhibit double the churn velocity of French and Spanish counterparts, identifying regional operational friction.

---

## 💡 Strategic Recommendations

* **Audit 3+ Product Bundles**: Immediate review of terms, overlapping fees, and user experience for clients holding 3 or 4 banking products.
* **Tier-1 Executive Priority Queue**: Implement automated ticketing to route flagged Tier 1 accounts (>€100k balance) directly to relationship managers within 48 hours.
* **German Regional Strategy**: Investigate customer support and competitor pricing pressures in the German market to curb localized outflow.
* **Preventive Re-engagement for Inactive Members**: Automated credit card reward incentives triggered when accounts drop below 3 monthly transactions.

---

## ⚙️ Architecture & Pipeline

```text
├── assets/                  # High-res charts, evaluation plots, and previews
│   ├── feature_importance.png
│   └── model_evaluation_metrics.png
├── data/
│   ├── raw/                 # Original banking records (Git ignored)
│   └── processed/           # Scaled splits and pipeline artifacts (Git ignored)
├── models/                  # Serialized XGBoost model artifacts (Git ignored)
├── notebook/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering_pipeline.ipynb
│   └── 03_predictive_churn_modeling.ipynb
├── src/
│   └── predict.py           # Standalone real-time inference & tiering logic
├── .gitignore
└── README.md