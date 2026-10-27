import os
import joblib
import numpy as np
import pandas as pd


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost_churn_model.joblib")


def load_model():
    """Load the trained XGBoost model artifact."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train model first.")
    return joblib.load(MODEL_PATH)


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply feature transformations to align raw customer input with the training schema."""
    df = df_raw.copy()

    # 1. Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # 2. Financial & Behavioral Ratios
    df['balance_salary_ratio'] = df['balance'] / df['estimated_salary'].replace(0, np.nan)
    df['balance_salary_ratio'] = df['balance_salary_ratio'].fillna(0).round(4)
    df['tenure_age_ratio'] = (df['tenure'] / df['age']).round(4)
    df['is_high_product_risk'] = df['num_of_products'].apply(lambda x: 1 if x >= 3 else 0)
    df['is_zero_balance'] = (df['balance'] == 0).astype(int)

    # 3. Categorical Encodings (match training one-hot dummy structure)
    df['geography_Germany'] = (df['geography'] == 'Germany').astype(int)
    df['geography_Spain'] = (df['geography'] == 'Spain').astype(int)
    df['gender_Male'] = (df['gender'] == 'Male').astype(int)

    # 4. Enforce expected feature schema and ordering
    feature_columns = [
        'creditscore', 'age', 'tenure', 'balance', 'num_of_products',
        'has_credit_card', 'is_active_member', 'estimated_salary',
        'balance_salary_ratio', 'tenure_age_ratio', 'is_high_product_risk',
        'is_zero_balance', 'geography_Germany', 'geography_Spain', 'gender_Male'
    ]

    return df[feature_columns]


def assign_retention_tier(prob: float, balance: float) -> str:
    """Categorize customer risk based on attrition probability and capital exposure."""
    if prob >= 0.50 and balance >= 100000:
        return "TIER_1_CRITICAL_CAPITAL_RISK"
    elif prob >= 0.50:
        return "TIER_2_HIGH_ATTRITION_RISK"
    elif prob >= 0.35:
        return "TIER_3_WATCHLIST_MONITOR"
    else:
        return "TIER_4_LOW_RISK"


def predict_churn(customer_data: dict) -> dict:
    """Predict churn probability and retention priority for an individual customer profile."""
    model = load_model()
    raw_df = pd.DataFrame([customer_data])
    processed_df = engineer_features(raw_df)

    churn_prob = float(model.predict_proba(processed_df)[:, 1][0])
    prediction = int(churn_prob >= 0.50)
    tier = assign_retention_tier(churn_prob, raw_df['balance'].iloc[0])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(churn_prob, 4),
        "retention_tier": tier,
        "balance_exposure_eur": float(raw_df['balance'].iloc[0])
    }


if __name__ == "__main__":
    # Test Scenario A: High-balance German customer with multiple products
    high_risk_profile = {
        "creditscore": 620,
        "geography": "Germany",
        "gender": "Female",
        "age": 52,
        "tenure": 3,
        "balance": 135000.00,
        "num_of_products": 3,
        "has_credit_card": 1,
        "is_active_member": 0,
        "estimated_salary": 95000.00
    }

    # Test Scenario B: Low-risk young customer in France
    low_risk_profile = {
        "creditscore": 710,
        "geography": "France",
        "gender": "Male",
        "age": 28,
        "tenure": 5,
        "balance": 25000.00,
        "num_of_products": 2,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 60000.00
    }

    print("=== HIGH-RISK PORTFOLIO PROFILE ===")
    print(predict_churn(high_risk_profile))

    print("\n=== LOW-RISK PORTFOLIO PROFILE ===")
    print(predict_churn(low_risk_profile))