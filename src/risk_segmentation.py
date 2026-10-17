"""Score every customer and assign a risk tier.

Run from the project root, AFTER python src/compare_models.py:
    python src/risk_segmentation.py

Uses the best model exported by compare_models.py (best_model.pkl,
best_preprocessor.pkl, best_model_config.json). If those files do not exist
it falls back to the original Logistic Regression files.

Risk tiers use the SAME rule as the Streamlit app (churn_features.py):
    HIGH   : probability >= tuned decision threshold
    MEDIUM : probability >= half of the threshold
    LOW    : below that
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_features import (  # noqa: E402
    REQUIRED_COLUMNS,
    assign_risk_tier,
    prepare_features,
)


# ============================================================
# FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_FILE = ROOT / "data" / "risk_segmented_customers.csv"


# ============================================================
# LOAD MODEL (best model first, original Logistic Regression as fallback)
# ============================================================

def load_scoring_artifacts():
    config_path = MODEL_DIR / "best_model_config.json"

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return (
            joblib.load(MODEL_DIR / config["model_file"]),
            joblib.load(MODEL_DIR / config["preprocessor_file"]),
            float(config.get("threshold", 0.50)),
            config.get("model_name", "Best model"),
            bool(config.get("engineered_features", False)),
        )

    threshold = 0.50
    threshold_path = MODEL_DIR / "threshold_config.json"
    if threshold_path.exists():
        threshold = float(
            json.loads(threshold_path.read_text(encoding="utf-8"))
            .get("threshold", 0.50)
        )

    return (
        joblib.load(MODEL_DIR / "logistic_regression.pkl"),
        joblib.load(MODEL_DIR / "model_preprocessor.pkl"),
        threshold,
        "Logistic Regression",
        False,
    )


model, preprocessor, threshold, model_name, engineered = load_scoring_artifacts()

print(f"Model: {model_name} | threshold: {threshold:.2f} | "
      f"engineered features: {engineered}")


# ============================================================
# LOAD DATA AND PREDICT
# ============================================================

df = pd.read_csv(DATA_FILE)

X = prepare_features(df[REQUIRED_COLUMNS].copy(), engineered=engineered)

churn_probability = model.predict_proba(
    preprocessor.transform(X)
)[:, 1]

df["churn_probability"] = churn_probability


# ============================================================
# RISK SEGMENT (shared rule)
# ============================================================

df["risk_level"] = assign_risk_tier(churn_probability, threshold)


# ============================================================
# BUSINESS ACTION
# ============================================================

def recommended_action(risk):

    if risk == "HIGH":
        return "Proactive intervention: CRM alert and call from a senior CSM, tailored with churn reasons"

    elif risk == "MEDIUM":
        return "Automated email drip campaign with an optional 15% discount"

    else:
        return "Normal customer engagement"


df["recommended_action"] = df["risk_level"].apply(recommended_action)


# ============================================================
# SORT BY RISK AND SAVE
# ============================================================

df = df.sort_values("churn_probability", ascending=False)

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 50)
print("RISK SEGMENTATION")
print("=" * 50)

print()
print(df["risk_level"].value_counts())

print()
print("Risk percentage:")
print((df["risk_level"].value_counts(normalize=True) * 100).round(2))

print()
print(
    "Average churn probability:",
    f"{df['churn_probability'].mean() * 100:.2f}%"
)

print()
print("Highest-risk customers:")
print(
    df[
        [
            "customer_id",
            "churn_probability",
            "risk_level",
            "monthly_charges",
            "satisfaction_score",
        ]
    ].head(10).to_string(index=False)
)

print()
print("=" * 50)
print(f"Saved: {OUTPUT_FILE.relative_to(ROOT)}")
print("=" * 50)
