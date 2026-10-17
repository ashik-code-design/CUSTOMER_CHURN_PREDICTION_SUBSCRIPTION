"""SHAP explainability plots for the model the app serves.

Run from the project root, AFTER python src/compare_models.py:
    python src/shap_analysis.py

Uses the best model exported by compare_models.py. If those files do not
exist it falls back to the original Logistic Regression files. Explains the
same held-out test rows used in compare_models.py (same split settings).
"""
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_features import REQUIRED_COLUMNS, prepare_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
SHAP_DIR = ROOT / "data" / "shap"
os.makedirs(SHAP_DIR, exist_ok=True)


# ============================================================
# 1. LOAD MODEL (best model first, Logistic Regression as fallback)
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
            config.get("model_kind", "tree"),
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
        "linear",
        False,
    )


model, preprocessor, threshold, model_name, model_kind, engineered = (
    load_scoring_artifacts()
)

# SHAP explains the underlying classifier (not the calibration wrapper)
base_model = getattr(model, "base_model", model)

print(f"Model: {model_name} ({model_kind}) | threshold: {threshold:.2f} | "
      f"engineered features: {engineered}")


# ============================================================
# 2. LOAD DATA AND RECREATE THE SAME TRAIN / TEST SPLIT
# ============================================================

df = pd.read_csv(DATA_FILE).dropna(subset=["churn"]).copy()

X = prepare_features(
    df.drop(columns=["customer_id", "churn"]),
    engineered=engineered,
)
y = df["churn"].astype(int)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Dataset loaded:", df.shape, "| test rows:", len(X_test))


# ============================================================
# 3. PREPROCESS AND CLEAN FEATURE NAMES
# ============================================================

def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


feature_names = [
    name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
]

X_test_dense = to_dense(preprocessor.transform(X_test))
X_background = to_dense(
    preprocessor.transform(
        X_train_val.sample(min(100, len(X_train_val)), random_state=42)
    )
)


# ============================================================
# 4. CREATE SHAP EXPLAINER
# ============================================================

if model_kind == "linear":
    explainer = shap.LinearExplainer(base_model, X_background)
else:
    explainer = shap.TreeExplainer(base_model)

raw = explainer(X_test_dense)

values = np.asarray(raw.values)
base_values = np.asarray(raw.base_values)

if values.ndim == 3:  # some tree models return one slice per class
    values = values[:, :, 1]
    if base_values.ndim == 2:
        base_values = base_values[:, 1]

shap_values = shap.Explanation(
    values=values,
    base_values=base_values,
    data=X_test_dense,
    feature_names=feature_names,
)

print("SHAP values created successfully!")


# ============================================================
# 5. GLOBAL SHAP BAR PLOT
# ============================================================

print("\nCreating global SHAP bar plot...")

plt.figure()
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title(f"Global Feature Importance - SHAP ({model_name})")
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_global_bar.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: data/shap/shap_global_bar.png")


# ============================================================
# 6. GLOBAL SHAP BEESWARM PLOT
# ============================================================

print("Creating SHAP beeswarm plot...")

plt.figure()
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title(f"SHAP Feature Impact on Churn ({model_name})")
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: data/shap/shap_beeswarm.png")


# ============================================================
# 7. LOCAL WATERFALL PLOT
# ============================================================

sample_index = 0

print("\nCreating local explanation...")
print("Explaining test customer:", sample_index)

plt.figure()
shap.plots.waterfall(shap_values[sample_index], max_display=12, show=False)
plt.title(f"Local SHAP Explanation - Customer ({model_name})")
plt.tight_layout()
plt.savefig(SHAP_DIR / "shap_local_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: data/shap/shap_local_waterfall.png")


# ============================================================
# 8. PREDICT SAMPLE CUSTOMER (using the tuned threshold)
# ============================================================

sample_data = X_test.iloc[[sample_index]]

probability = model.predict_proba(
    to_dense(preprocessor.transform(sample_data))
)[0][1]

prediction = int(probability >= threshold)

print("\n")
print("=" * 60)
print("CUSTOMER CHURN PREDICTION")
print("=" * 60)

print("Prediction:", "CHURN" if prediction == 1 else "NOT CHURN")
print(f"Decision threshold: {threshold:.2f}")
print(f"Churn Probability: {probability:.4f}")
print(f"Churn Probability: {probability * 100:.2f}%")

print("\nCustomer information:")
print("-" * 60)
print(sample_data[REQUIRED_COLUMNS].to_string(index=False))


# ============================================================
# 9. TOP SHAP FEATURES FOR THIS CUSTOMER
# ============================================================

print("\nTop factors influencing this prediction:")
print("-" * 60)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values[sample_index].values,
})

feature_importance["Absolute SHAP"] = feature_importance["SHAP Value"].abs()

feature_importance = (
    feature_importance
    .sort_values("Absolute SHAP", ascending=False)
    .head(10)
)

print(feature_importance[["Feature", "SHAP Value"]].to_string(index=False))

feature_importance.to_csv(SHAP_DIR / "shap_local_features.csv", index=False)

print("\nSaved: data/shap/shap_local_features.csv")


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("1. data/shap/shap_global_bar.png")
print("2. data/shap/shap_beeswarm.png")
print("3. data/shap/shap_local_waterfall.png")
print("4. data/shap/shap_local_features.csv")
