"""Batch scoring job: score a customer file from the command line.

Run from the project root, AFTER python src/compare_models.py:

    python src/batch_scoring.py
    python src/batch_scoring.py --input data/new_customers.csv --output data/new_scores.csv

Input : CSV or Excel file with the model's required customer columns
        (customer_id is optional but recommended).
Output: one row per customer with
        customer_id, churn_probability, risk_tier, top_3_churn_reasons,
        model, scored_at
        (the same fields the project brief lists for the warehouse table).

Also appends one line per run to data/batch_scoring_log.csv (rows scored,
model, risk-tier counts, mean churn probability) so runs can be compared
over time. The script can be scheduled (see README); it uses the best model
exported by compare_models.py, with the original Logistic Regression files
as a fallback.
"""
import argparse
import json
import sys
from datetime import datetime
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

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
TRAINING_DATA_FILE = ROOT / "data" / "customer_churn.csv"
DEFAULT_INPUT = ROOT / "data" / "customer_churn.csv"
DEFAULT_OUTPUT = ROOT / "data" / "batch_scores.csv"
LOG_FILE = ROOT / "data" / "batch_scoring_log.csv"


# ============================================================
# LOAD MODEL (best model first, Logistic Regression as fallback)
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


# ============================================================
# HELPERS
# ============================================================

def read_customers(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Input file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    return pd.read_csv(path)


def to_dense(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def build_explainer(base_model, kind, preprocessor, engineered, fallback_df):
    import shap

    if kind != "linear":
        return shap.TreeExplainer(base_model)

    # Linear models need a real background sample (not the rows being scored).
    source = (
        pd.read_csv(TRAINING_DATA_FILE)
        if TRAINING_DATA_FILE.exists() else fallback_df
    )
    sample = source[REQUIRED_COLUMNS].dropna()
    sample = sample.sample(min(100, len(sample)), random_state=42)

    background = to_dense(
        preprocessor.transform(prepare_features(sample, engineered=engineered))
    )
    return shap.LinearExplainer(base_model, background)


def top_churn_reasons(shap_matrix, feature_names, k=3):
    """Top k features pushing each customer TOWARD churn."""
    names = [str(n).split("__", 1)[-1] for n in feature_names]
    reasons = []
    for row in shap_matrix:
        order = np.argsort(row)[::-1][:k]
        picked = [names[i] for i in order if row[i] > 0]
        reasons.append("; ".join(picked) if picked else "No strong churn drivers")
    return reasons


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Batch churn scoring job.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help="CSV/Excel file of customers to score")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="CSV file to write the scores to")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model, preprocessor, threshold, model_name, kind, engineered = (
        load_scoring_artifacts()
    )
    base_model = getattr(model, "base_model", model)

    customers = read_customers(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in customers.columns]
    if missing:
        sys.exit("Missing required columns: " + ", ".join(missing))

    required = customers[REQUIRED_COLUMNS]
    rows_with_gaps = int(required.isna().any(axis=1).sum())
    if rows_with_gaps:
        sys.exit(
            f"{rows_with_gaps} row(s) have missing values in required columns. "
            "Fill or remove them and run again."
        )

    features = prepare_features(required.copy(), engineered=engineered)
    transformed = preprocessor.transform(features)

    probabilities = model.predict_proba(transformed)[:, 1]
    risk_tier = assign_risk_tier(probabilities, threshold)

    # Top churn reasons from SHAP (explains the underlying classifier)
    explainer = build_explainer(
        base_model, kind, preprocessor, engineered, customers
    )
    shap_values = np.asarray(explainer(to_dense(transformed)).values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    reasons = top_churn_reasons(shap_values, preprocessor.get_feature_names_out())

    scored_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if "customer_id" in customers.columns:
        ids = customers["customer_id"].to_numpy()
    else:
        ids = np.arange(1, len(customers) + 1)

    result = pd.DataFrame({
        "customer_id": ids,
        "churn_probability": np.round(probabilities, 4),
        "risk_tier": risk_tier,
        "top_3_churn_reasons": reasons,
        "model": model_name,
        "scored_at": scored_at,
    }).sort_values("churn_probability", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    # One-line run log (for comparing runs over time)
    counts = result["risk_tier"].value_counts()
    log_row = pd.DataFrame([{
        "scored_at": scored_at,
        "input_file": input_path.name,
        "rows_scored": len(result),
        "model": model_name,
        "threshold": threshold,
        "high": int(counts.get("HIGH", 0)),
        "medium": int(counts.get("MEDIUM", 0)),
        "low": int(counts.get("LOW", 0)),
        "mean_churn_probability": round(float(probabilities.mean()), 4),
    }])
    log_row.to_csv(
        LOG_FILE, mode="a", index=False, header=not LOG_FILE.exists()
    )

    print("=" * 60)
    print("BATCH SCORING COMPLETED")
    print("=" * 60)
    print(f"Model          : {model_name} (threshold {threshold:.2f})")
    print(f"Customers      : {len(result):,}")
    print(f"Risk tiers     : HIGH {counts.get('HIGH', 0):,} | "
          f"MEDIUM {counts.get('MEDIUM', 0):,} | LOW {counts.get('LOW', 0):,}")
    print(f"Mean churn prob: {probabilities.mean():.2%}")
    print(f"Scores saved   : {output_path}")
    print(f"Run log        : {LOG_FILE}")


if __name__ == "__main__":
    main()