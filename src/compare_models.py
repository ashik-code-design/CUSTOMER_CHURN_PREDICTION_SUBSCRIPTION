"""Fair churn-model comparison. Run from the project root:
python src/compare_models.py

- Same stratified 60/20/20 train/validation/test split for every model.
- 5-fold stratified cross-validation on the train+validation rows.
- Thresholds are chosen on validation data only; final metrics use test data.
- The model that the Streamlit app serves is chosen on VALIDATION cost
  (never on the test set), then saved together with its own preprocessor
  and threshold so all three always match.
- Business costs are illustrative assumptions (see COST SETTINGS), not
  measured financial results.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from churn_features import (  # noqa: E402
    CATEGORICAL_COLUMNS, USE_ENGINEERED_FEATURES, prepare_features,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "customer_churn.csv"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- COST SETTINGS (edit with real business numbers) -------
FP_COST = 50.0          # PDF example: $50 retention discount given needlessly
CLV_MONTHS = 12         # assumed months of revenue lost when a churner is missed
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
RANDOM_STATE = 42
CV_FOLDS = 5

# 1. Load data
df = pd.read_csv(DATA_FILE).dropna(subset=["churn"]).copy()
if not {"customer_id", "churn"}.issubset(df.columns):
    raise ValueError("Dataset must contain customer_id and churn columns.")

FN_COST = round(float(df["monthly_charges"].mean()) * CLV_MONTHS, 2)

X = prepare_features(
    df.drop(columns=["customer_id", "churn"]),
    engineered=USE_ENGINEERED_FEATURES,
)
y = df["churn"].astype(int)
if set(y.unique()) != {0, 1}:
    raise ValueError("churn must contain binary labels 0 and 1.")

categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
numeric = [c for c in X.columns if c not in categorical]

# 2. Split 60% train, 20% validation, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25,
    random_state=RANDOM_STATE, stratify=y_train_val
)

print(f"Dataset: {df.shape}; train={len(X_train)}, "
      f"validation={len(X_val)}, test={len(X_test)}")
print("Churn distribution:", y.value_counts().sort_index().to_dict())
print(f"Engineered features used: {USE_ENGINEERED_FEATURES}")
print(f"Costs: false positive=${FP_COST:,.2f}, "
      f"false negative=${FN_COST:,.2f} ({CLV_MONTHS} x avg monthly charge)")

# 3. Preprocessors (fit on training data only)
def make_preprocessor(scale: bool):
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numerical", StandardScaler() if scale else "passthrough", numeric),
    ])

neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())

# name, estimator, scale numeric columns?, file slug
model_specs = [
    ("Logistic Regression",
     LogisticRegression(max_iter=2000, class_weight="balanced",
                        random_state=RANDOM_STATE), True, "logistic_regression"),
    ("Random Forest",
     RandomForestClassifier(n_estimators=300, max_depth=12,
                            min_samples_split=5, min_samples_leaf=2,
                            class_weight="balanced", random_state=RANDOM_STATE,
                            n_jobs=-1), False, "random_forest"),
    ("XGBoost",
     XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   scale_pos_weight=neg / max(pos, 1),
                   objective="binary:logistic", eval_metric="logloss",
                   random_state=RANDOM_STATE, n_jobs=-1), False, "xgboost"),
]

def cost_and_counts(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return FP_COST * fp + FN_COST * fn, int(tn), int(fp), int(fn), int(tp)

def make_row(name, y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    cost, tn, fp, fn, tp = cost_and_counts(y_true, pred)
    return {
        "Model": name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 Score": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Illustrative Cost": float(cost),
    }

# 4. Cross-validation (train+validation rows only; test set stays untouched)
print(f"\nRunning {CV_FOLDS}-fold stratified cross-validation...")
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, estimator, scale, _slug in model_specs:
    pipe = Pipeline([
        ("pre", make_preprocessor(scale)),
        ("model", clone(estimator)),
    ])
    scores = cross_validate(
        pipe, X_train_val, y_train_val, cv=skf,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        n_jobs=1,
    )
    cv_rows.append({
        "Model": name,
        "CV Folds": CV_FOLDS,
        "ROC-AUC Mean": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std(),
        "PR-AUC Mean": scores["test_pr_auc"].mean(),
        "PR-AUC Std": scores["test_pr_auc"].std(),
    })
cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUT_DIR / "cross_validation_results.csv", index=False)

# 5. Train on the 60% training rows, tune threshold on validation, test once
tuned_rows, default_rows, threshold_rows, candidates = [], [], [], []
for name, estimator, scale, slug in model_specs:
    print(f"Training {name}...")
    pre = make_preprocessor(scale)
    Xtr = pre.fit_transform(X_train)
    Xv = pre.transform(X_val)
    Xte = pre.transform(X_test)

    model = clone(estimator)
    model.fit(Xtr, y_train)
    val_prob = model.predict_proba(Xv)[:, 1]
    test_prob = model.predict_proba(Xte)[:, 1]

    # Threshold chosen with validation labels only. Tie-break toward 0.50.
    scored = []
    for threshold in THRESHOLDS:
        pred = (val_prob >= threshold).astype(int)
        cost, *_ = cost_and_counts(y_val, pred)
        scored.append((cost, abs(float(threshold) - 0.50), float(threshold)))
    val_cost, _, chosen = min(scored)

    tuned_rows.append(make_row(name, y_test, test_prob, chosen))
    default_rows.append(make_row(name, y_test, test_prob, 0.50))
    for threshold in THRESHOLDS:
        threshold_rows.append(make_row(name, y_test, test_prob, threshold))

    candidates.append({
        "name": name, "slug": slug, "model": model, "pre": pre,
        "threshold": chosen, "val_cost": float(val_cost),
        "val_pr_auc": float(average_precision_score(y_val, val_prob)),
    })

tuned = pd.DataFrame(tuned_rows)
default = pd.DataFrame(default_rows)
audit = pd.DataFrame(threshold_rows)

tuned_path = OUT_DIR / "model_comparison.csv"
default_path = OUT_DIR / "model_comparison_at_050.csv"
audit_path = OUT_DIR / "threshold_comparison_all_models.csv"
tuned.to_csv(tuned_path, index=False)
default.to_csv(default_path, index=False)
audit.to_csv(audit_path, index=False)

# 6. Pick the model the app will serve: lowest VALIDATION cost, then PR-AUC
best = min(candidates, key=lambda c: (c["val_cost"], -c["val_pr_auc"]))
joblib.dump(best["model"], OUT_DIR / "best_model.pkl")
joblib.dump(best["pre"], OUT_DIR / "best_preprocessor.pkl")
with open(OUT_DIR / "best_model_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model_name": best["name"],
        "model_kind": "linear" if best["slug"] == "logistic_regression" else "tree",
        "model_file": "best_model.pkl",
        "preprocessor_file": "best_preprocessor.pkl",
        "threshold": best["threshold"],
        "engineered_features": bool(USE_ENGINEERED_FEATURES),
        "selected_by": "lowest validation cost, then validation PR-AUC",
        "validation_cost": best["val_cost"],
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
        "costs_are_illustrative": True,
    }, f, indent=4)
with open(OUT_DIR / "cost_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "false_positive_cost": FP_COST,
        "false_negative_cost": FN_COST,
        "false_negative_basis": f"{CLV_MONTHS} x average monthly charge",
        "costs_are_illustrative": True,
    }, f, indent=4)

cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP",
        "Illustrative Cost"]
print("\nCROSS-VALIDATION (train+validation rows)")
print(cv_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nMODEL COMPARISON - VALIDATION-TUNED THRESHOLDS, HELD-OUT TEST SET")
print(tuned[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nMODEL COMPARISON - COMMON THRESHOLD 0.50, HELD-OUT TEST SET")
print(default[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nModel served by the app: {best['name']} "
      f"(threshold {best['threshold']:.2f}, chosen on validation data)")
print("\nSaved in models/: model_comparison.csv, model_comparison_at_050.csv, "
      "threshold_comparison_all_models.csv, cross_validation_results.csv, "
      "cost_config.json, best_model.pkl, best_preprocessor.pkl, "
      "best_model_config.json")
print("Note: costs are illustrative. Existing logistic_regression.pkl / "
      "threshold_config.json are NOT overwritten.")
