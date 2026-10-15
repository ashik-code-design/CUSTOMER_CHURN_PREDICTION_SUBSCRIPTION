"""Fair churn-model comparison. Run from the project root:
python src/compare_models.py

All models use the same stratified 60/20/20 train/validation/test split.
Thresholds are chosen on validation data only; final metrics use test data.
Business costs below are illustrative, not measured financial costs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "customer_churn.csv"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP_COST = 10.0
FN_COST = 100.0
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.05), 2)
RANDOM_STATE = 42

# 1. Load data
df = pd.read_csv(DATA_FILE).dropna(subset=["churn"]).copy()
if not {"customer_id", "churn"}.issubset(df.columns):
    raise ValueError("Dataset must contain customer_id and churn columns.")
X = df.drop(columns=["customer_id", "churn"])
y = df["churn"].astype(int)
if set(y.unique()) != {0, 1}:
    raise ValueError("churn must contain binary labels 0 and 1.")

categorical = [c for c in
               ["gender", "contract_type", "internet_service", "payment_method"]
               if c in X.columns]
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

# 3. Fit preprocessors on training data only
log_pre = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", StandardScaler(), numeric),
])
tree_pre = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numeric),
])
Xtr_log = log_pre.fit_transform(X_train)
Xv_log = log_pre.transform(X_val)
Xte_log = log_pre.transform(X_test)
Xtr_tree = tree_pre.fit_transform(X_train)
Xv_tree = tree_pre.transform(X_val)
Xte_tree = tree_pre.transform(X_test)

# 4. Train all models on the same training rows
neg = int((y_train == 0).sum())
pos = int((y_train == 1).sum())
models = [
    ("Logistic Regression",
     LogisticRegression(max_iter=2000, class_weight="balanced",
                        random_state=RANDOM_STATE),
     Xtr_log, Xv_log, Xte_log),
    ("Random Forest",
     RandomForestClassifier(n_estimators=300, max_depth=12,
                            min_samples_split=5, min_samples_leaf=2,
                            class_weight="balanced", random_state=RANDOM_STATE,
                            n_jobs=-1),
     Xtr_tree, Xv_tree, Xte_tree),
    ("XGBoost",
     XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8,
                   scale_pos_weight=neg / max(pos, 1),
                   objective="binary:logistic", eval_metric="logloss",
                   random_state=RANDOM_STATE, n_jobs=-1),
     Xtr_tree, Xv_tree, Xte_tree),
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

tuned_rows, default_rows, threshold_rows = [], [], []
for name, model, Xtr, Xv, Xte in models:
    print(f"Training {name}...")
    model.fit(Xtr, y_train)
    val_prob = model.predict_proba(Xv)[:, 1]
    test_prob = model.predict_proba(Xte)[:, 1]

    # Select threshold using validation labels only. Tie-break toward 0.50.
    scored = []
    for threshold in THRESHOLDS:
        pred = (val_prob >= threshold).astype(int)
        cost, *_ = cost_and_counts(y_val, pred)
        scored.append((cost, abs(float(threshold) - 0.50), float(threshold)))
    chosen = min(scored)[2]

    tuned_rows.append(make_row(name, y_test, test_prob, chosen))
    default_rows.append(make_row(name, y_test, test_prob, 0.50))
    for threshold in THRESHOLDS:
        threshold_rows.append(make_row(name, y_test, test_prob, threshold))

tuned = pd.DataFrame(tuned_rows)
default = pd.DataFrame(default_rows)
audit = pd.DataFrame(threshold_rows)

tuned_path = OUT_DIR / "model_comparison.csv"
default_path = OUT_DIR / "model_comparison_at_050.csv"
audit_path = OUT_DIR / "threshold_comparison_all_models.csv"
tuned.to_csv(tuned_path, index=False)
default.to_csv(default_path, index=False)
audit.to_csv(audit_path, index=False)

cols = ["Model", "Threshold", "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP",
        "Illustrative Cost"]
print("\nMODEL COMPARISON — VALIDATION-TUNED THRESHOLDS, HELD-OUT TEST SET")
print(tuned[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nMODEL COMPARISON — COMMON THRESHOLD 0.50, HELD-OUT TEST SET")
print(default[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved:")
print(tuned_path.relative_to(ROOT))
print(default_path.relative_to(ROOT))
print(audit_path.relative_to(ROOT))
print("\nNote: FP/FN costs are illustrative. This script does not overwrite trained")
print("model files or threshold_config.json used by the Streamlit application.")
