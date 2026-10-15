import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATA_PATH = "data/customer_churn.csv"
MODEL_DIR = "models"

# Illustrative costs only; replace with real business estimates.
FALSE_POSITIVE_COST = 10.0
FALSE_NEGATIVE_COST = 100.0

os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# 2. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

print("\nChurn distribution:")
print(y.value_counts())

# ==================================================
# 3. DEFINE FEATURES
# ==================================================

categorical_columns = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method"
]

numerical_columns = [
    "age",
    "senior_citizen",
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "auto_payment",
    "discount",
    "support_tickets_30d",
    "logins_30d",
    "usage_hours_30d",
    "last_login_days",
    "satisfaction_score"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

# ==================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ==================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\nData split completed!")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

# ==================================================
# 5. PREPROCESS DATA
# ==================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# ==================================================
# 6. TRAIN CLASS-WEIGHTED MODEL
# ==================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

print("\nTraining Logistic Regression...")
model.fit(X_train_processed, y_train)

print("Training completed!")

# ==================================================
# 7. SELECT THRESHOLD ON VALIDATION DATA
# ==================================================

val_probabilities = model.predict_proba(X_val_processed)[:, 1]

best_threshold = 0.50
best_cost = float("inf")

threshold_results = []

for threshold in [i / 100 for i in range(10, 91, 5)]:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        val_predictions,
        labels=[0, 1]
    ).ravel()

    total_cost = (
        fp * FALSE_POSITIVE_COST
        + fn * FALSE_NEGATIVE_COST
    )

    threshold_results.append({
        "threshold": threshold,
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "estimated_cost": float(total_cost)
    })

    if total_cost < best_cost:
        best_cost = total_cost
        best_threshold = threshold

print("\nThreshold selection using validation data")
print("------------------------------------------")
print("Selected threshold:", best_threshold)
print("Estimated validation cost: $", round(best_cost, 2))

# ==================================================
# 8. FINAL TEST EVALUATION
# ==================================================

test_probabilities = model.predict_proba(X_test_processed)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)
roc_auc = roc_auc_score(y_test, test_probabilities)
pr_auc = average_precision_score(y_test, test_probabilities)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

test_cost = (
    fp * FALSE_POSITIVE_COST
    + fn * FALSE_NEGATIVE_COST
)

print("\nFINAL TEST RESULTS")
print("=" * 45)
print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"Test estimated cost: ${test_cost:.2f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))

# ==================================================
# 9. SAVE MODEL AND PREPROCESSOR
# ==================================================

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

joblib.dump(
    preprocessor,
    os.path.join(MODEL_DIR, "model_preprocessor.pkl")
)

with open(
    os.path.join(MODEL_DIR, "threshold_config.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump({
        "threshold": best_threshold,
        "false_positive_cost": FALSE_POSITIVE_COST,
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "threshold_selection": "validation set",
        "costs_are_illustrative": True
    }, file, indent=4)

pd.DataFrame(threshold_results).to_csv(
    os.path.join(MODEL_DIR, "threshold_comparison.csv"),
    index=False
)

print("\nSaved successfully:")
print("models/logistic_regression.pkl")
print("models/model_preprocessor.pkl")
print("models/threshold_config.json")
print("models/threshold_comparison.csv")