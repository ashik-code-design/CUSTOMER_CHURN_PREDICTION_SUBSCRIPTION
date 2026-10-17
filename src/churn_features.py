"""Shared helpers used by compare_models.py, app.py and any scoring script.

Keeping feature engineering and risk-tier rules in ONE place guarantees that
training and the Streamlit app always treat a customer the same way.
"""
import numpy as np
import pandas as pd

# Set to False to train/score with only the original columns.
USE_ENGINEERED_FEATURES = True

CATEGORICAL_COLUMNS = [
    "gender",
    "contract_type",
    "internet_service",
    "payment_method",
]

BASE_NUMERIC_COLUMNS = [
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
    "satisfaction_score",
]

ENGINEERED_COLUMNS = [
    "support_friction_proxy",
    "usage_hours_per_login",
    "inactive_14d",
    "charge_vs_avg_ratio",
]

REQUIRED_COLUMNS = CATEGORICAL_COLUMNS + BASE_NUMERIC_COLUMNS


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural proxy features derived from the existing columns.

    These are single-snapshot proxies, NOT true 7-day vs 30-day velocity
    features (the dataset has no time series).

    support_friction_proxy : tickets per login. Approximates the 'support
                             friction index' (no resolution-time data exists).
    usage_hours_per_login  : depth of engagement per session.
    inactive_14d           : 1 if no login for 14+ days (inactivity signal).
    charge_vs_avg_ratio    : current monthly charge vs the customer's average
                             historical monthly charge (price-change proxy).
    """
    out = data.copy()

    out["support_friction_proxy"] = (
        out["support_tickets_30d"] / (out["logins_30d"] + 1.0)
    )

    out["usage_hours_per_login"] = (
        out["usage_hours_30d"] / (out["logins_30d"] + 1.0)
    )

    out["inactive_14d"] = (out["last_login_days"] >= 14).astype(int)

    average_charge = out["total_charges"] / out["tenure_months"].clip(lower=1)
    out["charge_vs_avg_ratio"] = (
        out["monthly_charges"] / average_charge.replace(0, np.nan)
    )
    out["charge_vs_avg_ratio"] = (
        out["charge_vs_avg_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
        .clip(0, 10)
    )

    return out


def prepare_features(data: pd.DataFrame, engineered: bool = True) -> pd.DataFrame:
    """Return the columns the model expects, adding engineered ones if used."""
    if engineered:
        return add_features(data)
    return data.copy()


def assign_risk_tier(probability, threshold):
    """Single risk rule used everywhere (app, batch, segmentation scripts).

    HIGH   : probability >= tuned decision threshold  (model says 'Churn')
    MEDIUM : probability >= half of the threshold
    LOW    : below that
    """
    probability = np.asarray(probability, dtype=float)
    return np.where(
        probability >= threshold,
        "HIGH",
        np.where(probability >= threshold / 2.0, "MEDIUM", "LOW"),
    )


# ----------------------------------------------------------------------
# Probability calibration
# ----------------------------------------------------------------------
# Class weighting (class_weight="balanced", scale_pos_weight) makes models
# over-predict churn probabilities. A Platt (sigmoid) calibrator fitted on
# the validation set maps the raw scores back to realistic probabilities.
# The wrapper lives in THIS module so saved model files can always be loaded
# by app.py and the scripts (they all import churn_features first).

def logit_clip(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability))


class CalibratedChurnModel:
    """A fitted classifier plus a Platt calibrator, behaving like one model.

    predict_proba() returns CALIBRATED probabilities. The untouched
    classifier is available as .base_model (used for SHAP explanations).
    """

    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw = self.base_model.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict_proba(
            logit_clip(raw).reshape(-1, 1)
        )[:, 1]
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
