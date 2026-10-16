"""Rank customers by churn risk AND estimated customer value.

Run from the project root, AFTER python src/risk_segmentation.py:
    python src/retention_priority.py

What changed: the old fixed cut-offs ($150 / $100 expected revenue at risk)
were set for the old fixed 30% / 60% risk bands. Risk tiers now follow the
tuned threshold, so the cut-offs are computed from the data instead:
a customer is CRITICAL (or MEDIUM-HIGH) when their expected revenue at risk
is in the top part of their own risk tier. Adjust the two quantiles below.
"""
import pandas as pd
import numpy as np


# ============================================================
# FILE PATHS AND SETTINGS
# ============================================================

INPUT_FILE = "data/risk_segmented_customers.csv"

OUTPUT_FILE = "data/retention_priority.csv"

# Share of each tier treated as "high value":
# 0.50 -> the top half (by expected revenue at risk) of that tier.
HIGH_TIER_VALUE_QUANTILE = 0.50
MEDIUM_TIER_VALUE_QUANTILE = 0.50


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# ESTIMATED CUSTOMER VALUE
# ============================================================
#
# Student-project approximation: monthly charges, scaled up for
# longer-tenure customers. A production system would use a full
# Customer Lifetime Value model.
# ============================================================

df["estimated_customer_value"] = (
    df["monthly_charges"] * (1 + df["tenure_months"] / 12)
)


# ============================================================
# EXPECTED REVENUE AT RISK AND PRIORITY SCORE
# ============================================================

df["expected_revenue_at_risk"] = (
    df["estimated_customer_value"] * df["churn_probability"]
)

df["retention_priority_score"] = (
    df["churn_probability"] * df["estimated_customer_value"]
)


# ============================================================
# DATA-DRIVEN VALUE CUT-OFFS (per risk tier)
# ============================================================

def tier_cutoff(tier, quantile):
    values = df.loc[df["risk_level"] == tier, "expected_revenue_at_risk"]
    return values.quantile(quantile) if len(values) > 0 else np.inf


high_cutoff = tier_cutoff("HIGH", HIGH_TIER_VALUE_QUANTILE)
medium_cutoff = tier_cutoff("MEDIUM", MEDIUM_TIER_VALUE_QUANTILE)


# ============================================================
# PRIORITY CATEGORY
# ============================================================

def classify_priority(row):

    if row["risk_level"] == "HIGH":

        if row["expected_revenue_at_risk"] >= high_cutoff:
            return "CRITICAL"

        return "HIGH"

    elif row["risk_level"] == "MEDIUM":

        if row["expected_revenue_at_risk"] >= medium_cutoff:
            return "MEDIUM-HIGH"

        return "MEDIUM"

    else:

        return "LOW"


df["retention_priority"] = df.apply(classify_priority, axis=1)


# ============================================================
# RECOMMENDED ACTION
# ============================================================

def recommended_action(priority):

    if priority == "CRITICAL":
        return "Immediate personalized retention intervention"

    elif priority == "HIGH":
        return "Priority retention campaign"

    elif priority == "MEDIUM-HIGH":
        return "Targeted retention campaign"

    elif priority == "MEDIUM":
        return "Monitor and engage customer"

    else:
        return "Normal customer engagement"


df["retention_action"] = df["retention_priority"].apply(recommended_action)


# ============================================================
# SORT AND SAVE
# ============================================================

df = df.sort_values("retention_priority_score", ascending=False)

df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("RETENTION PRIORITY ANALYSIS")
print("=" * 60)

print()
print(f"Value cut-off for CRITICAL (HIGH tier): ${high_cutoff:,.2f}")
print(f"Value cut-off for MEDIUM-HIGH (MEDIUM tier): ${medium_cutoff:,.2f}")

print()
print("Priority distribution:")
print(df["retention_priority"].value_counts())

print()
print(
    "Total estimated customer value:",
    f"${df['estimated_customer_value'].sum():,.2f}"
)

print()
print(
    "Total expected revenue at risk:",
    f"${df['expected_revenue_at_risk'].sum():,.2f}"
)

print()
print("Top 10 retention-priority customers:")
print()

top_customers = df[
    [
        "customer_id",
        "churn_probability",
        "risk_level",
        "monthly_charges",
        "tenure_months",
        "estimated_customer_value",
        "expected_revenue_at_risk",
        "retention_priority",
        "retention_action",
    ]
].head(10)

print(top_customers.to_string(index=False))

print()
print("=" * 60)
print(f"Saved: {OUTPUT_FILE}")
print("=" * 60)
