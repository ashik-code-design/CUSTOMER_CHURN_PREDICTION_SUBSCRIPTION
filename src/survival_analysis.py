"""Cohort and Kaplan-Meier survival analysis.

Run from the project root:
    python src/survival_analysis.py

New: a churn-rate HEATMAP of tenure cohort x contract type
(data/cohort_heatmap.csv and data/cohort_heatmap.png).

Honest note: the dataset has no signup/acquisition date, so true
"acquisition-month" cohorts (as described in the project brief) cannot be
built. Tenure cohorts are used instead, and the heatmap adds a second
dimension (contract type) to show where churn concentrates.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# Cells with fewer customers than this are greyed out as unreliable.
MIN_CELL_SIZE = 10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/customer_churn.csv")


# ============================================================
# COHORT ANALYSIS
# ============================================================

# Create tenure cohorts
df["tenure_cohort"] = pd.cut(
    df["tenure_months"],
    bins=[-1, 6, 12, 24, 36, 60, 120],
    labels=[
        "0-6 Months",
        "7-12 Months",
        "13-24 Months",
        "25-36 Months",
        "37-60 Months",
        "60+ Months"
    ]
)


cohort_analysis = (
    df.groupby("tenure_cohort", observed=True)
    .agg(
        customers=("customer_id", "count"),
        churned=("churn", "sum"),
        avg_monthly_charges=("monthly_charges", "mean"),
        avg_satisfaction=("satisfaction_score", "mean")
    )
    .reset_index()
)

cohort_analysis["churn_rate"] = (
    cohort_analysis["churned"]
    / cohort_analysis["customers"]
    * 100
)


# ============================================================
# DISPLAY AND SAVE COHORT RESULTS
# ============================================================

print("\n======================================")
print("COHORT ANALYSIS")
print("======================================\n")

print(cohort_analysis.to_string(index=False))

cohort_analysis.to_csv("data/cohort_analysis.csv", index=False)


# ============================================================
# COHORT CHURN CHART
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    cohort_analysis["tenure_cohort"].astype(str),
    cohort_analysis["churn_rate"]
)

plt.xlabel("Customer Tenure Cohort")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Tenure Cohort")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("data/cohort_churn_rate.png", dpi=300)
plt.close()


# ============================================================
# COHORT HEATMAP: TENURE COHORT x CONTRACT TYPE
# ============================================================

heat = (
    df.groupby(["tenure_cohort", "contract_type"], observed=False)
    .agg(
        customers=("customer_id", "count"),
        churned=("churn", "sum")
    )
    .reset_index()
)

heat["churn_rate"] = np.where(
    heat["customers"] > 0,
    heat["churned"] / heat["customers"].replace(0, np.nan) * 100,
    np.nan
)

heat["reliable"] = heat["customers"] >= MIN_CELL_SIZE

heat.to_csv("data/cohort_heatmap.csv", index=False)

rate_matrix = heat.pivot(
    index="tenure_cohort",
    columns="contract_type",
    values="churn_rate"
)

count_matrix = heat.pivot(
    index="tenure_cohort",
    columns="contract_type",
    values="customers"
)

reliable_matrix = heat.pivot(
    index="tenure_cohort",
    columns="contract_type",
    values="reliable"
).astype(bool)

plot_values = rate_matrix.where(reliable_matrix).to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(9, 6))

image = ax.imshow(
    np.ma.masked_invalid(plot_values),
    cmap="YlOrRd",
    aspect="auto",
    vmin=0,
    vmax=100
)

ax.set_xticks(range(len(rate_matrix.columns)))
ax.set_xticklabels(rate_matrix.columns)
ax.set_yticks(range(len(rate_matrix.index)))
ax.set_yticklabels(rate_matrix.index.astype(str))

for row in range(plot_values.shape[0]):
    for col in range(plot_values.shape[1]):
        n = int(count_matrix.iloc[row, col])
        value = plot_values[row, col]

        if np.isnan(value):
            label = f"n={n}\n(too few)"
            colour = "grey"
        else:
            label = f"{value:.1f}%\n(n={n})"
            colour = "white" if value > 60 else "black"

        ax.text(col, row, label, ha="center", va="center",
                fontsize=9, color=colour)

ax.set_xlabel("Contract Type")
ax.set_ylabel("Tenure Cohort")
ax.set_title("Churn Rate Heatmap: Tenure Cohort x Contract Type")

colour_bar = fig.colorbar(image, ax=ax)
colour_bar.set_label("Churn Rate (%)")

fig.tight_layout()
fig.savefig("data/cohort_heatmap.png", dpi=300)
plt.close(fig)

print("\nHeatmap saved (cells with fewer than "
      f"{MIN_CELL_SIZE} customers are greyed out).")


# ============================================================
# KAPLAN-MEIER SURVIVAL ANALYSIS
# ============================================================

kmf = KaplanMeierFitter()

# duration = customer's observed tenure
# event = whether churn occurred (1 = churned, 0 = still active/censored)

duration = df["tenure_months"]

event = df["churn"]

kmf.fit(
    durations=duration,
    event_observed=event,
    label="Customer Survival"
)


# ============================================================
# SURVIVAL CURVE
# ============================================================

plt.figure(figsize=(10, 6))

kmf.plot_survival_function()

plt.title("Kaplan-Meier Customer Survival Curve")
plt.xlabel("Tenure (Months)")
plt.ylabel("Probability of Remaining Active")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("data/kaplan_meier_curve.png", dpi=300)
plt.close()


# ============================================================
# MEDIAN SURVIVAL
# ============================================================

print("\n======================================")
print("KAPLAN-MEIER ANALYSIS")
print("======================================\n")

print("Median survival time:", kmf.median_survival_time_)

print("\nSurvival probability at selected months:")

for month in [6, 12, 24, 36, 48, 60]:

    try:

        probability = kmf.survival_function_at_times(month).iloc[0]

        print(f"{month} months: {probability:.4f}")

    except Exception:

        pass


# ============================================================
# SAVE SURVIVAL TABLE
# ============================================================

survival_table = kmf.survival_function_.reset_index()

survival_table.columns = ["tenure_months", "survival_probability"]

survival_table.to_csv("data/kaplan_meier_survival.csv", index=False)


print("\nAnalysis completed.")

print("\nFiles created:")
print("data/cohort_analysis.csv")
print("data/cohort_churn_rate.png")
print("data/cohort_heatmap.csv")
print("data/cohort_heatmap.png")
print("data/kaplan_meier_curve.png")
print("data/kaplan_meier_survival.csv")
