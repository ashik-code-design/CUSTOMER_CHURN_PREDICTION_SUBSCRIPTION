"""
Generate clearly labelled SYNTHETIC demo fields and monthly history.

This is for demonstrating cohort analysis and temporal feature engineering only.
It does not recover real missing data. Replace these files with dated source-system
exports before making real business claims.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RNG = np.random.default_rng(20261017)
OBSERVATION_DATE = pd.Timestamp("2026-09-30")

def main():
    source = pd.read_csv(DATA / "customer_churn.csv")
    n = len(source)
    # Deterministic acquisition dates consistent with available tenure.
    tenure = source["tenure_months"].fillna(1).clip(1, 72).astype(int)
    acquisition = OBSERVATION_DATE - pd.to_timedelta(tenure * 30, unit="D")
    # Supplementary fields are simulated, not observed. Explicit prefix avoids
    # accidentally presenting these as genuine CRM facts.
    expanded = source.copy()
    expanded["acquisition_date"] = acquisition.dt.strftime("%Y-%m-%d")
    expanded["observation_date"] = OBSERVATION_DATE.strftime("%Y-%m-%d")
    expanded["billing_frequency"] = source["contract_type"].map({
        "Monthly": "Monthly", "One year": "Annual", "Two year": "Annual"
    }).fillna("Monthly")
    expanded["demo_industry"] = RNG.choice(
        ["Retail", "Technology", "Healthcare", "Education", "Other"], n
    )
    expanded["demo_company_size"] = RNG.choice(
        ["1-10", "11-50", "51-200", "201-1000", "1000+"], n
    )
    expanded["demo_geography"] = RNG.choice(
        ["South", "North", "West", "East", "Central"], n
    )
    expanded["demo_subscription_tier"] = RNG.choice(
        ["Basic", "Standard", "Premium"], n, p=[0.35, 0.45, 0.20]
    )
    expanded["demo_payment_delay_days"] = RNG.poisson(
        np.where(source["churn"].to_numpy() == 1, 4.0, 1.3)
    ).clip(0, 30)
    expanded["demo_payment_failure_count_90d"] = RNG.poisson(
        np.where(source["churn"].to_numpy() == 1, 1.1, 0.25)
    ).clip(0, 8)
    expanded["demo_feature_adoption_rate"] = np.clip(
        RNG.normal(np.where(source["churn"].to_numpy() == 1, 0.38, 0.66), 0.16), 0, 1
    ).round(3)
    expanded["demo_support_resolution_hours"] = np.clip(
        RNG.gamma(shape=2.0, scale=np.where(source["churn"].to_numpy() == 1, 8.0, 4.0)), 0.2, 120
    ).round(2)
    expanded["demo_support_sentiment"] = np.clip(
        RNG.normal(np.where(source["churn"].to_numpy() == 1, -0.25, 0.35), 0.35), -1, 1
    ).round(3)
    expanded["demo_nps_score"] = np.clip(
        RNG.normal(np.where(source["churn"].to_numpy() == 1, 3.5, 7.5), 2.0), 0, 10
    ).round(1)
    expanded["data_provenance"] = "SYNTHETIC_DEMO_ONLY"
    expanded.to_csv(DATA / "customer_churn_expanded_demo.csv", index=False)

    # Monthly dated usage/support event records. Build history for each customer
    # from acquisition through observation month. Trend signals are synthetic.
    records = []
    for i, row in source.iterrows():
        months = int(tenure.iloc[i])
        months = max(1, min(months, 72))
        base_login = max(1.0, float(row["logins_30d"]))
        base_usage = max(0.5, float(row["usage_hours_30d"]))
        churn = int(row["churn"])
        # Churned customers are given a stronger decline in the final months
        # solely to make a demo dataset with known synthetic patterns.
        for m in range(months):
            progress = m / max(1, months - 1)
            decline = (0.48 if churn else 0.12) * progress
            logins = max(0, int(round(base_login * (1 - decline) * RNG.uniform(0.78, 1.22))))
            usage = max(0.0, base_usage * (1 - decline) * RNG.uniform(0.80, 1.20))
            tickets = max(0, int(RNG.poisson(max(0.05, float(row["support_tickets_30d"]) / 2))))
            payment_delay = max(0, int(RNG.poisson(2.6 if churn else 0.8)))
            month_date = (OBSERVATION_DATE.to_period("M") - (months - 1 - m)).to_timestamp()
            records.append({
                "customer_id": str(row["customer_id"]),
                "event_month": month_date.strftime("%Y-%m-%d"),
                "logins_30d": logins,
                "usage_hours_30d": round(usage, 2),
                "support_tickets_30d": tickets,
                "payment_delay_days": payment_delay,
                "data_provenance": "SYNTHETIC_DEMO_ONLY",
            })
    history = pd.DataFrame(records)
    history.to_csv(DATA / "customer_monthly_history_demo.csv", index=False)

    # Event-date cohort input. Churn date is synthetic and explicitly labelled.
    cohort = expanded[["customer_id", "acquisition_date", "observation_date", "churn"]].copy()
    churn_dates = []
    for _, row in cohort.iterrows():
        if int(row["churn"]) == 1:
            acq = pd.Timestamp(row["acquisition_date"])
            obs = pd.Timestamp(row["observation_date"])
            elapsed = max(1, (obs.year - acq.year) * 12 + obs.month - acq.month)
            # Random synthetic cancellation month before observation cutoff.
            offset = int(RNG.integers(1, elapsed + 1))
            churn_dates.append((acq + pd.DateOffset(months=offset)).strftime("%Y-%m-%d"))
        else:
            churn_dates.append("")
    cohort["churn_date"] = churn_dates
    cohort["data_provenance"] = "SYNTHETIC_DEMO_ONLY"
    cohort.to_csv(DATA / "customer_history_demo.csv", index=False)

    print(f"Created synthetic expanded dataset: {len(expanded)} rows")
    print(f"Created synthetic monthly history: {len(history)} rows")
    print("Created synthetic acquisition/churn dates for cohort demonstration.")
    print("IMPORTANT: all generated supplementary data is synthetic demo data.")

if __name__ == "__main__":
    main()
