"""SQLite local feature store with feature registry and versioned snapshots.

Run after generating demo history:
    python src/feature_store.py

This is a practical local/offline feature store for a portfolio project. It
records features and provenance in SQLite; it is not an online serving system.
"""

from pathlib import Path
import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd
from temporal_feature_engineering import build_temporal_features


ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "feature_store" / "feature_store.sqlite"
MANIFEST = ROOT / "data" / "feature_store" / "manifest.json"


# These columns identify records or describe storage/provenance rather than
# customer behavior. They are documented but not classified as predictive features.
METADATA_COLUMNS = {
    "customer_id",
    "acquisition_date",
    "observation_date",
    "last_event_month",
    "base_data_provenance",
    "temporal_data_provenance",
    "feature_snapshot_created_at_utc",
    "feature_version",
    "data_provenance",
}

TARGET_COLUMNS = {"churn"}

SYNTHETIC_COLUMNS = {
    "demo_industry",
    "demo_company_size",
    "demo_geography",
    "demo_subscription_tier",
    "demo_payment_delay_days",
    "demo_payment_failure_count_90d",
    "demo_feature_adoption_rate",
    "demo_support_resolution_hours",
    "demo_support_sentiment",
    "demo_nps_score",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe_column(column):
    """Return a short, transparent description for a stored column."""
    known = {
        "customer_id": ("customer identifier", "metadata"),
        "churn": ("existing churn label; not a feature available before the outcome", "target"),
        "acquisition_date": ("customer acquisition date in the demo dataset", "metadata"),
        "observation_date": ("record observation date in the demo dataset", "metadata"),
        "last_event_month": ("last month represented in the available history", "metadata"),
        "base_data_provenance": ("provenance label for the base customer row", "provenance"),
        "temporal_data_provenance": ("provenance label for the temporal history", "provenance"),
        "feature_snapshot_created_at_utc": ("UTC time this feature snapshot was created", "metadata"),
        "feature_version": ("version label for this feature snapshot", "metadata"),
        "history_months_available": ("number of monthly history records available", "temporal"),
        "login_trend_3m_vs_prior3m": ("relative change in mean logins: latest three records versus preceding three", "temporal"),
        "usage_trend_3m_vs_prior3m": ("relative change in mean usage hours: latest three records versus preceding three", "temporal"),
        "support_ticket_trend_3m_vs_prior3m": ("relative change in mean support tickets: latest three records versus preceding three", "temporal"),
        "payment_delay_trend_3m_vs_prior3m": ("relative change in mean payment delays: latest three records versus preceding three", "temporal"),
        "recent_avg_logins_3m": ("mean logins in the latest three monthly records", "temporal"),
        "recent_avg_usage_hours_3m": ("mean usage hours in the latest three monthly records", "temporal"),
        "recent_avg_support_tickets_3m": ("mean support tickets in the latest three monthly records", "temporal"),
        "recent_avg_payment_delay_days_3m": ("mean payment delay days in the latest three monthly records", "temporal"),
    }

    if column in known:
        definition, group = known[column]
    elif column in METADATA_COLUMNS:
        definition, group = ("identifier, date, or storage metadata", "metadata")
    elif column in SYNTHETIC_COLUMNS:
        definition, group = ("synthetic demonstration field; not an observed customer fact", "synthetic_demo")
    else:
        definition = f"Stored customer field: {column}"
        group = "base_customer_data"

    provenance = (
        "SYNTHETIC_DEMO_ONLY; generated or derived for demonstration"
        if column in SYNTHETIC_COLUMNS
        or column in {
            "history_months_available",
            "login_trend_3m_vs_prior3m",
            "usage_trend_3m_vs_prior3m",
            "support_ticket_trend_3m_vs_prior3m",
            "payment_delay_trend_3m_vs_prior3m",
            "recent_avg_logins_3m",
            "recent_avg_usage_hours_3m",
            "recent_avg_support_tickets_3m",
            "recent_avg_payment_delay_days_3m",
        }
        else "synthetic source dataset; verify provenance before real-world use"
    )

    return definition, group, provenance


def main():
    data_dir = ROOT / "data"
    base_path = data_dir / "customer_churn_expanded_demo.csv"
    hist_path = data_dir / "customer_monthly_history_demo.csv"

    if not base_path.exists() or not hist_path.exists():
        raise SystemExit("Run first: python src/generate_demo_history.py")

    base = pd.read_csv(base_path)
    hist = pd.read_csv(hist_path)

    if "data_provenance" in base.columns:
        base = base.rename(
            columns={"data_provenance": "base_data_provenance"}
        )

    temporal = build_temporal_features(hist)

    if "data_provenance" in temporal.columns:
        temporal = temporal.rename(
            columns={"data_provenance": "temporal_data_provenance"}
        )

    merged = base.merge(
        temporal,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    merged["feature_snapshot_created_at_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )
    merged["feature_version"] = "v1"

    DB.parent.mkdir(parents=True, exist_ok=True)

    # Build a registry entry for every stored column.
    registry_rows = []
    for column in merged.columns:
        definition, group, provenance = describe_column(column)
        registry_rows.append(
            {
                "feature_name": column,
                "definition": definition,
                "feature_group": group,
                "provenance": provenance,
                "is_predictive_feature": (
                    column not in METADATA_COLUMNS
                    and column not in TARGET_COLUMNS
                ),
                "is_target": column in TARGET_COLUMNS,
                "feature_version": "v1",
            }
        )

    registry = pd.DataFrame(registry_rows)

    with sqlite3.connect(DB) as con:
        merged.to_sql(
            "customer_features",
            con,
            if_exists="replace",
            index=False,
        )

        registry.to_sql(
            "feature_registry",
            con,
            if_exists="replace",
            index=False,
        )

        con.execute(
            "CREATE INDEX IF NOT EXISTS "
            "idx_customer_features_customer_id "
            "ON customer_features(customer_id)"
        )

    manifest = {
        "store_type": "SQLite offline feature store",
        "database": "data/feature_store/feature_store.sqlite",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(merged)),
        "columns": list(merged.columns),
        "registry_rows": int(len(registry)),
        "source_files": {
            str(base_path.relative_to(ROOT)): sha256(base_path),
            str(hist_path.relative_to(ROOT)): sha256(hist_path),
        },
        "tables": ["customer_features", "feature_registry"],
        "point_in_time_note": (
            "The demo data is synthetic; real point-in-time correctness "
            "requires real event timestamps and as-of joins."
        ),
        "provenance": (
            "SYNTHETIC_DEMO_ONLY; replace with source-system data "
            "for real business use."
        ),
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"SQLite feature store created: {DB}")
    print(f"Customer features: {len(merged):,} rows")
    print(f"Registered columns: {len(registry):,}")
    print("Tables: customer_features, feature_registry")
    print("NOTICE: demo-only synthetic values are not real customer records.")


if __name__ == "__main__":
    main()