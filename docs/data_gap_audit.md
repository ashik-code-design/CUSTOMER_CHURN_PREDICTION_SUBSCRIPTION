
# Data and Feature Store Gap Audit

## Verified current-state findings

- `data/customer_churn.csv` contains 1,000 rows and 18 columns.
- The available data is a single snapshot and is documented in README as synthetic.
- The original cohort artifacts use `tenure_cohort` and `contract_type`, not acquisition month.
- `src/churn_features.py` creates four proxy features: `support_friction_proxy`, `usage_hours_per_login`, `inactive_14d`, and `charge_vs_avg_ratio`.
- No reusable feature-store layer was present in the inspected project files before this update.

## Changes added

### Local feature-store snapshot

Run from the project root:

```powershell
python src/feature_store.py
```

This creates:

- `data/feature_store/feature_store.sqlite`, containing the `customer_features` and `feature_registry` tables.
- `data/feature_store/manifest.json`, containing feature columns, source-file hashes, generation time, and known data limitations.

The SQLite database is a local offline feature store, not a production online serving system. The generated historical data and temporal features are synthetic demo data and must not be represented as real customer history.

### Acquisition-month cohort analysis

The script is available at `src/acquisition_cohort_analysis.py`.

It requires a longitudinal input file with these columns:

- `customer_id`
- `acquisition_date`
- `observation_date`
- `churn_date` - blank when the customer had not churned by the observation date.

For the generated synthetic demo data, run:

```powershell
python src/acquisition_cohort_analysis.py --input data/customer_history_demo.csv
```

The script generates acquisition-cohort retention analysis and saves the resulting artifacts under `data/acquisition_cohorts/`.

The original `data/customer_churn.csv` does not contain the required acquisition and churn dates. Therefore, the generated historical data is explicitly synthetic demo data, not genuine historical customer activity. Use actual source-system dates for real retention analysis.

## Missing fields to consider adding from source systems

The project brief describes fields not currently represented in the original 18-column dataset, including:

- Industry, company size, geography, and subscription tier
- Billing frequency
- Payment delays and payment failures
- Feature adoption
- Historical usage and login activity
- Support resolution time
- Customer sentiment and NPS
- Acquisition and churn event dates

Do not fill these columns with invented values and describe them as real. Synthetic values may be added for demonstrations only when they are clearly labelled and appropriate to the intended experiment.

## Important modelling note

A timestamp indicating when a feature-store file was generated is not the same as a real customer observation timestamp.

Point-in-time correctness and reliable trend features require genuine event history or reliable dated snapshots. The current generated historical data is intended for demonstration and testing; it does not establish real-world customer behaviour or production-grade point-in-time correctness.
