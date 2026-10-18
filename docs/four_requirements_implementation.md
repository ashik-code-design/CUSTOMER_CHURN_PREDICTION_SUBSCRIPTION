
# Completion of the Four Data Requirements

## What Is Now Implemented

1. **Acquisition-month cohort heatmap**
   `src/acquisition_cohort_analysis.py` creates acquisition-month by month-since-acquisition retention tables and a PNG heatmap.

2. **Temporal feature engineering**
   `src/temporal_feature_engineering.py` computes three-month-versus-prior-three-month trends for logins, usage, support tickets, and payment delays from dated monthly records.

3. **Expanded dataset schema**
   `src/generate_demo_history.py` creates an expanded demo dataset with billing frequency, industry, company size, geography, subscription tier, payment delays and failures, feature adoption rate, support resolution time, sentiment, and NPS.

4. **Local feature store**
   `src/feature_store.py` creates a SQLite offline feature store with `customer_features` and `feature_registry` tables, plus a source-hash manifest.

## Important Data Integrity Warning

The supplied 1,000-row, 18-column source dataset does not contain the historical dates, usage events, payment history, industry, geography, NPS, or other newly added fields.

To make the full pipeline runnable and demonstrable, `generate_demo_history.py` creates **synthetic demo-only values** and labels their provenance. These values are not real-world evidence and must not be described as genuine customer history.

Replace the generated demo files with appropriate source-system data before making real business claims or using the resulting features in a production model.

## Run the Full Demo Pipeline

Run these commands in PowerShell from the project root, with your Python virtual environment activated:

```powershell
python src/generate_demo_history.py
python src/temporal_feature_engineering.py
python src/acquisition_cohort_analysis.py --input data/customer_history_demo.csv
python src/feature_store.py
```

The commands generate or update the demo data, temporal features, cohort analysis outputs, and SQLite feature store.

## Generated Outputs

### Demo datasets

- `data/customer_churn_expanded_demo.csv`
- `data/customer_monthly_history_demo.csv`
- `data/customer_history_demo.csv`
- `data/temporal_features_demo.csv`

### Acquisition-cohort analysis

- `data/acquisition_cohorts/acquisition_cohort_retention_heatmap.csv`
- `data/acquisition_cohorts/acquisition_cohort_retention_heatmap.png`
- `data/acquisition_cohorts/acquisition_cohort_retention_long.csv`
- `data/acquisition_cohorts/acquisition_cohort_retention_summary.csv`

### Feature store

- `data/feature_store/feature_store.sqlite` - SQLite database containing the `customer_features` and `feature_registry` tables.
- `data/feature_store/manifest.json` - metadata describing the feature snapshot, source-file hashes, columns, and data provenance.

The SQLite database is ignored by Git through `.gitignore`; the manifest can be committed to document the generated feature-store snapshot.

## Use Real Data

For genuine acquisition-cohort analysis, the input must contain:

- `customer_id`
- `acquisition_date`
- `observation_date`
- `churn_date` - blank when the customer had not churned as of the observation date.

For temporal trends, provide dated, customer-level monthly records with the metrics required by `src/temporal_feature_engineering.py`.

Populate expanded fields using appropriate source systems, such as billing, CRM, product analytics, and customer-support systems. Do not replace missing real values with fabricated values and then treat them as genuine observations.

## Scope and Limitations

This implementation provides runnable, documented functionality for a student or portfolio project.

The SQLite feature store is an offline local feature store, not a production online feature-serving platform. The cohort and temporal outputs currently use synthetic demo data until genuine historical source data is connected.

A generated feature timestamp does not establish point-in-time correctness. Reliable historical analysis requires suitable event timestamps or dated snapshots and appropriate point-in-time joins.
