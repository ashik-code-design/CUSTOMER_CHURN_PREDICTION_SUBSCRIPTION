# Customer Churn Prediction for Subscription Businesses

## Project Overview

This project uses machine learning to predict customer churn for subscription-based businesses. It analyzes customer information, estimates churn probability, identifies high-risk customers, explains each prediction, and supports data-driven customer retention decisions.

**Note:** The included dataset is synthetic (1,000 customers, 29.5% churn) and intended for educational and demonstration purposes.

## Features

- **Business Dashboard:** Customer statistics, churn rates, revenue metrics, and interactive charts (light and dark mode).
- **Churn Prediction:** Estimate the churn probability of one customer, with a SHAP explanation of the top factors.
- **Batch Prediction:** Upload a CSV or Excel file and download predictions with `churn_probability`, `risk_level` and `top_3_churn_reasons`.
- **Model Analysis:** Compare Logistic Regression, Random Forest and XGBoost using cross-validation, held-out test metrics, tuned vs. 0.50 thresholds, a calibration check, an illustrative cost, SHAP plots, and a note on why accuracy is misleading for churn.
- **Customer Survival:** Tenure cohorts, a churn-rate heatmap (tenure cohort × contract type), a Kaplan-Meier survival curve, and a reading of where survival falls fastest.
- **Risk Segmentation:** Group customers into low-, medium- and high-risk tiers, with a risk-decile view (decile 1 = the highest-risk 10%).
- **Retention Priority:** Rank customers by calibrated churn probability and estimated customer value, with a white-glove tier (top 10% risk and above-median value) and an automated tier (medium risk, below-median value).
- **Batch scoring job:** `python src/batch_scoring.py` scores a customer file (CSV or Excel) from the command line and writes `churn_probability`, `risk_tier` and `top_3_churn_reasons`; it can be scheduled (see Installation and Setup).

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Plotly and Matplotlib
- SHAP
- Lifelines
- Joblib

## Methodology

- **Split:** one stratified 60% train / 20% validation / 20% test split shared by all models.
- **Class imbalance:** `class_weight="balanced"` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost). SMOTE is not used.
- **Cross-validation:** 5-fold stratified cross-validation on the training + validation rows. The test set is never used for model selection or tuning.
- **Feature engineering** (`src/churn_features.py`): four single-snapshot proxy features: `support_friction_proxy` (tickets per login), `usage_hours_per_login`, `inactive_14d`, and `charge_vs_avg_ratio`. They can be switched off with `USE_ENGINEERED_FEATURES = False`.
- **Probability calibration:** class weighting inflates raw churn probabilities, so each model's scores are mapped to realistic probabilities with a sigmoid (Platt) calibrator fitted on the validation set.
- **Cost model and threshold tuning** (illustrative assumptions, editable in `src/compare_models.py`): every customer contacted costs a $50 offer; a churner who leaves costs the customer value ($837.90 = 12 × average monthly charge); an offer is assumed to save 30% of contacted churners. The break-even probability for contacting a customer is therefore about 0.20. The decision threshold is chosen on the validation set by minimising this cost.
- **Model served by the app:** the model with the highest cross-validated PR-AUC (PR-AUC is prioritised in the project brief because it is sensitive to the minority class). It is exported together with its calibrator, preprocessor and threshold, so they always match.
- **Explainability:** SHAP global plots, a local waterfall plot, and per-customer top churn reasons.

## Results

All numbers below come from `python src/compare_models.py` on this synthetic dataset.

### Cross-validation (5-fold, train + validation rows)

| Model | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |
|---|---:|---:|
| Logistic Regression | 0.767 ± 0.028 | 0.602 ± 0.029 |
| Random Forest | 0.751 ± 0.047 | 0.559 ± 0.036 |
| XGBoost | 0.734 ± 0.046 | 0.547 ± 0.058 |

### Probability calibration (test set)

| Model | Actual churn rate | Mean predicted (raw) | Mean predicted (calibrated) | Brier score, raw | Brier score, calibrated |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 29.5% | 45.2% | 28.6% | 0.189 | 0.162 |
| Random Forest | 29.5% | 40.1% | 28.5% | 0.189 | 0.176 |
| XGBoost | 29.5% | 30.3% | 29.3% | 0.190 | 0.177 |

### Held-out test set (200 customers, 59 churners) at the validation-tuned threshold

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Illustrative cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.25 | 0.695 | 0.489 | 0.746 | 0.591 | 0.791 | 0.631 | $42,876 |
| Random Forest | 0.25 | 0.660 | 0.454 | 0.746 | 0.564 | 0.735 | 0.576 | $43,226 |
| XGBoost | 0.15 | 0.495 | 0.348 | 0.814 | 0.487 | 0.728 | 0.550 | $44,270 |

### Held-out test set at a common 0.50 threshold

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.745 | 0.600 | 0.407 | 0.485 | 0.791 | 0.631 |
| Random Forest | 0.720 | 0.541 | 0.339 | 0.417 | 0.735 | 0.576 |
| XGBoost | 0.755 | 0.632 | 0.407 | 0.495 | 0.728 | 0.550 |

### Reference strategies on the test set (same assumed cost model)

| Strategy | Illustrative cost |
|---|---:|
| Contact nobody | $49,436 |
| Contact everyone | $44,605 |
| Logistic Regression at its tuned threshold | $42,876 |

### Interpretation

- **Logistic Regression performed best** in cross-validation and on the test set, so it is the model the app serves. The gaps between models are of the same size as the fold-to-fold variation, so read them as "comparable", not "clearly better". XGBoost did not outperform the simpler models on this small dataset.
- **Calibration worked:** after calibration the average predicted churn is close to the actual rate (about 29%), and the Brier score improved for every model. Calibrated probabilities can be read as probabilities, not just rankings. Calibration was fitted on only 200 validation customers, so it is approximate.
- **Business value is modest and depends on assumptions.** Under the assumed costs, the model's cost is about 13% below contacting nobody and about 4% (roughly $1,700 on 200 customers) below contacting everyone. The result is sensitive to the assumed 30% save rate, the offer cost and the customer value, and the test set is small, so this should be read as a demonstration of the method, not a forecast.
- The test set has only 200 customers (59 churners), so test metrics are noisy.

### Data findings

- Churn is highest in the first year (38.8% for 0–6 months and 41.3% for 7–12 months) and lowest after five years (21.5%).
- Kaplan-Meier estimates: about 92.5% of customers remain active after 12 months and about 58% after 60 months; the median survival time is 66 months. Survival falls fastest between months 48 and 60 (from about 69% to 58%, roughly 0.9 percentage points per month), so retention outreach would be scheduled shortly before that window. This analysis is illustrative (see Limitations).

## Risk Categories

Risk tiers use the tuned decision threshold, so "HIGH" always matches a "Churn" prediction:

- **High Risk:** calibrated probability at or above the tuned threshold (0.25 for the current model).
- **Medium Risk:** probability at or above half of the threshold.
- **Low Risk:** probability below half of the threshold.

The same rule is used in the app, in batch predictions and in `src/risk_segmentation.py`. Retention priority then splits the HIGH and MEDIUM tiers using expected revenue at risk (the top half of each tier becomes CRITICAL or MEDIUM-HIGH).

Recommended actions follow the project brief: CRITICAL / white-glove customers get an immediate call from a senior Customer Success Manager tailored with the SHAP churn reasons, and lower-value medium-risk customers get an automated email drip campaign with an optional 15% discount.

## Project Structure

```text
customer-churn-prediction/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── customer_churn.csv
│   ├── cohort_analysis.csv
│   ├── cohort_churn_rate.png
│   ├── cohort_heatmap.csv
│   ├── cohort_heatmap.png
│   ├── kaplan_meier_survival.csv
│   ├── kaplan_meier_curve.png
│   ├── risk_segmented_customers.csv
│   ├── retention_priority.csv
│   ├── batch_scores.csv                # written by batch_scoring.py
│   ├── batch_scoring_log.csv           # one line per scoring run
│   └── shap/
├── models/
│   ├── best_model.pkl                  # calibrated model served by the app
│   ├── best_preprocessor.pkl
│   ├── best_model_config.json          # name, threshold, feature settings
│   ├── cost_config.json
│   ├── cross_validation_results.csv
│   ├── calibration_check.csv
│   ├── model_comparison.csv            # tuned thresholds
│   ├── model_comparison_at_050.csv
│   ├── threshold_comparison_all_models.csv
│   └── logistic_regression.pkl, random_forest.pkl, xgboost.pkl (+ preprocessors)
├── notebooks/
├── src/
│   ├── churn_features.py               # shared features, risk rule, calibration wrapper
│   ├── preprocess.py
│   ├── eda.py
│   ├── compare_models.py               # main training / comparison pipeline
│   ├── risk_segmentation.py
│   ├── retention_priority.py
│   ├── shap_analysis.py
│   ├── batch_scoring.py                # schedulable batch scoring job
│   ├── survival_analysis.py
│   └── train_model.py, train_random_forest.py, train_xgboost.py   # single-model scripts (reference)
└── venv/
```

## Installation and Setup

### 1. Open the project directory

```powershell
cd C:\customer-churn-prediction
```

### 2. Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Build the models and analysis files (run in this order)

```powershell
python src/compare_models.py
python src/risk_segmentation.py
python src/retention_priority.py
python src/shap_analysis.py
python src/survival_analysis.py
```

Re-run the first four commands whenever `compare_models.py`, the cost settings or the feature settings change. Keep `src/churn_features.py` in place: the saved model needs it to load.

### 5. Optional: score a new customer file and schedule it

```powershell
python src/batch_scoring.py --input data/new_customers.csv --output data/new_scores.csv
```

The file needs the same customer columns as `data/customer_churn.csv` (without `churn`). The output has `customer_id`, `churn_probability`, `risk_tier`, `top_3_churn_reasons`, `model` and `scored_at`, and each run adds one line to `data/batch_scoring_log.csv`.

To run it every night on Windows (example, replace the paths with yours):

```powershell
schtasks /Create /SC DAILY /ST 02:00 /TN "ChurnBatchScoring" /TR "C:\customer-churn-prediction\venv\Scripts\python.exe C:\customer-churn-prediction\src\batch_scoring.py"
```

### 6. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## Coverage of the Project Brief

| Brief requirement | Status |
|---|---|
| Compare Logistic Regression, Random Forest, XGBoost | Implemented |
| Class imbalance handling, cross-validation | Implemented (class weights, 5-fold CV) |
| Recall, Precision, ROC-AUC, PR-AUC | Implemented |
| Why accuracy is misleading | Implemented (explained in the app) |
| Expected-value cost framework and threshold tuning | Implemented with illustrative costs and an assumed save rate |
| Calibrated probabilities | Implemented (Platt scaling) |
| SHAP global and per-customer explanations | Implemented |
| Per-customer top churn reasons in the output | Implemented (app batch page and `src/batch_scoring.py`) |
| Batch inference for all customers, schedulable | Implemented as a command-line job (Airflow orchestration is design-only) |
| Cohort analysis and Kaplan-Meier survival | Implemented, using tenure cohorts (no signup dates in the data), with a reading of the steepest survival decline |
| Risk tiers, risk deciles and value-based retention strategy (white-glove and automated tiers) | Implemented |
| Feature engineering (RFM, support friction, inactivity) | Partly implemented as single-snapshot proxies |
| Velocity features (7-day vs. 30-day), DAU, feature adoption, NPS/sentiment text | Not possible: the dataset has no time series or product-usage logs |
| Predict churn within the next 30 days | Not possible: the dataset is a single snapshot |
| Involuntary churn mitigation | Not possible: no payment-failure data |
| Writing scores to Salesforce / HubSpot / Intercom | Not implemented (CSV export only) |
| Airflow, MLflow, warehouse, reverse ETL | Designed only (see below) |
| Drift monitoring and automated retraining | Designed only (see below) |
| Uplift modeling | Future scope |

## Mapping to the Brief's Data Domains

| Brief domain and field | In this dataset |
|---|---|
| **Demographic and account:** industry, company size, geography | Not available |
| Subscription tier | Proxy: `contract_type`, `internet_service` |
| Tenure | `tenure_months` |
| *(consumer fields not in the brief)* | `age`, `gender`, `senior_citizen` |
| **Financial and billing:** MRR | `monthly_charges` (also `total_charges`) |
| Billing frequency | Not available (partly reflected by `contract_type`) |
| Discount usage | `discount` |
| Payment delays | Not available (only `payment_method` and `auto_payment`) |
| **Product telemetry:** daily active users | Proxy: `logins_30d` |
| Session length | Proxy: `usage_hours_30d`, engineered `usage_hours_per_login` |
| Recency | `last_login_days` (plus engineered `inactive_14d`) |
| Feature adoption rate, time-to-value | Not available |
| **Support and sentiment:** open tickets | `support_tickets_30d` |
| Average resolution time | Not available (so the support friction index is a proxy: tickets per login) |
| NPS | Proxy: `satisfaction_score` (1-10) |
| Sentiment of support chat logs | Not available |

About half of the brief's fields exist in this dataset, which is why this requirement is only partly met. Adding the missing fields requires real CRM, billing and product-analytics data.

## Deployment and MLOps Design (designed, not implemented)

```text
CRM / Billing / Product events
        |  nightly ETL (Apache Airflow)
        v
Feature store -> batch scoring job <- model registry (MLflow)
        |
        v
Data warehouse: user_id, churn_probability, risk_tier, top_3_churn_reasons
        |  reverse ETL (Hightouch / Census)
        v
Salesforce - Zendesk - Marketo - email / SMS campaigns
```

- **Implemented in this project:** feature engineering, model comparison with cross-validation, calibration, cost-based threshold tuning, SHAP explanations, and a batch scoring job with top churn reasons (`src/batch_scoring.py`, schedulable with Windows Task Scheduler or cron).
- **Designed only:** scheduled pipelines, a model registry, warehouse and CRM sync, and drift monitoring. Monitoring would track feature drift (for example, a sudden change in login frequency) and concept drift (for example, a weakening link between support tickets and churn), and trigger retraining when PR-AUC falls below an agreed floor.

## Limitations

- The dataset is synthetic and a single snapshot, so results may not reflect real customer behavior and model performance on real data may differ.
- There are no time-based features, so velocity and trend signals from the brief cannot be built; the engineered features are proxies.
- Retention costs, customer value and the 30% save rate are illustrative assumptions. Revenue-at-risk and retention-priority figures are simplified estimates, not a Customer Lifetime Value model.
- Calibration is fitted on only 200 validation customers, and test results come from only 200 customers, so both are approximate and noisy.
- Risk segmentation and retention priority score all 1,000 customers, including customers the model was trained on, so those probabilities are somewhat optimistic (more so for tree-based models).
- Survival analysis is illustrative: tenure is used as the duration and churn as the event, without validated time-to-event and censoring data.
- Predictions support decision-making but do not guarantee that a customer will churn.

## Future Enhancements

- Evaluate models on real, appropriately authorized customer data with signup dates and time-stamped usage.
- Measure the real success rate of retention offers (A/B tests) instead of assuming one.
- Add uplift modeling (Persuadables, Sure Things, Lost Causes, Sleeping Dogs) using A/B retention-campaign data to avoid contacting customers who would stay, or leave because of the contact.
- Implement the designed pipeline: scheduling, model registry, CRM sync, drift monitoring and automated retraining.
- Try SMOTE and more tuning, and add automated tests and authenticated deployment.

## Author

Developed as a machine learning and business analytics project.


## Additional Data Requirements Demo (Synthetic Data Only)

This project includes four additional implementations for exploring customer retention, temporal trends, expanded subscription attributes, and feature storage.

**Important:** The original dataset (`data/customer_churn.csv`) contains a single snapshot of 1,000 synthetic customer records. It does not contain genuine historical acquisition dates, churn events, or monthly activity records. The additional historical data generated by these scripts is **synthetic demo data**, not real customer history.

### 1. Acquisition-Month Cohort Analysis

Script: `src/acquisition_cohort_analysis.py`

Generates an acquisition-month retention heatmap using customer acquisition dates, observation dates, and churn dates.

Run:

```powershell
python src/acquisition_cohort_analysis.py --input data/customer_history_demo.csv
```

Outputs are saved in `data/acquisition_cohorts/`, including the heatmap image, retention tables, and cohort summary.

### 2. Temporal Feature Engineering

Script: `src/temporal_feature_engineering.py`

Calculates changes between recent and previous three-month periods for metrics such as logins, usage, support tickets, and payment delays.

Run:

```powershell
python src/temporal_feature_engineering.py
```

Generated temporal features are saved to `data/temporal_features_demo.csv`.

### 3. Expanded Subscription Demo Dataset

Script: `src/generate_demo_history.py`

Generates an expanded demonstration dataset with additional subscription and business attributes, such as billing frequency, industry, company size, geography, payment activity, feature adoption, support resolution time, customer sentiment, and NPS.

Run:

```powershell
python src/generate_demo_history.py
```

Generated files include:

- `data/customer_churn_expanded_demo.csv`
- `data/customer_monthly_history_demo.csv`
- `data/customer_history_demo.csv`

All newly generated values are synthetic and must not be presented as genuine customer observations.

### 4. Local SQLite Feature Store

Script: `src/feature_store.py`

Creates a local offline SQLite feature store containing customer features and a feature registry.

Run:

```powershell
python src/feature_store.py
```

The generated database is stored at `data/feature_store/feature_store.sqlite`, with accompanying metadata in `data/feature_store/manifest.json`.

The SQLite database is excluded from Git through `.gitignore`. This is an educational offline feature store, not a production online feature-serving system.

### Run the Complete Demo Pipeline

Run these commands from the project root in the activated Python virtual environment:

```powershell
python src/generate_demo_history.py
python src/temporal_feature_engineering.py
python src/acquisition_cohort_analysis.py --input data/customer_history_demo.csv
python src/feature_store.py
```

### Data Quality and Limitations

- Use genuine source-system records for real business analysis.
- Historical retention analysis requires suitable acquisition, observation, and churn dates.
- Temporal features require dated customer activity records.
- Generated demo values must not be treated as evidence of actual customer behaviour.
- A generated feature timestamp alone does not guarantee point-in-time correctness.
