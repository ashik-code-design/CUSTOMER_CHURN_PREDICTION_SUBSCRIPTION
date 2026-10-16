# Customer Churn Prediction for Subscription Businesses

## Project Overview

This project uses machine learning to predict customer churn for subscription-based businesses. It analyzes customer information, estimates churn probability, identifies high-risk customers, explains each prediction, and supports data-driven customer retention decisions.

**Note:** The included dataset is synthetic (1,000 customers, 29.5% churn) and intended for educational and demonstration purposes.

## Features

- **Business Dashboard:** Customer statistics, churn rates, revenue metrics, and interactive charts (light and dark mode).
- **Churn Prediction:** Estimate the churn probability of one customer, with a SHAP explanation of the top factors.
- **Batch Prediction:** Upload a CSV or Excel file and download predictions with `churn_probability`, `risk_level` and `top_3_churn_reasons`.
- **Model Analysis:** Compare Logistic Regression, Random Forest and XGBoost using cross-validation, held-out test metrics, tuned vs. 0.50 thresholds, an illustrative cost, and SHAP plots.
- **Customer Survival:** Tenure cohorts, a churn-rate heatmap (tenure cohort × contract type), and a Kaplan-Meier survival curve.
- **Risk Segmentation:** Group customers into low-, medium- and high-risk tiers.
- **Retention Priority:** Rank customers by churn probability and estimated customer value.

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
- **Cross-validation:** 5-fold stratified cross-validation on the training + validation rows; the test set is never used for model selection.
- **Feature engineering** (`src/churn_features.py`): four single-snapshot proxy features — `support_friction_proxy` (tickets per login), `usage_hours_per_login`, `inactive_14d`, and `charge_vs_avg_ratio`. They can be switched off with `USE_ENGINEERED_FEATURES = False`.
- **Threshold tuning:** the decision threshold is chosen on the validation set by minimising an illustrative cost: false positive = $50 (retention discount), false negative = $837.90 (12 × average monthly charge). These are assumptions, not measured business costs.
- **Model served by the app:** the model with the lowest validation cost. It is exported together with its own preprocessor and threshold, so they always match.
- **Explainability:** SHAP global plots, a local waterfall plot, and per-customer top churn reasons.

## Results

All numbers below come from `python src/compare_models.py` on this synthetic dataset.

### Cross-validation (5-fold, train + validation rows)

| Model | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |
|---|---:|---:|
| Logistic Regression | 0.767 ± 0.028 | 0.602 ± 0.029 |
| Random Forest | 0.751 ± 0.047 | 0.559 ± 0.036 |
| XGBoost | 0.734 ± 0.046 | 0.547 ± 0.058 |

### Held-out test set (200 customers, 59 churners) at the validation-tuned threshold

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Illustrative cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.25 | 0.550 | 0.392 | 0.949 | 0.555 | 0.791 | 0.631 | $6,864 |
| Random Forest | 0.25 | 0.450 | 0.340 | 0.915 | 0.495 | 0.735 | 0.576 | $9,440 |
| XGBoost | 0.05 | 0.470 | 0.340 | 0.847 | 0.485 | 0.728 | 0.550 | $12,391 |

### Held-out test set at a common 0.50 threshold

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.710 | 0.506 | 0.712 | 0.592 | 0.791 | 0.631 |
| Random Forest | 0.730 | 0.547 | 0.492 | 0.518 | 0.735 | 0.576 |
| XGBoost | 0.745 | 0.577 | 0.508 | 0.541 | 0.728 | 0.550 |

### Interpretation

- **Logistic Regression performed best** in cross-validation and on the test set, and it is the model the app serves. The differences between models are smaller than the fold-to-fold variation, so they should be read as "comparable", not "clearly better". XGBoost did not outperform the simpler models on this small dataset.
- The test set is small (200 rows, 59 churners), so test metrics are noisy.
- **With the assumed costs, the tuned threshold flags most customers** (about 70% of the test set; recall 95%, precision 39%). Offering the discount to every non-churner would cost about $7,050 on the test set, so the model's cost saving over "contact everyone" is small (about 3%). A more realistic cost model would include how often a retention offer actually works.
- **Predicted probabilities are not calibrated.** Class weighting inflates them (average predicted probability about 45% vs. 29.5% actual churn). Use them for ranking customers, and treat "expected revenue at risk" as a relative measure.

## Risk Categories

Risk tiers use the tuned decision threshold, so "HIGH" always matches a "Churn" prediction:

- **High Risk:** probability at or above the tuned threshold (0.25 for the current model).
- **Medium Risk:** probability at or above half of the threshold.
- **Low Risk:** probability below half of the threshold.

The same rule is used in the app, in batch predictions and in `src/risk_segmentation.py`. Retention priority then splits each of the HIGH and MEDIUM tiers using expected revenue at risk (top half of the tier = CRITICAL / MEDIUM-HIGH).

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
│   └── shap/
├── models/
│   ├── best_model.pkl                  # model served by the app
│   ├── best_preprocessor.pkl
│   ├── best_model_config.json          # name, threshold, feature settings
│   ├── cost_config.json
│   ├── cross_validation_results.csv
│   ├── model_comparison.csv            # tuned thresholds
│   ├── model_comparison_at_050.csv
│   ├── threshold_comparison_all_models.csv
│   └── logistic_regression.pkl, random_forest.pkl, xgboost.pkl (+ preprocessors)
├── notebooks/
├── src/
│   ├── churn_features.py               # shared features and risk rule
│   ├── preprocess.py
│   ├── eda.py
│   ├── compare_models.py               # main training / comparison pipeline
│   ├── risk_segmentation.py
│   ├── retention_priority.py
│   ├── shap_analysis.py
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

Re-run steps 1–3 of this list whenever `compare_models.py` or the feature settings change.

### 5. Run the Streamlit application

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
| Cost-based threshold tuning | Implemented with illustrative costs |
| SHAP global and per-customer explanations | Implemented |
| Per-customer top churn reasons in the output | Implemented (batch CSV) |
| Cohort analysis and Kaplan-Meier survival | Implemented, using tenure cohorts (no signup dates in the data) |
| Risk tiers and value-based retention strategy | Implemented |
| Feature engineering (RFM, support friction, inactivity) | Partly implemented as single-snapshot proxies |
| Velocity features (7-day vs. 30-day), DAU, feature adoption, NPS/sentiment text | Not possible: the dataset has no time series or product-usage logs |
| Predict churn within the next 30 days | Not possible: the dataset is a single snapshot |
| Involuntary churn mitigation | Not possible: no payment-failure data |
| Writing scores to Salesforce / HubSpot / Intercom | Not implemented (CSV export only) |
| Airflow, MLflow, warehouse, reverse ETL | Designed only (see below) |
| Drift monitoring and automated retraining | Designed only (see below) |
| Uplift modeling | Future scope |

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

- **Implemented in this project:** feature engineering, model comparison with cross-validation, cost-based threshold tuning, SHAP explanations, and batch scoring with top churn reasons (CSV export).
- **Designed only:** scheduled pipelines, a model registry, warehouse and CRM sync, and drift monitoring. Monitoring would track feature drift (for example, a sudden change in login frequency) and concept drift (for example, a weakening link between support tickets and churn), and trigger retraining when PR-AUC falls below an agreed floor.

## Limitations

- The dataset is synthetic and a single snapshot, so results may not reflect real customer behavior and model performance on real data may differ.
- There are no time-based features, so velocity and trend signals from the brief cannot be built; the engineered features are proxies.
- Retention costs and customer value are illustrative assumptions. Revenue-at-risk and retention-priority figures are simplified estimates, not a Customer Lifetime Value model.
- Predicted probabilities are not calibrated (class weighting inflates them); use them to rank customers.
- Survival analysis is illustrative: tenure is used as the duration and churn as the event, without validated time-to-event and censoring data.
- Test-set results come from only 200 customers and are noisy.
- Predictions support decision-making but do not guarantee that a customer will churn.

## Future Enhancements

- Evaluate models on real, appropriately authorized customer data with signup dates and time-stamped usage.
- Calibrate predicted probabilities and model the chance that a retention offer succeeds.
- Add uplift modeling (Persuadables, Sure Things, Lost Causes, Sleeping Dogs) using A/B retention-campaign data to avoid contacting customers who would stay, or leave because of the contact.
- Implement the designed pipeline: scheduling, model registry, CRM sync, drift monitoring and automated retraining.
- Try SMOTE and more tuning, and add automated tests and authenticated deployment.

## Author

Developed as a machine learning and business analytics project.