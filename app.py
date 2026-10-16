import os
import json

import joblib

import numpy as np

import pandas as pd

import streamlit as st

import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS  (theme-safe: works in both light and dark mode)
# ============================================================
# Fix: the old CSS hardcoded colors (white background, dark text,
# light-grey boxes). In dark mode that gave dark text on a dark
# page and bright white boxes. Now every color either inherits
# from the active Streamlit theme or uses semi-transparent
# values that look right on both light and dark backgrounds.

st.markdown("""
<style>

    /* Header */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: inherit;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 20px;
        color: inherit;
        opacity: 0.70;
        margin-bottom: 30px;
    }

    /* Section titles */
    .section-title {
        font-size: 30px;
        font-weight: 750;
        color: inherit;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* KPI cards */
    .kpi-card {
        background: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.30);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.10);
        min-height: 145px;
        color: inherit;
    }

    .kpi-title {
        font-size: 16px;
        color: inherit;
        opacity: 0.75;
        margin-bottom: 10px;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: inherit;
    }

    /* Info boxes */
    .info-box {
        background: rgba(128, 128, 128, 0.12);
        border-left: 5px solid #6366f1;
        color: inherit;
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: inherit;
        opacity: 0.65;
        padding: 30px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_FILE = os.path.join(DATA_DIR, "customer_churn.csv")

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "logistic_regression.pkl"
)

PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR,
    "model_preprocessor.pkl"
)

THRESHOLD_CONFIG_FILE = os.path.join(
    MODEL_DIR,
    "threshold_config.json"
)


# ============================================================
# LOAD MAIN DATASET
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_FILE):
        return None

    return pd.read_csv(DATA_FILE)


df = load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_preprocessor():

    if not os.path.exists(PREPROCESSOR_FILE):
        return None

    return joblib.load(PREPROCESSOR_FILE)


@st.cache_data
def load_prediction_threshold():
    """Load the tuned threshold, falling back to 0.50 if unavailable."""
    if not os.path.exists(THRESHOLD_CONFIG_FILE):
        return 0.50

    try:
        with open(THRESHOLD_CONFIG_FILE, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        threshold_value = float(config.get("threshold", 0.50))
        if not 0.0 <= threshold_value <= 1.0:
            return 0.50
        return threshold_value
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.50


model = load_model()
preprocessor = load_preprocessor()
prediction_threshold = load_prediction_threshold()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📊 Customer Churn")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "📊 Business Dashboard",
            "🔮 Churn Prediction",
            "📂 Batch Prediction",
            "📈 Model Analysis",
            "📈 Customer Survival",
            "🎯 Risk Segmentation",
            "💰 Retention Priority",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.caption("Customer Churn Prediction")
    st.caption("Data Science Project")
    st.caption("Built with Python & Streamlit")


# ============================================================
# CHECK DATA
# ============================================================

if df is None:

    st.error(
        "customer_churn.csv was not found inside the data folder."
    )

    st.stop()


# ============================================================
# BUSINESS DASHBOARD
# ============================================================

if page == "📊 Business Dashboard":

    st.markdown(
        '<div class="main-title">📊 Customer Churn Business Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">'
        'Executive overview of customer churn, customer behavior, '
        'revenue exposure, and retention opportunities.'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # BUSINESS OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📌 Business Overview</div>',
        unsafe_allow_html=True
    )

    total_customers = len(df)

    churned_customers = int(df["churn"].sum())

    active_customers = total_customers - churned_customers

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers > 0
        else 0
    )

    avg_satisfaction = df["satisfaction_score"].mean()

    avg_tenure = df["tenure_months"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Customers", f"{total_customers:,}")

    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}",
            f"{churn_rate:.2f}% churn"
        )

    with col3:
        st.metric(
            "Active Customers",
            f"{active_customers:,}",
            f"{100 - churn_rate:.2f}% active"
        )

    with col4:
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}/10")

    with col5:
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

    # --------------------------------------------------------
    # REVENUE OVERVIEW
    # --------------------------------------------------------

    st.markdown("---")

    avg_monthly_charges = df["monthly_charges"].mean()

    total_monthly_revenue = df["monthly_charges"].sum()

    churned_revenue = df.loc[
        df["churn"] == 1,
        "monthly_charges"
    ].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Monthly Charges",
            f"${avg_monthly_charges:,.2f}"
        )

    with col2:
        st.metric(
            "Total Monthly Revenue",
            f"${total_monthly_revenue:,.2f}"
        )

    with col3:
        st.metric(
            "Revenue from Churned Customers",
            f"${churned_revenue:,.2f}"
        )

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">'
            'Customer Churn Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        churn_chart_data = pd.DataFrame({
            "Status": ["Active", "Churned"],
            "Customers": [active_customers, churned_customers]
        })

        fig = px.pie(
            churn_chart_data,
            names="Status",
            values="Customers",
            hole=0.45,
            title="Active vs Churned Customers"
        )

        fig.update_layout(height=430)

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # CONTRACT ANALYSIS
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="section-title">'
            'Churn by Contract Type'
            '</div>',
            unsafe_allow_html=True
        )

        contract_data = (
            df.groupby("contract_type")
            .agg(
                customers=("churn", "count"),
                churned=("churn", "sum")
            )
            .reset_index()
        )

        contract_data["churn_rate"] = (
            contract_data["churned"]
            / contract_data["customers"]
            * 100
        )

        fig = px.bar(
            contract_data,
            x="contract_type",
            y="churn_rate",
            text="churn_rate",
            title="Churn Rate by Contract"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Churn Rate (%)",
            xaxis_title="Contract Type",
            height=430
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # INTERNET SERVICE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Internet Service Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    internet_data = (
        df.groupby("internet_service")
        .agg(
            customers=("churn", "count"),
            churned=("churn", "sum")
        )
        .reset_index()
    )

    internet_data["churn_rate"] = (
        internet_data["churned"]
        / internet_data["customers"]
        * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            internet_data,
            x="internet_service",
            y="customers",
            text="customers",
            title="Customers by Internet Service"
        )

        fig.update_layout(
            xaxis_title="Internet Service",
            yaxis_title="Customers",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            internet_data,
            x="internet_service",
            y="churn_rate",
            text="churn_rate",
            title="Churn Rate by Internet Service"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Internet Service",
            yaxis_title="Churn Rate (%)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # MONTHLY CHARGES VS CHURN
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        'Customer Charges vs Churn'
        '</div>',
        unsafe_allow_html=True
    )

    fig = px.box(
        df,
        x="churn",
        y="monthly_charges",
        points="outliers",
        labels={
            "churn": "Churn Status",
            "monthly_charges": "Monthly Charges"
        },
        title="Monthly Charges Distribution by Churn"
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=["Active", "Churned"]
        ),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # SATISFACTION VS CHURN
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            x="churn",
            y="satisfaction_score",
            points="outliers",
            title="Satisfaction Score vs Churn"
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Active", "Churned"]
            ),
            yaxis_title="Satisfaction Score",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # TENURE VS CHURN
    # --------------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="churn",
            y="tenure_months",
            points="outliers",
            title="Tenure vs Churn"
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Active", "Churned"]
            ),
            yaxis_title="Tenure (Months)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # SUPPORT TICKETS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(
            df,
            x="churn",
            y="support_tickets_30d",
            points="outliers",
            title="Support Tickets vs Churn"
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Active", "Churned"]
            ),
            yaxis_title="Support Tickets - Last 30 Days",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # LOGIN ACTIVITY
    # --------------------------------------------------------

    with col2:

        fig = px.box(
            df,
            x="churn",
            y="logins_30d",
            points="outliers",
            title="Login Activity vs Churn"
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=["Active", "Churned"]
            ),
            yaxis_title="Logins - Last 30 Days",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # BUSINESS INSIGHTS
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '💡 Business Insights'
        '</div>',
        unsafe_allow_html=True
    )

    highest_contract = contract_data.loc[
        contract_data["churn_rate"].idxmax(),
        "contract_type"
    ]

    highest_contract_rate = contract_data["churn_rate"].max()

    highest_internet = internet_data.loc[
        internet_data["churn_rate"].idxmax(),
        "internet_service"
    ]

    highest_internet_rate = internet_data["churn_rate"].max()

    churned_avg_satisfaction = df.loc[
        df["churn"] == 1,
        "satisfaction_score"
    ].mean()

    active_avg_satisfaction = df.loc[
        df["churn"] == 0,
        "satisfaction_score"
    ].mean()

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"""
            **Overall Churn Rate:** {churn_rate:.2f}%

            **Highest Churn Contract:** {highest_contract}
            ({highest_contract_rate:.2f}%)

            **Highest Churn Internet Service:** {highest_internet}
            ({highest_internet_rate:.2f}%)
            """
        )

    with col2:

        st.info(
            f"""
            **Average Satisfaction:** {avg_satisfaction:.2f}/10

            **Churned Customer Satisfaction:**
            {churned_avg_satisfaction:.2f}/10

            **Active Customer Satisfaction:**
            {active_avg_satisfaction:.2f}/10
            """
        )

    # --------------------------------------------------------
    # RECOMMENDED BUSINESS ACTIONS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🎯 Recommended Actions'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="info-box">
<b>1. Focus on high-risk customers</b><br>
Use the churn prediction model to identify customers who are most
likely to leave and prioritize retention activities.
</div>

<div class="info-box">
<b>2. Improve customer satisfaction</b><br>
Customers with lower satisfaction should receive targeted support
and proactive engagement.
</div>

<div class="info-box">
<b>3. Monitor contract-level churn</b><br>
Contract categories with higher churn rates should be investigated
for pricing, service, or retention opportunities.
</div>

<div class="info-box">
<b>4. Prioritize revenue at risk</b><br>
Combine churn probability with estimated customer value to identify
customers who could have the greatest business impact if they churn.
</div>

<div class="info-box">
<b>5. Monitor customer engagement</b><br>
Low login activity, high support interactions, and reduced usage can
be useful indicators for retention campaigns.
</div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with st.expander("📋 View Customer Dataset"):

        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Dataset",
            data=csv,
            file_name="customer_churn.csv",
            mime="text/csv"
        )


# ============================================================
# CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate the probability "
        "that the customer will churn."
    )

    if model is None or preprocessor is None:

        st.error(
            "Model files were not found. Please train the Logistic "
            "Regression model first."
        )

        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].unique())
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        tenure_months = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=120,
            value=12
        )

        contract_type = st.selectbox(
            "Contract Type",
            sorted(df["contract_type"].unique())
        )

    with col2:

        internet_service = st.selectbox(
            "Internet Service",
            sorted(df["internet_service"].unique())
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(df["payment_method"].unique())
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=float(df["monthly_charges"].mean())
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(df["total_charges"].mean())
        )

        auto_payment = st.selectbox(
            "Auto Payment",
            [0, 1]
        )

    with col3:

        discount = st.number_input(
            "Discount",
            min_value=0.0,
            value=0.0
        )

        support_tickets = st.number_input(
            "Support Tickets - 30 Days",
            min_value=0,
            value=2
        )

        logins = st.number_input(
            "Logins - 30 Days",
            min_value=0,
            value=10
        )

        usage_hours = st.number_input(
            "Usage Hours - 30 Days",
            min_value=0.0,
            value=50.0
        )

        last_login_days = st.number_input(
            "Days Since Last Login",
            min_value=0,
            value=5
        )

        satisfaction_score = st.slider(
            "Satisfaction Score",
            min_value=1,
            max_value=10,
            value=7
        )

    if st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "auto_payment": auto_payment,
            "discount": discount,
            "support_tickets_30d": support_tickets,
            "logins_30d": logins,
            "usage_hours_30d": usage_hours,
            "last_login_days": last_login_days,
            "satisfaction_score": satisfaction_score
        }])

        try:

            transformed = preprocessor.transform(input_data)

            probability = model.predict_proba(transformed)[0][1]

            prediction = int(probability >= prediction_threshold)

            st.markdown("---")

            if probability < 0.30:

                risk = "LOW"
                st.success(
                    f"🟢 Low Churn Risk — {probability * 100:.2f}%"
                )

            elif probability < 0.60:

                risk = "MEDIUM"
                st.warning(
                    f"🟡 Medium Churn Risk — {probability * 100:.2f}%"
                )

            else:

                risk = "HIGH"
                st.error(
                    f"🔴 High Churn Risk — {probability * 100:.2f}%"
                )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric("Risk Level", risk)

            with col3:
                st.metric(
                    "Prediction",
                    "Churn" if prediction == 1 else "No Churn"
                )

            # ------------------------------------------------
            # SHAP EXPLANATION
            # ------------------------------------------------

            st.markdown("### 🔍 Prediction Explanation")

            try:

                import shap

                explainer = shap.LinearExplainer(
                    model,
                    transformed
                )

                shap_values = explainer(transformed)

                values = shap_values.values[0]

                feature_names = preprocessor.get_feature_names_out()

                explanation_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": values
                })

                explanation_df["Absolute Impact"] = (
                    explanation_df["SHAP Value"].abs()
                )

                explanation_df = (
                    explanation_df
                    .sort_values("Absolute Impact", ascending=False)
                    .head(10)
                )

                fig = px.bar(
                    explanation_df.sort_values("SHAP Value"),
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    title="Top Factors Affecting Prediction"
                )

                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:

                st.info(
                    "SHAP explanation could not be generated "
                    f"for this prediction: {e}"
                )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "📂 Batch Prediction":

    st.title("📂 Batch Churn Prediction")

    st.write(
        "Upload a CSV or Excel file to generate churn probabilities "
        "for multiple customers."
    )

    if model is None or preprocessor is None:

        st.error("Model files are missing.")

        st.stop()

    uploaded_file = st.file_uploader(
        "Upload Customer Dataset",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.endswith(".csv"):
                batch_df = pd.read_csv(uploaded_file)
            else:
                batch_df = pd.read_excel(uploaded_file)

            st.success(f"Uploaded {len(batch_df):,} customers.")

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            required_columns = [
                "age",
                "gender",
                "senior_citizen",
                "tenure_months",
                "contract_type",
                "internet_service",
                "payment_method",
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

            missing_columns = [
                col for col in required_columns
                if col not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing columns: "
                    + ", ".join(missing_columns)
                )

            else:

                if st.button(
                    "🔮 Generate Predictions",
                    type="primary"
                ):

                    transformed = preprocessor.transform(
                        batch_df[required_columns]
                    )

                    probabilities = model.predict_proba(
                        transformed
                    )[:, 1]

                    predictions = (
                        probabilities >= prediction_threshold
                    ).astype(int)

                    result_df = batch_df.copy()

                    result_df["churn_probability"] = probabilities

                    result_df["prediction"] = np.where(
                        predictions == 1,
                        "Churn",
                        "No Churn"
                    )

                    result_df["risk_level"] = pd.cut(
                        probabilities,
                        bins=[-0.01, 0.30, 0.60, 1.01],
                        labels=["LOW", "MEDIUM", "HIGH"]
                    )

                    st.success("Predictions generated successfully.")

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        height=500
                    )

                    csv = result_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:

            st.error(f"Error processing file: {e}")


# ============================================================
# MODEL ANALYSIS
# ============================================================

elif page == "📈 Model Analysis":
    st.title("📈 Model Analysis")
    st.caption(
        "Compare models on the held-out test set. Tuned thresholds were selected "
        "using validation data; financial costs shown here are illustrative."
    )

    tuned_file = os.path.join(MODEL_DIR, "model_comparison.csv")
    standard_file = os.path.join(MODEL_DIR, "model_comparison_at_050.csv")

    if os.path.exists(tuned_file):
        comparison_df = pd.read_csv(tuned_file)
        comparison_df.columns = [str(c).strip() for c in comparison_df.columns]

        if comparison_df.empty:
            st.warning("The model comparison file is empty. Run src/compare_models.py first.")
        else:
            # Compact KPI cards
            st.subheader("At a glance — validation-tuned thresholds")
            k1, k2, k3, k4 = st.columns(4)
            if "ROC-AUC" in comparison_df.columns:
                best = comparison_df.loc[comparison_df["ROC-AUC"].idxmax()]
                k1.metric("Best ROC-AUC", str(best["Model"]), f'{best["ROC-AUC"]:.3f}')
            if "Recall" in comparison_df.columns:
                best = comparison_df.loc[comparison_df["Recall"].idxmax()]
                k2.metric("Best Recall", str(best["Model"]), f'{best["Recall"]:.1%}')
            if "F1 Score" in comparison_df.columns:
                best = comparison_df.loc[comparison_df["F1 Score"].idxmax()]
                k3.metric("Best F1 Score", str(best["Model"]), f'{best["F1 Score"]:.1%}')
            if "Illustrative Cost" in comparison_df.columns:
                best = comparison_df.loc[comparison_df["Illustrative Cost"].idxmin()]
                k4.metric("Lowest Demo Cost", str(best["Model"]), f'${best["Illustrative Cost"]:,.0f}')

            st.markdown("---")
            tuned_tab, standard_tab = st.tabs(["🎯 Tuned thresholds", "⚖️ Common 0.50 threshold"])

            with tuned_tab:
                st.subheader("Model comparison")
                st.caption("Each threshold was chosen on the validation set and evaluated on the separate test set.")
                display_columns = [c for c in [
                    "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1 Score",
                    "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP", "Illustrative Cost"
                ] if c in comparison_df.columns]
                styled = comparison_df[display_columns].style
                percent_cols = [c for c in ["Threshold", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"] if c in display_columns]
                if percent_cols:
                    styled = styled.format({c: "{:.3f}" for c in percent_cols})
                if "Illustrative Cost" in display_columns:
                    styled = styled.format({"Illustrative Cost": "${:,.0f}"})
                st.dataframe(styled, use_container_width=True, hide_index=True)

                metric_options = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"] if c in comparison_df.columns]
                if metric_options:
                    st.subheader("Compare one metric at a time")
                    default_metric = "Recall" if "Recall" in metric_options else metric_options[0]
                    selected_metric = st.selectbox("Metric", metric_options, index=metric_options.index(default_metric), key="tuned_metric_select")
                    chart_df = comparison_df.sort_values(selected_metric, ascending=True)
                    fig = px.bar(
                        chart_df, x=selected_metric, y="Model", orientation="h",
                        text=selected_metric, title=f"{selected_metric} by Model",
                        labels={selected_metric: selected_metric, "Model": "Model"},
                        hover_data=[c for c in ["Threshold", "FP", "FN", "Illustrative Cost"] if c in chart_df.columns]
                    )
                    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
                    fig.update_layout(height=330, margin=dict(l=15, r=55, t=65, b=25), xaxis=dict(range=[0, 1.08], title=selected_metric), yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                if all(c in comparison_df.columns for c in ["FP", "FN", "Model"]):
                    st.subheader("Classification errors on the test set")
                    error_df = comparison_df.melt(id_vars=["Model"], value_vars=["FP", "FN"], var_name="Error type", value_name="Customers")
                    fig = px.bar(error_df, x="Model", y="Customers", color="Error type", barmode="group", text="Customers")
                    fig.update_traces(textposition="outside", cliponaxis=False)
                    fig.update_layout(height=340, margin=dict(l=15, r=15, t=35, b=25), legend_title_text="")
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                if "Illustrative Cost" in comparison_df.columns:
                    st.info(
                        "Demo cost = false positives × 10 + false negatives × 100. These are illustrative assumptions, "
                        "not actual business losses or guaranteed savings. Choose thresholds using validated business costs."
                    )

            with standard_tab:
                if os.path.exists(standard_file):
                    standard_df = pd.read_csv(standard_file)
                    standard_df.columns = [str(c).strip() for c in standard_df.columns]
                    st.caption("All models use the same 0.50 probability threshold here. ROC-AUC and PR-AUC measure ranking quality and do not depend on this classification threshold.")
                    standard_display = [c for c in [
                        "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1 Score",
                        "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP", "Illustrative Cost"
                    ] if c in standard_df.columns]
                    st.dataframe(standard_df[standard_display], use_container_width=True, hide_index=True)
                    if all(c in standard_df.columns for c in ["Model", "Recall", "Precision", "F1 Score"]):
                        metric = st.selectbox("Metric to visualize", [c for c in ["Recall", "Precision", "F1 Score", "Accuracy", "ROC-AUC", "PR-AUC"] if c in standard_df.columns], key="standard_metric_select")
                        chart_df = standard_df.sort_values(metric, ascending=True)
                        fig = px.bar(chart_df, x=metric, y="Model", orientation="h", text=metric, title=f"{metric} — threshold 0.50")
                        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
                        fig.update_layout(height=320, margin=dict(l=15, r=55, t=60, b=25), xaxis=dict(range=[0, 1.08]), yaxis_title="")
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                else:
                    st.warning("models/model_comparison_at_050.csv was not found. Run python src/compare_models.py.")
    else:
        st.warning("models/model_comparison.csv was not found. Run python src/compare_models.py from the project folder.")

    # SHAP explainability: two compact global views and one larger local view.
    st.markdown("---")
    st.subheader("🔍 SHAP Explainability")
    st.caption(
        "These saved plots explain feature impact for the model analysis run that generated them; "
        "they may not correspond to every model shown above."
    )
    shap_dir = os.path.join(DATA_DIR, "shap")
    shap_paths = {
        "Global Feature Importance": os.path.join(shap_dir, "shap_global_bar.png"),
        "SHAP Beeswarm": os.path.join(shap_dir, "shap_beeswarm.png"),
        "Local Customer Explanation": os.path.join(shap_dir, "shap_local_waterfall.png"),
    }
    global_col1, global_col2 = st.columns(2, gap="large")
    for col, title in [(global_col1, "Global Feature Importance"), (global_col2, "SHAP Beeswarm")]:
        with col:
            path = shap_paths[title]
            st.markdown(f"**{title}**")
            if os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.caption("Plot not found. Run src/shap_analysis.py to generate it.")

    local_path = shap_paths["Local Customer Explanation"]
    if os.path.exists(local_path):
        with st.expander("🔎 View local customer explanation", expanded=False):
            st.caption("A local explanation shows which features influenced one selected example prediction.")
            st.image(local_path, use_container_width=True)
    else:
        st.caption("Local customer explanation not found. Run src/shap_analysis.py to generate it.")


# ============================================================
# CUSTOMER SURVIVAL
# ============================================================

elif page == "📈 Customer Survival":

    st.title("📈 Customer Survival Analysis")

    st.write(
        "Customer retention and survival analysis using tenure "
        "and churn information."
    )

    cohort_file = os.path.join(DATA_DIR, "cohort_analysis.csv")

    km_file = os.path.join(DATA_DIR, "kaplan_meier_survival.csv")

    cohort_image = os.path.join(DATA_DIR, "cohort_churn_rate.png")

    km_image = os.path.join(DATA_DIR, "kaplan_meier_curve.png")

    # --------------------------------------------------------
    # COHORT
    # --------------------------------------------------------

    if os.path.exists(cohort_file):

        cohort_df = pd.read_csv(cohort_file)

        st.subheader("Customer Cohort Analysis")

        st.dataframe(cohort_df, use_container_width=True)

    if os.path.exists(cohort_image):

        st.image(cohort_image, use_container_width=True)

    # --------------------------------------------------------
    # KAPLAN MEIER
    # --------------------------------------------------------

    st.subheader("Kaplan-Meier Customer Survival Curve")

    if os.path.exists(km_image):

        st.image(km_image, use_container_width=True)

    if os.path.exists(km_file):

        km_df = pd.read_csv(km_file)

        st.subheader("Customer Survival Probability")

        st.dataframe(km_df, use_container_width=True)

        if (
            "timeline" in km_df.columns
            and "survival_probability" in km_df.columns
        ):

            fig = px.line(
                km_df,
                x="timeline",
                y="survival_probability",
                markers=True,
                title="Survival Probability Over Tenure"
            )

            fig.update_layout(
                xaxis_title="Tenure (Months)",
                yaxis_title="Survival Probability",
                yaxis_range=[0, 1],
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Note: the current synthetic dataset uses tenure as the "
        "duration variable and churn as the event indicator. "
        "This is a simplified student-project demonstration of "
        "Kaplan-Meier analysis."
    )


# ============================================================
# RISK SEGMENTATION
# ============================================================

elif page == "🎯 Risk Segmentation":

    st.title("🎯 Customer Risk Segmentation")

    risk_file = os.path.join(DATA_DIR, "risk_segmented_customers.csv")

    if not os.path.exists(risk_file):

        st.warning("risk_segmented_customers.csv was not found.")

        st.stop()

    risk_df = pd.read_csv(risk_file)

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    low_count = (risk_df["risk_level"] == "LOW").sum()

    medium_count = (risk_df["risk_level"] == "MEDIUM").sum()

    high_count = (risk_df["risk_level"] == "HIGH").sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🟢 Low Risk", f"{low_count:,}")

    with col2:
        st.metric("🟡 Medium Risk", f"{medium_count:,}")

    with col3:
        st.metric("🔴 High Risk", f"{high_count:,}")

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    risk_distribution = (
        risk_df["risk_level"]
        .value_counts()
        .reset_index()
    )

    risk_distribution.columns = ["Risk Level", "Customers"]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            risk_distribution,
            names="Risk Level",
            values="Customers",
            hole=0.45,
            title="Customer Risk Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.bar(
            risk_distribution,
            x="Risk Level",
            y="Customers",
            text="Customers",
            title="Customers by Risk Level"
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # RISK CHARACTERISTICS
    # --------------------------------------------------------

    st.subheader("Risk Group Characteristics")

    characteristics = (
        risk_df.groupby("risk_level")
        .agg(
            customers=("customer_id", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_monthly_charges=("monthly_charges", "mean"),
            avg_satisfaction=("satisfaction_score", "mean")
        )
        .reset_index()
    )

    st.dataframe(characteristics, use_container_width=True)

    # --------------------------------------------------------
    # HIGH RISK CUSTOMERS
    # --------------------------------------------------------

    st.subheader("🔴 Highest-Risk Customers")

    high_risk = (
        risk_df[risk_df["risk_level"] == "HIGH"]
        .sort_values("churn_probability", ascending=False)
        .head(20)
    )

    st.dataframe(high_risk, use_container_width=True)

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv = risk_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Risk Segmentation",
        data=csv,
        file_name="risk_segmented_customers.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    st.subheader("🎯 Recommended Retention Strategy")

    st.markdown("""
    **🔴 HIGH RISK**

    - Proactive retention intervention
    - Personalized customer support
    - Review billing/service issues
    - Consider targeted retention offers

    **🟡 MEDIUM RISK**

    - Targeted retention campaigns
    - Monitor customer engagement
    - Improve satisfaction

    **🟢 LOW RISK**

    - Normal customer engagement
    - Maintain service quality
    - Monitor future behavior
    """)


# ============================================================
# RETENTION PRIORITY
# ============================================================

elif page == "💰 Retention Priority":

    st.title("💰 Retention Priority Analysis")

    st.write(
        "Prioritize customers using churn probability and "
        "estimated customer value."
    )

    priority_file = os.path.join(DATA_DIR, "retention_priority.csv")

    if not os.path.exists(priority_file):

        st.warning("retention_priority.csv was not found.")

        st.stop()

    priority_df = pd.read_csv(priority_file)

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    critical_count = (
        priority_df["retention_priority"] == "CRITICAL"
    ).sum()

    high_count = (
        priority_df["retention_priority"] == "HIGH"
    ).sum()

    estimated_value = priority_df["estimated_customer_value"].sum()

    revenue_at_risk = priority_df["expected_revenue_at_risk"].sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔴 Critical Customers", f"{critical_count:,}")

    with col2:
        st.metric("🟠 High Priority", f"{high_count:,}")

    with col3:
        st.metric(
            "Estimated Customer Value",
            f"${estimated_value:,.2f}"
        )

    with col4:
        st.metric(
            "Expected Revenue at Risk",
            f"${revenue_at_risk:,.2f}"
        )

    # --------------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------------

    priority_distribution = (
        priority_df["retention_priority"]
        .value_counts()
        .reset_index()
    )

    priority_distribution.columns = ["Priority", "Customers"]

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(
            priority_distribution,
            names="Priority",
            values="Customers",
            hole=0.45,
            title="Retention Priority Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        revenue_data = (
            priority_df
            .groupby("retention_priority")["expected_revenue_at_risk"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            revenue_data,
            x="retention_priority",
            y="expected_revenue_at_risk",
            text="expected_revenue_at_risk",
            title="Revenue at Risk by Priority"
        )

        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside"
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------
    # TOP CUSTOMERS
    # --------------------------------------------------------

    st.subheader("🎯 Top Retention-Priority Customers")

    top_customers = (
        priority_df
        .sort_values("retention_priority_score", ascending=False)
        .head(20)
    )

    st.dataframe(top_customers, use_container_width=True)

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    selected_priority = st.multiselect(
        "Filter by Priority",
        sorted(priority_df["retention_priority"].dropna().unique()),
        default=sorted(priority_df["retention_priority"].dropna().unique())
    )

    filtered_df = priority_df[
        priority_df["retention_priority"].isin(selected_priority)
    ]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        height=450
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    csv = priority_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Retention Priority",
        data=csv,
        file_name="retention_priority.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------
    # STRATEGY
    # --------------------------------------------------------

    st.subheader("🎯 Retention Strategy")

    st.markdown("""
    **🔴 CRITICAL**

    Immediate personalized retention intervention.

    **🟠 HIGH**

    Priority retention campaign.

    **🟡 MEDIUM-HIGH**

    Targeted retention campaign.

    **🟡 MEDIUM**

    Monitor and engage the customer.

    **🟢 LOW**

    Normal customer engagement.
    """)

    st.info(
        "The estimated customer value and revenue-at-risk values "
        "are simplified student-project estimates. A production "
        "system would use a complete Customer Lifetime Value model "
        "and business-specific retention costs."
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About Customer Churn Prediction")

    st.markdown("""
    ## Customer Churn Prediction for Subscription Businesses

    This project uses machine learning to identify customers who
    are at risk of leaving a subscription-based service.

    ### 🎯 Project Goal

    Predict customer churn and identify the factors that contribute
    to customer loss so that businesses can take proactive retention
    actions.

    ### 🧠 Machine Learning Models

    The project compares:

    - Logistic Regression
    - Random Forest
    - XGBoost

    The models are evaluated using:

    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - PR-AUC

    ### 🔍 Explainability

    SHAP is used to explain which features contribute to individual
    churn predictions and overall model behavior.

    ### 📊 Business Analysis

    The dashboard provides:

    - Customer churn analysis
    - Contract analysis
    - Internet service analysis
    - Revenue analysis
    - Customer survival analysis
    - Risk segmentation
    - Retention prioritization

    ### 🛠️ Technology Stack

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - XGBoost
    - SHAP
    - Plotly
    - Streamlit
    - Lifelines
    - Joblib

    ### ⚠️ Risk Thresholds

    The demonstration application uses:

    - Below 30% → LOW
    - 30%–60% → MEDIUM
    - Above 60% → HIGH

    These thresholds are demonstration values. A production system
    should determine thresholds using business costs, retention
    campaign effectiveness, and expected financial value.

    ### 📌 Dataset

    The current project uses a synthetic customer churn dataset
    containing customer demographics, billing information, service
    usage, support interactions, satisfaction, and churn status.
    """)

    st.markdown("---")

    st.subheader("📁 Project Structure")

    st.code("""
customer-churn-prediction/
│
├── data/
│   ├── customer_churn.csv
│   ├── cohort_analysis.csv
│   ├── kaplan_meier_survival.csv
│   ├── risk_segmented_customers.csv
│   ├── retention_priority.csv
│   └── shap/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── model_preprocessor.pkl
│   ├── random_forest.pkl
│   ├── random_forest_preprocessor.pkl
│   ├── xgboost.pkl
│   ├── xgboost_preprocessor.pkl
│   └── model_comparison.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── compare_models.py
│   ├── shap_analysis.py
│   ├── survival_analysis.py
│   ├── risk_segmentation.py
│   └── retention_priority.py
│
├── app.py
├── README.md
└── requirements.txt
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    'Customer Churn Prediction • Data Science Project • '
    'Built with Python & Streamlit'
    '</div>',
    unsafe_allow_html=True
)