
"""Acquisition-month retention cohorts.

The included demo history is synthetic.
For real analysis, pass actual customer history.
"""

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent

REQUIRED = {
    "customer_id",
    "acquisition_date",
    "observation_date",
    "churn_date",
}


def build_cohorts(data):
    missing = sorted(REQUIRED - set(data.columns))

    if missing:
        raise ValueError(
            "Missing required fields: " + ", ".join(missing)
        )

    df = data.copy()

    for col in (
        "acquisition_date",
        "observation_date",
        "churn_date",
    ):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if df[["acquisition_date", "observation_date"]].isna().any().any():
        raise ValueError(
            "Acquisition and observation dates must be valid."
        )

    if (df["acquisition_date"] > df["observation_date"]).any():
        raise ValueError(
            "Acquisition date cannot be after observation date."
        )

    df["acquisition_cohort"] = (
        df["acquisition_date"].dt.to_period("M").astype(str)
    )

    df["age_months"] = (
        (df["observation_date"].dt.year - df["acquisition_date"].dt.year)
        * 12
        + df["observation_date"].dt.month
        - df["acquisition_date"].dt.month
    )

    records = []


    for r in df.itertuples(index=False):
        for month in range(int(r.age_months) + 1):
            period_end = (
                r.acquisition_date
                + pd.DateOffset(months=month + 1)
            ).normalize() - pd.DateOffset(days=1)

            observed = period_end <= r.observation_date
            retained = (
                pd.isna(r.churn_date)
                or r.churn_date > period_end
            )

            if observed:
                records.append({
                    "customer_id": str(r.customer_id),
                    "acquisition_cohort": r.acquisition_cohort,
                    "months_since_acquisition": month,
                    "retained": int(retained),
                })


    long = pd.DataFrame(records)

    if long.empty:
        raise ValueError(
            "No complete cohort periods were found in the input data."
        )

    summary = (
        long.groupby(
            ["acquisition_cohort", "months_since_acquisition"],
            as_index=False,
        )
        .agg(
            customers_observed=("customer_id", "nunique"),
            retained_customers=("retained", "sum"),
        )
    )

    summary["retention_rate_pct"] = (
        100
        * summary["retained_customers"]
        / summary["customers_observed"]
    )

    return long, summary


def save_heatmap(matrix, output_path, is_synthetic):
    row_count, col_count = matrix.shape

    # Increase figure size when there are many cohorts or months.
    fig_width = max(12, min(20, col_count * 0.65 + 5))
    fig_height = max(7, min(22, row_count * 0.28 + 3))

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    # Keep labels readable for large datasets.
    x_step = max(1, col_count // 12)
    y_step = max(1, row_count // 24)

    x_labels = [
        str(col) if i % x_step == 0 else ""
        for i, col in enumerate(matrix.columns)
    ]

    y_labels = [
        str(label) if i % y_step == 0 else ""
        for i, label in enumerate(matrix.index)
    ]

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=100,
        mask=matrix.isna(),
        xticklabels=x_labels,
        yticklabels=y_labels,
        linewidths=0.15,
        linecolor="white",
        cbar_kws={
            "label": "Retention rate (%)",
            "shrink": 0.8,
            "ticks": [0, 20, 40, 60, 80, 100],
        },
    )

    title = "Acquisition-Month Cohort Retention"
    if is_synthetic:
        title += "\n(Synthetic demo data — not real customer history)"

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("Months Since Acquisition", fontsize=11, labelpad=10)
    ax.set_ylabel("Acquisition Cohort", fontsize=11, labelpad=10)

    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)

    # Add percentages inside cells only when the chart is small enough.
    if row_count <= 18 and col_count <= 15:
        for y in range(row_count):
            for x in range(col_count):
                value = matrix.iloc[y, x]
                if pd.notna(value):
                    ax.text(
                        x + 0.5,
                        y + 0.5,
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value < 50 else "black",
                    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "customer_history_demo.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "acquisition_cohorts"),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    data = pd.read_csv(input_path)

    long, summary = build_cohorts(data)

    output_dir.mkdir(parents=True, exist_ok=True)

    long.to_csv(
        output_dir / "acquisition_cohort_retention_long.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "acquisition_cohort_retention_summary.csv",
        index=False,
    )

    matrix = summary.pivot(
        index="acquisition_cohort",
        columns="months_since_acquisition",
        values="retention_rate_pct",
    ).sort_index()

    matrix.to_csv(
        output_dir / "acquisition_cohort_retention_heatmap.csv"
    )

    provenance = (
        data["data_provenance"].astype(str).str.upper()
        if "data_provenance" in data.columns
        else pd.Series(dtype=str)
    )

    is_synthetic = provenance.str.contains(
        "SYNTHETIC_DEMO_ONLY",
        regex=False,
    ).any()

    save_heatmap(
        matrix,
        output_dir / "acquisition_cohort_retention_heatmap.png",
        is_synthetic,
    )

    print(f"Saved cohort analysis to: {output_dir.resolve()}")
    print(f"Acquisition cohorts: {matrix.shape[0]}")
    print(f"Maximum months displayed: {matrix.shape[1]}")
    print("Generated files:")
    print("- acquisition_cohort_retention_heatmap.png")
    print("- acquisition_cohort_retention_heatmap.csv")
    print("- acquisition_cohort_retention_summary.csv")
    print("- acquisition_cohort_retention_long.csv")


if __name__ == "__main__":
    main()
