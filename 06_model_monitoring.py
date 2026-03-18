# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring Notebook
# MAGIC
# MAGIC Scores old predictions against newly available actuals and tracks model staleness.
# MAGIC
# MAGIC When the data table gets a new version with updated BB values, this notebook joins
# MAGIC previous predictions against the now-available actuals and computes realized accuracy
# MAGIC metrics. Also checks how old each champion model is and flags stale partitions.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------

import sys
import os

notebook_path = os.getcwd()
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

# COMMAND ----------

from datetime import datetime

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import mlflow

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Modules
# MAGIC
# MAGIC Pipeline utilities from `modeling_dev/`: data loading, Delta lineage logging.

# COMMAND ----------

from modeling_dev.data_utils import (
    load_latest_version,
    enable_iceberg,
    write_pipeline_errors,
    truncate_partition_name,
    is_valid_metric,
)
from modeling_dev.mlflow_utils import log_delta_lineage

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration
# MAGIC
# MAGIC Widget parameters plus YAML-driven thresholds for backtest scoring and staleness
# MAGIC monitoring. Thresholds are in the `MONITORING` section of `model_config.yaml`.

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("experiment_name", "/Shared/dab-pipeline-forecasting")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib")
dbutils.widgets.text("config_path", "configs/model_config.yaml")
dbutils.widgets.text("scoring_table", "scoring_output")
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO")
dbutils.widgets.dropdown("is_synthetic_data", "true", ["true", "false"])
dbutils.widgets.dropdown("enable_iceberg", "true", ["true", "false"])

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
DATA_TABLE_NAME = dbutils.widgets.get("data_table")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{DATA_TABLE_NAME}"
CONFIG_PATH = dbutils.widgets.get("config_path")
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('scoring_table')}"

PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.{DATA_TABLE_NAME}_predictions"
METRICS_TABLE = f"{CATALOG}.{SCHEMA}.predictions_metrics"
MONITORING_TABLE = f"{CATALOG}.{SCHEMA}.{DATA_TABLE_NAME}_monitoring"

PARTITION_COLS = [
    c.strip() for c in dbutils.widgets.get("partition_columns").split(",")
]
IS_SYNTHETIC_DATA = dbutils.widgets.get("is_synthetic_data") == "true"
ENABLE_ICEBERG = dbutils.widgets.get("enable_iceberg") == "true"

print(f"Data table: {DATA_TABLE}")
print(f"Predictions table: {PREDICTIONS_TABLE}")
print(f"Metrics table: {METRICS_TABLE}")
print(f"Monitoring output: {MONITORING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
with open("configs/variable_config.yaml", "r") as f:
    variable_config = yaml.safe_load(f)

monitoring_config = config.get("MONITORING", {})
backtest_config = monitoring_config.get("BACKTEST_ACTUALS", {})
staleness_config = monitoring_config.get("STALENESS", {})

N_PREVIOUS_VERSIONS = backtest_config.get("N_PREVIOUS_VERSIONS", 3)
MAPE_RATIO_ALERT_THRESHOLD = backtest_config.get("MAPE_RATIO_ALERT_THRESHOLD", 2.0)

MAX_MODEL_AGE_DAYS = staleness_config.get("MAX_MODEL_AGE_DAYS", 30)

# Display config summary
config_summary = pd.DataFrame([
    {"Check": "Backtest", "Parameter": "N previous versions", "Value": N_PREVIOUS_VERSIONS},
    {"Check": "Backtest", "Parameter": "MAPE ratio alert threshold", "Value": MAPE_RATIO_ALERT_THRESHOLD},
    {"Check": "Staleness", "Parameter": "Max model age (days)", "Value": MAX_MODEL_AGE_DAYS},
])
display(config_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Score Predictions vs True Values
# MAGIC
# MAGIC Core monitoring function: load old predictions where actuals were not available
# MAGIC at inference time, join against the current data table to get the now-available BB values,
# MAGIC and compute realized accuracy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Current Actuals
# MAGIC
# MAGIC Load the latest version of the horizon data table. The BB values here are the "ground truth"
# MAGIC that old predictions are scored against. Only rows with non-null, non-zero BB are kept.

# COMMAND ----------

SELECT_COLUMNS = list(variable_config["SELECT_COLUMNS"])
for col in PARTITION_COLS:
    if col not in SELECT_COLUMNS:
        SELECT_COLUMNS.append(col)

df_actuals, CURRENT_DATA_VERSION = load_latest_version(
    spark, DATA_TABLE, SELECT_COLUMNS, is_synthetic_data=IS_SYNTHETIC_DATA
)
df_actuals = df_actuals.withColumn(
    "PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLS])
)

# Keep only rows where BB is non-null and non-zero (valid actuals)
df_actuals = df_actuals.filter(F.col("BB").isNotNull() & (F.col("BB") != 0))
print(f"Current data version: {CURRENT_DATA_VERSION}")
print(f"Rows with valid actuals: {df_actuals.count()}")

# COMMAND ----------

# Summary of actuals by partition
actuals_summary = (
    df_actuals.groupBy("PARTITION_COLUMN")
    .agg(
        F.count("*").alias("row_count"),
        F.min("QUARTER").alias("earliest_quarter"),
        F.max("QUARTER").alias("latest_quarter"),
    )
    .orderBy("PARTITION_COLUMN")
    .toPandas()
)
print(f"{len(actuals_summary)} partitions with valid actuals:")
display(actuals_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Previous Predictions
# MAGIC
# MAGIC Retrieve the N most recent prediction versions (configured via `N_PREVIOUS_VERSIONS`)
# MAGIC for backtesting against current actuals.

# COMMAND ----------

backtest_results = []
backtest_errors = []
versions_to_check = []

if not spark.catalog.tableExists(PREDICTIONS_TABLE):
    print(f"Predictions table {PREDICTIONS_TABLE} does not exist. Skipping backtest.")
    all_versions = []
else:
    # Get distinct prediction versions with metadata
    all_versions = (
        spark.table(PREDICTIONS_TABLE)
        .groupBy("PREDICTIONS_VERSION", "LOAD_TS")
        .agg(
            F.countDistinct("PARTITION_COLUMN").alias("partition_count"),
            F.count("*").alias("row_count"),
        )
        .orderBy(F.desc("LOAD_TS"))
        .collect()
    )

    if len(all_versions) == 0:
        print("No prediction versions found. Skipping backtest.")
    else:
        versions_to_check = [
            row["PREDICTIONS_VERSION"] for row in all_versions[:N_PREVIOUS_VERSIONS]
        ]

        # Display version inventory
        version_summary = pd.DataFrame([
            {
                "PREDICTIONS_VERSION": row["PREDICTIONS_VERSION"],
                "LOAD_TS": row["LOAD_TS"],
                "Partitions": row["partition_count"],
                "Rows": row["row_count"],
                "Selected": row["PREDICTIONS_VERSION"] in versions_to_check,
            }
            for row in all_versions
        ])
        print(f"{len(all_versions)} prediction version(s) found, backtesting {len(versions_to_check)}:")
        display(version_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Each Version Against Current Actuals
# MAGIC
# MAGIC For each prediction version, find rows where BILLING was null at inference time (future quarters),
# MAGIC join against current actuals, and compute realized APE. Compare realized vs estimated MAPE
# MAGIC and flag partitions where the ratio exceeds `MAPE_RATIO_ALERT_THRESHOLD`.

# COMMAND ----------

# Pre-load estimated MAPE for all champion partitions into a dict.
# This avoids N+1 queries (one per partition) in the scoring loop below.
metrics_cols = (
    {f.name for f in spark.table(METRICS_TABLE).schema}
    if spark.catalog.tableExists(METRICS_TABLE) else set()
)
estimated_mapes = {}
if "is_champion" in metrics_cols and "HORIZON_MAPE" in metrics_cols:
    _w = Window.partitionBy("PARTITION_COLUMN").orderBy(F.desc("LOAD_TS"))
    est_df = (
        spark.table(METRICS_TABLE)
        .filter(F.col("DATA_TABLE") == DATA_TABLE)
        .filter(F.col("is_champion") == True)
        .withColumn("_rn", F.row_number().over(_w))
        .filter(F.col("_rn") == 1)
        .select("PARTITION_COLUMN", "HORIZON_MAPE")
        .toPandas()
    )
    estimated_mapes = dict(zip(est_df["PARTITION_COLUMN"], est_df["HORIZON_MAPE"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dedup Check and Scoring Loop

# COMMAND ----------

scored_dfs = {}  # pred_version -> cached scored Spark DataFrame

# Dedup check: skip backtest if all prediction versions have already been scored
# against this data version (prevents duplicate records on repeated trigger firings)
skip_backtest = False
if spark.catalog.tableExists(MONITORING_TABLE) and versions_to_check:
    mon_cols = {f.name for f in spark.table(MONITORING_TABLE).schema}
    if "data_version_scored_against" in mon_cols:
        already_scored = set(
            row["current_version"]
            for row in spark.table(MONITORING_TABLE)
            .filter(F.col("DATA_TABLE") == DATA_TABLE)
            .filter(F.col("check_type") == "backtest_actuals")
            .filter(F.col("data_version_scored_against") == str(CURRENT_DATA_VERSION))
            .select("current_version")
            .distinct()
            .collect()
        )
        if set(versions_to_check).issubset(already_scored):
            print(f"All prediction versions already scored against data version {CURRENT_DATA_VERSION}. Skipping backtest.")
            skip_backtest = True

# COMMAND ----------

# MAGIC %md
# MAGIC ### Score Predictions per Version

# COMMAND ----------

if not skip_backtest and spark.catalog.tableExists(PREDICTIONS_TABLE) and len(all_versions) > 0:
    for pred_version in versions_to_check:
        print(f"\nScoring {pred_version}...")

        # Load predictions for this version
        pred_df = (
            spark.table(PREDICTIONS_TABLE)
            .filter(F.col("PREDICTIONS_VERSION") == pred_version)
            .select(
                "PARTITION_COLUMN",
                "QUARTER",
                "BILLING",
                "BILLING_PREDICTED",
                "APE",
                "PREDICTION_HORIZON",
                "MODEL_USED",
                "DATA_VERSION",
            )
        )

        # "Unscored" = rows where BILLING was null/zero at inference time (future quarters).
        # Now that actuals are available, we can score these predictions.
        unscored = pred_df.filter(F.col("BILLING").isNull() | (F.col("BILLING") == 0))

        # REALIZED_APE: how far off the prediction was from the now-known actual value
        scored = unscored.join(
            df_actuals.select(
                "PARTITION_COLUMN", "QUARTER", F.col("BB").alias("BB_ACTUAL")
            ),
            on=["PARTITION_COLUMN", "QUARTER"],
            how="inner",
        ).withColumn(
            "REALIZED_APE",
            F.abs(F.col("BB_ACTUAL") - F.col("BILLING_PREDICTED"))
            / F.col("BB_ACTUAL")
            * 100,
        )

        n_scored = scored.count()
        if n_scored == 0:
            print(f"  No newly scorable rows for {pred_version}")
            continue

        scored_dfs[pred_version] = scored

        # Aggregate per partition
        partition_results = (
            scored.groupBy("PARTITION_COLUMN", "MODEL_USED")
            .agg(
                F.mean("REALIZED_APE").alias("realized_mape"),
                F.count("*").alias("n_quarters_scored"),
                F.mean("PREDICTION_HORIZON").alias("avg_prediction_horizon"),
            )
            .toPandas()
        )

        for _, row in partition_results.iterrows():
            estimated_mape = estimated_mapes.get(row["PARTITION_COLUMN"])

            # mape_ratio: realized / estimated (>1 means worse than expected)
            realized = row["realized_mape"]
            mape_ratio = (
                (realized / estimated_mape)
                if estimated_mape and estimated_mape > 0
                else None
            )
            # Alert if ratio exceeds threshold (e.g., 2x means realized is double the estimated MAPE)
            is_alert = (
                mape_ratio is not None and mape_ratio > MAPE_RATIO_ALERT_THRESHOLD
            )

            backtest_results.append(
                {
                    "PARTITION_COLUMN": row["PARTITION_COLUMN"],
                    "MODEL_USED": row["MODEL_USED"],
                    "PREDICTIONS_VERSION": pred_version,
                    "n_quarters_scored": int(row["n_quarters_scored"]),
                    "avg_prediction_horizon": float(row["avg_prediction_horizon"]),
                    "realized_mape": float(realized),
                    "estimated_mape": float(estimated_mape) if estimated_mape else None,
                    "mape_ratio": float(mape_ratio) if mape_ratio else None,
                    "is_alert": is_alert,
                }
            )

            if is_alert:
                backtest_errors.append(
                    {
                        "stage": "monitoring_backtest",
                        "partition": row["PARTITION_COLUMN"],
                        "model_type": row["MODEL_USED"],
                        "severity": "warning",
                        "message": (
                            f"Realized MAPE ({realized:.1f}%) is {mape_ratio:.1f}x the estimated MAPE "
                            f"({estimated_mape:.1f}%) for {pred_version}"
                        ),
                    }
                )

        print(f"  Scored {n_scored} rows across {len(partition_results)} partitions")

backtest_pdf = pd.DataFrame(backtest_results)
print(
    f"\nBacktest complete: {len(backtest_results)} partition-version results, {len(backtest_errors)} alerts"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Backtest Results
# MAGIC
# MAGIC Full results table sorted by realized MAPE (worst first), followed by alert-only view.

# COMMAND ----------

if not backtest_pdf.empty:
    display_df = backtest_pdf.copy()
    display_df["status"] = display_df["is_alert"].map({True: "ALERT", False: "OK"})
    display_df["partition"] = display_df["PARTITION_COLUMN"].apply(
        lambda p: truncate_partition_name(p, 30)
    )
    display(
        display_df[
            ["partition", "MODEL_USED", "PREDICTIONS_VERSION", "n_quarters_scored",
             "realized_mape", "estimated_mape", "mape_ratio", "status"]
        ].sort_values("realized_mape", ascending=False)
    )
else:
    print("No backtest results (no old predictions could be scored against current actuals).")

# COMMAND ----------

# Alert-only view
if not backtest_pdf.empty:
    alerts_df = backtest_pdf[backtest_pdf["is_alert"]].copy()
    if not alerts_df.empty:
        alerts_df["partition"] = alerts_df["PARTITION_COLUMN"].apply(
            lambda p: truncate_partition_name(p, 30)
        )
        print(f"{len(alerts_df)} backtest alert(s) -- realized MAPE significantly exceeds estimated:")
        display(
            alerts_df[
                ["partition", "MODEL_USED", "PREDICTIONS_VERSION",
                 "realized_mape", "estimated_mape", "mape_ratio"]
            ].sort_values("mape_ratio", ascending=False)
        )
    else:
        print("No backtest alerts. All realized MAPEs are within expected bounds.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Horizon Realized Accuracy
# MAGIC
# MAGIC Break down realized MAPE by prediction horizon (Q+1 through Q+6). Near-horizon predictions
# MAGIC should be more accurate than far-horizon. Large deviations from this pattern may indicate
# MAGIC data issues or model miscalibration at specific horizons.

# COMMAND ----------

horizon_backtest_pdf = pd.DataFrame()

if scored_dfs:
    horizon_results = []

    for pred_version, scored in scored_dfs.items():
        per_horizon = (
            scored.groupBy("PREDICTION_HORIZON")
            .agg(
                F.mean("REALIZED_APE").alias("realized_mape"),
                F.count("*").alias("n_quarters"),
            )
            .toPandas()
        )
        per_horizon["PREDICTIONS_VERSION"] = pred_version
        horizon_results.append(per_horizon)

    del scored_dfs

    if horizon_results:
        horizon_backtest_pdf = pd.concat(horizon_results, ignore_index=True)
        horizon_backtest_pdf = horizon_backtest_pdf.sort_values(
            ["PREDICTIONS_VERSION", "PREDICTION_HORIZON"]
        )
        print("Per-horizon realized MAPE:")
        display(horizon_backtest_pdf)
    else:
        print("No per-horizon backtest data available.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Model Staleness Check
# MAGIC
# MAGIC Per-partition check of how old the champion models are (days since last training run).
# MAGIC Stale models may be trained on outdated data and should be retrained.

# COMMAND ----------

staleness_results = []
staleness_alerts = []
staleness_pdf = pd.DataFrame()

if not spark.catalog.tableExists(SCORING_TABLE):
    print(f"Scoring table {SCORING_TABLE} does not exist. Skipping staleness check.")
else:
    scoring_cols = {f.name for f in spark.table(SCORING_TABLE).schema}
    scoring_df = spark.table(SCORING_TABLE).filter(F.col("DATA_TABLE") == DATA_TABLE)

    # Filter to champions if the column exists
    if "is_champion" in scoring_cols:
        scoring_df = scoring_df.filter(F.col("is_champion") == True)

    # Per-partition staleness: get latest LOAD_TS per partition
    staleness_spark = (
        scoring_df.groupBy("PARTITION_COLUMN", "MODEL_USED")
        .agg(F.max("LOAD_TS").alias("last_trained"))
        .orderBy("PARTITION_COLUMN")
    )
    staleness_pdf = staleness_spark.toPandas()

    if staleness_pdf.empty:
        print("No champion models found in scoring table.")
    else:
        now = datetime.now()
        staleness_pdf["model_age_days"] = staleness_pdf["last_trained"].apply(
            lambda ts: (now - ts).days if ts else None
        )
        staleness_pdf["is_stale"] = staleness_pdf["model_age_days"] > MAX_MODEL_AGE_DAYS
        staleness_pdf["partition"] = staleness_pdf["PARTITION_COLUMN"].apply(
            lambda p: truncate_partition_name(p, 30)
        )

        n_stale = staleness_pdf["is_stale"].sum()
        print(f"Model staleness: {len(staleness_pdf)} partitions, {n_stale} stale (>{MAX_MODEL_AGE_DAYS} days)")
        display(
            staleness_pdf[
                ["partition", "MODEL_USED", "last_trained", "model_age_days", "is_stale"]
            ].sort_values("model_age_days", ascending=False)
        )

        # Build per-partition results (all partitions) and alerts (stale only)
        for _, row in staleness_pdf.iterrows():
            if row["is_stale"]:
                staleness_alerts.append(
                    {
                        "stage": "monitoring_staleness",
                        "partition": row["PARTITION_COLUMN"],
                        "model_type": row["MODEL_USED"],
                        "severity": "warning",
                        "message": f"Champion model is {row['model_age_days']} days old (threshold: {MAX_MODEL_AGE_DAYS})",
                    }
                )

            staleness_results.append(
                {
                    "PARTITION_COLUMN": row["PARTITION_COLUMN"],
                    "MODEL_USED": row["MODEL_USED"],
                    "last_trained": row["last_trained"],
                    "model_age_days": int(row["model_age_days"]),
                    "is_stale": bool(row["is_stale"]),
                }
            )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Age Chart

# COMMAND ----------

staleness_fig = None

if not staleness_pdf.empty:
    plot_df = staleness_pdf.sort_values("model_age_days", ascending=True)
    n = len(plot_df)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4)))

    colors = [
        "#e74c3c" if row["is_stale"] else "#2ecc71" for _, row in plot_df.iterrows()
    ]
    labels = plot_df["partition"].values

    ax.barh(labels, plot_df["model_age_days"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(
        MAX_MODEL_AGE_DAYS, color="red", linestyle="--", alpha=0.5,
        label=f"Threshold ({MAX_MODEL_AGE_DAYS} days)",
    )

    # Annotate age on each bar
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.annotate(
            f"{row['model_age_days']}d",
            (row["model_age_days"] + 0.5, i), va="center", fontsize=9,
        )

    ax.set_xlabel("Model Age (days)")
    ax.set_title("Champion Model Age by Partition (red = stale)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    staleness_fig = fig
    display(fig)
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Monitoring Health Heatmap
# MAGIC
# MAGIC Single at-a-glance view: each partition vs each check type (backtest, staleness).
# MAGIC Green = pass, red = fail/alert.

# COMMAND ----------

def create_health_heatmap(all_partitions, check_results):
    """Build a red/green heatmap of partition health across check types.

    Args:
        all_partitions: sorted list of partition names
        check_results: dict of {check_name: set_of_failed_partitions}
    Returns:
        matplotlib Figure or None if no partitions
    """
    if not all_partitions:
        return None

    check_names = list(check_results.keys())
    n_parts = len(all_partitions)
    n_checks = len(check_names)

    # Build NxM binary matrix: rows=partitions, cols=checks. 1=failed, 0=passed.
    matrix = np.zeros((n_parts, n_checks))
    for j, check in enumerate(check_names):
        for i, part in enumerate(all_partitions):
            if part in check_results[check]:
                matrix[i, j] = 1

    # Colormap: green (#2ecc71) = passed, red (#e74c3c) = failed/alert
    cmap = mcolors.ListedColormap(["#2ecc71", "#e74c3c"])
    fig, ax = plt.subplots(figsize=(max(6, n_checks * 1.5), max(5, n_parts * 0.35)))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    labels = [truncate_partition_name(p, 25) for p in all_partitions]
    ax.set_yticks(range(n_parts))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(range(n_checks))
    ax.set_xticklabels(check_names, fontsize=10, rotation=30, ha="right")

    # Cell text
    for i in range(n_parts):
        for j in range(n_checks):
            text = "FAIL" if matrix[i, j] == 1 else "OK"
            color = "white" if matrix[i, j] == 1 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Monitoring Health by Partition")
    plt.tight_layout()
    return fig

# COMMAND ----------

# Collect all partitions and failure sets
all_known_partitions = set()
if not backtest_pdf.empty:
    all_known_partitions |= set(backtest_pdf["PARTITION_COLUMN"])
if not staleness_pdf.empty:
    all_known_partitions |= set(staleness_pdf["PARTITION_COLUMN"])

backtest_failed = set(
    backtest_pdf[backtest_pdf["is_alert"]]["PARTITION_COLUMN"]
) if not backtest_pdf.empty else set()

staleness_failed = set(
    staleness_pdf[staleness_pdf["is_stale"]]["PARTITION_COLUMN"]
) if not staleness_pdf.empty else set()

check_results = {
    "Backtest": backtest_failed,
    "Staleness": staleness_failed,
}

health_heatmap_fig = create_health_heatmap(sorted(all_known_partitions), check_results)
if health_heatmap_fig:
    display(health_heatmap_fig)
    plt.close(health_heatmap_fig)
else:
    print("No partition data available for health heatmap.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # MLflow Logging
# MAGIC
# MAGIC Resume the training run that produced the current champion models (looked up via
# MAGIC `run_id` in the scoring table) and log monitoring metrics, charts, and artifact tables.

# COMMAND ----------

mlflow.end_run()
experiment = mlflow.set_experiment(EXPERIMENT_NAME)

# Get training run_id from scoring table (the run that produced these champions)
champion_row = (
    spark.table(SCORING_TABLE)
    .filter(F.col("DATA_TABLE") == DATA_TABLE)
    .filter(F.col("is_champion") == True)
    .orderBy(F.desc("LOAD_TS"))
    .select("run_id")
    .first()
)

if champion_row is None or champion_row["run_id"] is None:
    print("No champion models found in scoring table. Skipping MLflow logging.")
    monitoring_run_id = "N/A"
    health_status = "PASS"
    total_alerts = len(backtest_errors) + len(staleness_alerts)
    if total_alerts > 0:
        health_status = "NEEDS_ATTENTION"
else:
    monitoring_run_id = champion_row["run_id"]
    print(f"Resuming training run: {monitoring_run_id}")

    # Verify the run exists in the active experiment before resuming.
    # The run may have been deleted or belong to a different experiment.
    try:
        mlflow.get_run(monitoring_run_id)
        resume_run = True
    except Exception:
        print(f"Run {monitoring_run_id} not found. Creating a new monitoring run.")
        resume_run = False

    run_kwargs = {"run_id": monitoring_run_id} if resume_run else {"run_name": "monitoring"}
    with mlflow.start_run(**run_kwargs) as run:
        if not resume_run:
            monitoring_run_id = run.info.run_id
        # Backtest metrics
        total_quarters_scored = (
            int(backtest_pdf["n_quarters_scored"].sum())
            if not backtest_pdf.empty
            else 0
        )
        mlflow.log_metric("monitoring_quarters_scored", total_quarters_scored)
        mlflow.log_metric("monitoring_backtest_alerts", len(backtest_errors))

        if not backtest_pdf.empty and "realized_mape" in backtest_pdf.columns:
            avg_realized = backtest_pdf["realized_mape"].mean()
            if is_valid_metric(avg_realized):
                mlflow.log_metric("monitoring_avg_realized_mape", avg_realized)

        # Staleness metrics
        mlflow.log_metric("monitoring_staleness_alerts", len(staleness_alerts))
        n_stale_partitions = int(staleness_pdf["is_stale"].sum()) if not staleness_pdf.empty else 0
        mlflow.log_metric("monitoring_stale_partitions", n_stale_partitions)
        if not staleness_pdf.empty:
            avg_age = staleness_pdf["model_age_days"].mean()
            if is_valid_metric(avg_age):
                mlflow.log_metric("monitoring_avg_model_age_days", avg_age)

        # Overall health
        total_alerts = len(backtest_errors) + len(staleness_alerts)
        health_status = "PASS" if total_alerts == 0 else "NEEDS_ATTENTION"
        mlflow.set_tag("monitoring_complete", "true")
        mlflow.set_tag("monitoring_status", health_status)

        # Run description (renders at top of MLflow run page)
        run_desc = (
            f"**Monitoring** | Status: {health_status} | "
            f"{total_alerts} total alerts\n\n"
            f"Backtest: {len(backtest_errors)} alerts | "
            f"Staleness: {n_stale_partitions} stale"
        )
        mlflow.set_tag("mlflow.note.content", run_desc)

        # Tables
        if not backtest_pdf.empty:
            mlflow.log_table(
                backtest_pdf, artifact_file="tables/monitoring_backtest.json"
            )
        if not horizon_backtest_pdf.empty:
            mlflow.log_table(
                horizon_backtest_pdf, artifact_file="tables/monitoring_per_horizon.json"
            )
        if not staleness_pdf.empty:
            staleness_log = staleness_pdf.drop(columns=["partition"], errors="ignore")
            mlflow.log_table(staleness_log, artifact_file="tables/monitoring_staleness.json")

        # Charts
        if staleness_fig is not None:
            mlflow.log_figure(staleness_fig, "charts/model_staleness.png")
        if health_heatmap_fig is not None:
            mlflow.log_figure(health_heatmap_fig, "charts/monitoring_health_heatmap.png")

        # Delta lineage
        tables_for_lineage = [
            (PREDICTIONS_TABLE, "monitoring_predictions_input"),
            (METRICS_TABLE, "monitoring_metrics_input"),
        ]
        if spark.catalog.tableExists(MONITORING_TABLE):
            tables_for_lineage.append((MONITORING_TABLE, "monitoring_output"))
        log_delta_lineage(tables_for_lineage)

print(f"MLflow logging complete (run: {monitoring_run_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Write to Delta Table
# MAGIC
# MAGIC Persist backtest and staleness results to the `_monitoring` Delta table.
# MAGIC Records are unified under a `check_type` column for downstream queries and dashboards.

# COMMAND ----------

monitoring_records = []
monitoring_timestamp = datetime.now()

# Backtest records: realized vs estimated MAPE per partition-version
for _, row in backtest_pdf.iterrows():
    monitoring_records.append(
        {
            "PARTITION_COLUMN": row["PARTITION_COLUMN"],
            "check_type": "backtest_actuals",
            "current_version": row["PREDICTIONS_VERSION"],
            "comparison_version": None,
            "metric_name": "realized_vs_estimated_mape",
            "current_value": row["realized_mape"],
            "comparison_value": row.get("estimated_mape"),
            "absolute_change": (
                row["realized_mape"] - row["estimated_mape"]
                if row.get("estimated_mape") and is_valid_metric(row["estimated_mape"])
                else None
            ),
            "relative_change_pct": row.get("mape_ratio"),
            "is_alert": row["is_alert"],
            "alert_reason": (
                f"realized MAPE {row['mape_ratio']:.1f}x estimated"
                if row["is_alert"]
                else None
            ),
            "LOAD_TS": monitoring_timestamp,
            "DATA_TABLE": DATA_TABLE,
            "data_version_scored_against": str(CURRENT_DATA_VERSION),
        }
    )

# Staleness records: model age vs threshold per partition
for result in staleness_results:
    monitoring_records.append(
        {
            "PARTITION_COLUMN": result["PARTITION_COLUMN"],
            "check_type": "model_staleness",
            "current_version": None,
            "comparison_version": None,
            "metric_name": "model_age_days",
            "current_value": float(result["model_age_days"]),
            "comparison_value": float(MAX_MODEL_AGE_DAYS),
            "absolute_change": float(result["model_age_days"] - MAX_MODEL_AGE_DAYS),
            "relative_change_pct": None,
            "is_alert": result["is_stale"],
            "alert_reason": f"model is {result['model_age_days']} days old" if result["is_stale"] else None,
            "LOAD_TS": monitoring_timestamp,
            "DATA_TABLE": DATA_TABLE,
        }
    )

# COMMAND ----------

if monitoring_records:
    monitoring_df = spark.createDataFrame(pd.DataFrame(monitoring_records))
    table_exists = spark.catalog.tableExists(MONITORING_TABLE)

    if table_exists:
        monitoring_df.write.format("delta").option("mergeSchema", "true").mode(
            "append"
        ).saveAsTable(MONITORING_TABLE)
    else:
        monitoring_df.write.format("delta").mode("overwrite").saveAsTable(
            MONITORING_TABLE
        )

    if ENABLE_ICEBERG:
        enable_iceberg(spark, MONITORING_TABLE)

    print(f"Wrote {len(monitoring_records)} records to {MONITORING_TABLE}")
else:
    print("No monitoring records to write (no backtest or staleness results)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Pipeline Error Logging
# MAGIC
# MAGIC Write backtest and staleness alerts to the `pipeline_errors` Delta table.

# COMMAND ----------

all_errors = backtest_errors + staleness_alerts

if all_errors:
    write_pipeline_errors(
        spark, CATALOG, SCHEMA, all_errors, "06_model_monitoring", monitoring_run_id
    )
    print(f"Logged {len(all_errors)} monitoring alerts to pipeline_errors table")
else:
    print("No monitoring alerts to log")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Monitoring Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics

# COMMAND ----------

# Build summary statistics table
n_backtest_total = len(backtest_results)
n_backtest_alerts = len(backtest_errors)
n_staleness_total = len(staleness_pdf) if not staleness_pdf.empty else 0
n_staleness_alerts = len(staleness_alerts)

summary_rows = []
for check, total, failed in [
    ("Backtest (realized vs estimated)", n_backtest_total, n_backtest_alerts),
    ("Model Staleness", n_staleness_total, n_staleness_alerts),
]:
    passed = total - failed
    rate = f"{failed / total * 100:.1f}%" if total > 0 else "N/A"
    summary_rows.append({
        "Check": check,
        "Total": total,
        "Passed": passed,
        "Failed": failed,
        "Alert Rate": rate,
    })

summary_table = pd.DataFrame(summary_rows)
display(summary_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Action Items

# COMMAND ----------

total_alerts = len(all_errors)

if total_alerts > 0:
    action_items = pd.DataFrame(all_errors)
    action_items["partition_short"] = action_items["partition"].apply(
        lambda p: truncate_partition_name(p, 30)
    )
    print(f"{total_alerts} action item(s) requiring attention:")
    display(
        action_items[["stage", "partition_short", "model_type", "severity", "message"]]
    )
else:
    print("All checks passed -- no action items.")

# COMMAND ----------

# Final status
alert_partitions = {err["partition"] for err in all_errors}

if total_alerts == 0:
    print("Monitoring status: PASS (0 alerts)")
else:
    print(
        f"Monitoring status: NEEDS ATTENTION "
        f"({total_alerts} alert(s) across {len(alert_partitions)} partition(s))"
    )

print(f"MLflow run: {monitoring_run_id}")
print(f"Output table: {MONITORING_TABLE}")

# COMMAND ----------

dbutils.notebook.exit(f"STATUS={health_status} ALERTS={total_alerts}")
