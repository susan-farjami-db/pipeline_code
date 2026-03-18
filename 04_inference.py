# Databricks notebook source
# MAGIC %md
# MAGIC # Inference Notebook
# MAGIC
# MAGIC Loads champion models and generates predictions for capacity forecasting.
# MAGIC
# MAGIC **model_source widget:**
# MAGIC - `mlflow` (default): Load champion models from MLflow registry via `@champion` alias
# MAGIC - `delta`: Load champion models from scoring Delta table via `is_champion=True`

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

import json
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

import mlflow

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Modules
# MAGIC
# MAGIC Pipeline utilities from `modeling_dev/`: data loading, MLflow helpers, modeling utils, metric calculations.

# COMMAND ----------

from modeling_dev.data_utils import (
    load_latest_version, enable_iceberg, write_pipeline_errors,
    truncate_partition_name,
    orchestrate_dq_checks, prepare_for_scoring,
    run_prediction_sanity_checks, print_sanity_check_report,
)
from modeling_dev.mlflow_utils import (
    setup_mlflow_client, load_champion_models_parallel,
    load_models_from_delta, log_delta_lineage,
    find_latest_training_run,
)
from modeling_dev.metric_utils import apply_evaluation_pipeline, calculate_horizon_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration
# MAGIC
# MAGIC Widget parameters control inference behavior (model source, data table, horizon range).
# MAGIC Set via the notebook UI or passed as job parameters from `databricks.yml`.

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("model_name_prefix", "forecasting")
dbutils.widgets.text("experiment_name", "/Shared/dab-pipeline-forecasting")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib")
dbutils.widgets.text("config_path", "configs/model_config.yaml")
dbutils.widgets.dropdown("model_source", "mlflow", ["mlflow", "delta"])
dbutils.widgets.text("scoring_table", "scoring_output")
dbutils.widgets.text("predictions_table", "")  # empty = derive from data_table

# Partition column configuration (comma-separated list of columns to partition by)
# Must match training notebook configuration
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO")

# Synthetic data toggle (enables MA imputation for masked/synthetic data)
dbutils.widgets.dropdown("is_synthetic_data", "true", ["true", "false"])
dbutils.widgets.dropdown("enable_iceberg", "true", ["true", "false"])

# Data quality check toggle (filter out poor quality partitions before inference)
dbutils.widgets.dropdown("data_quality_check", "true", ["true", "false"])

# Quality gate toggle: break pipeline if sanity checks flag partitions
dbutils.widgets.dropdown("break_on_sanity_failure", "false", ["true", "false"])

# Horizon range for filtering (comma-separated, e.g., "1,6" for Q+1 through Q+6)
dbutils.widgets.text("horizon_range", "1,6")

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME_PREFIX = dbutils.widgets.get("model_name_prefix")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
DATA_TABLE_NAME = dbutils.widgets.get("data_table")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{DATA_TABLE_NAME}"
CONFIG_PATH = dbutils.widgets.get("config_path")
MODEL_SOURCE = dbutils.widgets.get("model_source")
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('scoring_table')}"
PREDICTIONS_TABLE_PARAM = dbutils.widgets.get("predictions_table").strip()
if PREDICTIONS_TABLE_PARAM:
    PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.{PREDICTIONS_TABLE_PARAM}"
else:
    PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.{DATA_TABLE_NAME}_predictions"

# Partition columns (parsed from comma-separated widget)
PARTITION_COLS = [c.strip() for c in dbutils.widgets.get("partition_columns").split(",")]

IS_SYNTHETIC_DATA = dbutils.widgets.get("is_synthetic_data") == "true"
ENABLE_ICEBERG = dbutils.widgets.get("enable_iceberg") == "true"
DATA_QUALITY_CHECK = dbutils.widgets.get("data_quality_check") == "true"
BREAK_ON_SANITY_FAILURE = dbutils.widgets.get("break_on_sanity_failure") == "true"

# Parse horizon range (e.g., "1,6" -> max_horizon=6)
horizon_range_str = dbutils.widgets.get("horizon_range")
HORIZON_START, HORIZON_END = [int(x.strip()) for x in horizon_range_str.split(",")]
MAX_HORIZON = HORIZON_END

print(f"Model source: {MODEL_SOURCE}")
print(f"Data table: {DATA_TABLE}")
print(f"Predictions table: {PREDICTIONS_TABLE}")
print(f"Partitioning by: {PARTITION_COLS}")
print(f"Synthetic data imputation: {'enabled' if IS_SYNTHETIC_DATA else 'disabled'}")
print(f"Horizon range: Q+{HORIZON_START} through Q+{HORIZON_END}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
with open("configs/variable_config.yaml", "r") as f:
    variable_config = yaml.safe_load(f)

# Demand signals for indicator calculation (short names to match original AMD convention)
DEMAND_SIGNALS = config.get("DEMAND_SIGNALS", ["MOF", "MOF_PLUS_BUFFER", "SELLIN", "SR"])

# NeuralForecast model binaries are too large for Spark gRPC (~1.4GB vs 128MB limit).
# Mirror notebook 02: NF models infer on the driver, standard models via applyInPandas.
nf_model_types = set()
for m_name, m_cfg in config.get("models", {}).items():
    if m_cfg.get("type") == "neuralforecast":
        nf_model_types.add(m_name.lower())

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data
# MAGIC
# MAGIC Load the same horizon data used for training and create the `PARTITION_COLUMN`
# MAGIC (concatenation of configured partition columns, e.g. `SEGMENT_A_MOF_UPSIDE`).
# MAGIC Each unique partition gets matched to its champion model during inference.

# COMMAND ----------

SELECT_COLUMNS = list(variable_config["SELECT_COLUMNS"])
for col in PARTITION_COLS:
    if col not in SELECT_COLUMNS:
        SELECT_COLUMNS.append(col)

df_raw, DATA_VERSION = load_latest_version(spark, DATA_TABLE, SELECT_COLUMNS, is_synthetic_data=IS_SYNTHETIC_DATA)
print(f"Using data version: {DATA_VERSION}")

# Create PARTITION_COLUMN from configured partition columns
df_raw = df_raw.withColumn("PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLS]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks
# MAGIC
# MAGIC Filter out partitions with poor quality data before inference.

# COMMAND ----------

dq_errors_to_log = []
excluded_partitions = set()

if DATA_QUALITY_CHECK:
    df_raw, dq_errors_to_log, excluded_partitions = orchestrate_dq_checks(
        spark, df_raw, config, stage="data_quality_inference"
    )

# COMMAND ----------

partitions = [row["PARTITION_COLUMN"] for row in df_raw.select("PARTITION_COLUMN").distinct().collect()]
print(f"Found {len(partitions)} partitions for inference")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Champion Models
# MAGIC
# MAGIC Branches based on `model_source` widget:
# MAGIC - **mlflow**: Load models with `@champion` alias from MLflow registry
# MAGIC - **delta**: Load models with `is_champion=True` from scoring Delta table

# COMMAND ----------

client = setup_mlflow_client()
model_data = []

# Read training_run_id passed by NB02 via task values. Bypasses find_latest_training_run
# to prevent race conditions when two pipeline runs execute concurrently.
try:
    _job_training_run_id = dbutils.jobs.taskValues.get(
        taskKey="training", key="training_run_id", debugValue=None
    )
except Exception:
    _job_training_run_id = None
if _job_training_run_id:
    print(f"Using training_run_id from job task values: {_job_training_run_id}")

# COMMAND ----------

if MODEL_SOURCE == "mlflow":
    # --- MLflow path: load @champion models for partitions in latest training run ---
    print("Loading champion models from latest training run...")

    champions = load_champion_models_parallel(
        client, EXPERIMENT_NAME, SCORING_TABLE, CATALOG, SCHEMA, MODEL_NAME_PREFIX,
        parent_run_id=_job_training_run_id,
    )
    training_run_id, _ = find_latest_training_run(EXPERIMENT_NAME, SCORING_TABLE, _job_training_run_id)

    for m in champions:
        pyfunc = m["pyfunc_model"]
        entry = {
            "PARTITION_COLUMN": m["partition"],
            "_model_type": m["model_type"],
            "_model": pyfunc,
            "_model_config": pyfunc.model_config if pyfunc is not None else {},
            "_model_name": m["model_name"],
            "_version": m["version"],
            "_nested_run_id": m["nested_run_id"],
        }
        if m.get("artifact_uri"):
            entry["_artifact_uri"] = m["artifact_uri"]
        model_data.append(entry)

    if not model_data:
        dbutils.notebook.exit("No models found in latest training run")

# COMMAND ----------

if MODEL_SOURCE == 'delta':
    print(f"Loading champion models from {SCORING_TABLE}...")

    delta_models, _ = load_models_from_delta(
        spark, SCORING_TABLE,
        champions_only=True, data_table=DATA_TABLE,
        catalog=CATALOG, schema=SCHEMA, model_name_prefix=MODEL_NAME_PREFIX,
    )
    # Use job task value when available; fall back to scoring table lookup
    if _job_training_run_id:
        training_run_id = _job_training_run_id
    else:
        training_run_id = (
            spark.table(SCORING_TABLE)
            .orderBy(F.desc("LOAD_TS"))
            .select("run_id")
            .first()[0]
        )

    if not delta_models:
        dbutils.notebook.exit("No champion models found")

    for m in delta_models:
        is_nf = m["model_type"] in nf_model_types
        model_data.append({
            "PARTITION_COLUMN": m["PARTITION_COLUMN"],
            "_model_type": m["model_type"],
            "_model": m["model"] if is_nf else None,
            "_model_config": m["model_config"],
            "_model_name": m["model_name"],
            "_version": m["version"],
        })
    del delta_models

# COMMAND ----------

if not model_data:
    print("No champion models found. Run training and evaluation notebooks first.")
    dbutils.notebook.exit("No champion models found")

# COMMAND ----------

print(f"\nLoaded {len(model_data)} champion models via {MODEL_SOURCE}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Parallel Inference
# MAGIC
# MAGIC Models are stored in a dict and captured via closure when the function is sent to workers.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Model Lookup
# MAGIC
# MAGIC Create a dict mapping `partition` (MARKET_SEGMENT + SCENARIO) to model info.
# MAGIC
# MAGIC In Spark we use `applyInPandas` which runs a function on each group in parallel across workers. But we need to get the models to those workers somehow. Build a dict here, and reference it inside the function. When Spark serializes the function to workers, it automatically includes any variables the function references (this is called "closure capture"). Each worker gets one copy of the dict, not one copy per row.

# COMMAND ----------

model_lookup = {}     # standard models only — captured by applyInPandas closure in Phase 1
nf_model_lookup = {}  # NF models only — merged into model_lookup just before Phase 2
nf_partitions = set()
for m in model_data:
    partition = m["PARTITION_COLUMN"]
    entry = {
        "model": m["_model"],
        "config": m["_model_config"],
        "model_type": m["_model_type"],
        "version": m["_version"],
    }
    if m["_model_type"] in nf_model_types:
        nf_model_lookup[partition] = entry
        nf_partitions.add(partition)
    else:
        model_lookup[partition] = entry

print(f"Prepared {len(model_lookup)} standard + {len(nf_model_lookup)} NF champion models for inference")

# COMMAND ----------

# Output schema (required by applyInPandas - tells Spark what columns to expect back)
# Column names match original AMD naming convention from GENERATE_PREDICTIONS_ALL_QUARTER
prediction_schema = StructType([
    StructField("MARKET_SEGMENT", StringType()),
    StructField("SCENARIO", StringType()),
    StructField("QUARTER", StringType()),
    StructField("QUARTER_PARSED", TimestampType()),
    StructField("BILLING", DoubleType()),
    StructField("BILLING_PREDICTED", DoubleType()),
    StructField("VERIFICATION_HORIZON", DoubleType()),
    StructField("PREDICTION_HORIZON", DoubleType()),
    StructField("BB_VERSION", StringType()),
    StructField("MOF_VERSION", StringType()),
    StructField("BB_VERSION_LATEST", StringType()),
    StructField("MOF_VERSION_LATEST", StringType()),
    StructField("MOF", DoubleType()),
    StructField("MOF_PLUS_BUFFER", DoubleType()),
    StructField("SELLIN", DoubleType()),
    StructField("SR", DoubleType()),
    StructField("SR_MINUS3_MONTH", DoubleType()),
    StructField("SELLIN_MINUS3_MONTH", DoubleType()),
    StructField("MOF_PLUS_BUFFER_MINUS3_MONTH", DoubleType()),
    StructField("BILLING_MOVING_AVERAGE_2Q", DoubleType()),
    StructField("BILLING_MOVING_AVERAGE_3Q", DoubleType()),
    StructField("BILLING_MOVING_AVERAGE_4Q", DoubleType()),
    StructField("BILLING_MOVING_AVERAGE_5Q", DoubleType()),
    StructField("used_params", StringType()),
    StructField("MODEL_USED", StringType()),
    StructField("PARTITION_COLUMN", StringType()),
    StructField("APE", DoubleType()),
    StructField("MODEL_VERSION", StringType()),
])

# COMMAND ----------

def _run_inference_impl(pdf: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Core inference logic for one partition. Called by both applyInPandas (Phase 1)
    and driver loop (Phase 2), with different lookup dicts."""
    partition = pdf["PARTITION_COLUMN"].iloc[0]
    if partition not in lookup:
        print(f"WARNING: No champion model found for partition {partition}")
        return pd.DataFrame()

    info = lookup[partition]
    config, model_type = info["config"], info["model_type"]

    # Step 1: Resolve model (Delta binary takes priority, then closure/driver object)
    if "_model_binary" in pdf.columns and pdf["_model_binary"].iloc[0] is not None:
        import cloudpickle
        model = cloudpickle.loads(pdf["_model_binary"].iloc[0])
        pdf = pdf.drop(columns=["_model_binary"])
    elif info["model"] is not None:
        model = info["model"]
    else:
        print(f"WARNING: No model binary for partition {partition}")
        return pd.DataFrame()

    try:
        # Step 2: Prepare scoring data (rename columns, add demand signals)
        segment = pdf["MARKET_SEGMENT"].iloc[0]
        df = prepare_for_scoring(pdf, segment, variable_config)

        # Step 3: Generate predictions
        df['BILLING_PREDICTED'] = model.predict(None, df)

        # APE placeholder (computed in post-processing by apply_evaluation_pipeline)
        df['APE'] = np.nan

        # Step 4: Stamp metadata columns (model type, partition, segment)
        df['PARTITION_COLUMN'] = partition
        df['used_params'] = json.dumps(config)
        df['MODEL_USED'] = model_type.upper()
        df['MODEL_VERSION'] = str(info["version"])

        return df[[f.name for f in prediction_schema.fields]]
    except Exception as e:
        print(f"Error in {partition}: {e}")
        return pd.DataFrame()

_model_lookup_snapshot = dict(model_lookup)

def run_inference(pdf: pd.DataFrame) -> pd.DataFrame:
    """applyInPandas wrapper -- uses frozen snapshot so NF objects aren't serialized to workers."""
    return _run_inference_impl(pdf, _model_lookup_snapshot)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Parallel Inference
# MAGIC
# MAGIC `groupBy().applyInPandas()` is the parallelized Databricks equivalent of looping over partitions. Spark partitions the data by PARTITION_COLUMN (MARKET_SEGMENT + SCENARIO), sends each partition to a worker, and runs `run_inference` on each partition in parallel. The results are combined back into a single DataFrame.

# COMMAND ----------

print("Running inference...")

champion_partitions = list(model_lookup.keys())
non_nf_champion_partitions = [p for p in champion_partitions if p not in nf_partitions]

predictions_parts = []

# --- Phase 1: Standard models via applyInPandas ---
if non_nf_champion_partitions:
    print(f"  Standard models ({len(non_nf_champion_partitions)} partitions) via applyInPandas...")
    df_filtered = df_raw.filter(F.col("PARTITION_COLUMN").isin(non_nf_champion_partitions))

    # Delta path: join model_binary column (standard models only)
    if MODEL_SOURCE == "delta":
        from pyspark.sql.window import Window
        scoring_binary_df = (
            spark.table(SCORING_TABLE)
            .filter(F.col("DATA_TABLE") == DATA_TABLE)
            .filter(F.col("is_champion") == True)
            .filter(F.col("model_binary").isNotNull())
        )
        _w = Window.partitionBy("PARTITION_COLUMN").orderBy(F.desc("LOAD_TS"))
        scoring_binary_df = (
            scoring_binary_df
            .withColumn("_rank", F.row_number().over(_w))
            .filter(F.col("_rank") == 1)
            .drop("_rank")
            .select("PARTITION_COLUMN", F.col("model_binary").alias("_model_binary"))
        )
        if nf_partitions:
            scoring_binary_df = scoring_binary_df.filter(
                ~F.col("PARTITION_COLUMN").isin(list(nf_partitions))
            )
        df_filtered = df_filtered.join(scoring_binary_df, on="PARTITION_COLUMN", how="inner")

    std_predictions_df = (
        df_filtered
        .repartition(len(non_nf_champion_partitions), "PARTITION_COLUMN")
        .groupBy("PARTITION_COLUMN")
        .applyInPandas(run_inference, prediction_schema)
    )
    predictions_parts.append(std_predictions_df)

# --- Phase 2: NeuralForecast models on driver (too large for gRPC) ---
if nf_partitions:
    print(f"  NeuralForecast models ({len(nf_partitions)} partitions) on driver...")
    nf_pdf = df_raw.filter(
        F.col("PARTITION_COLUMN").isin(list(nf_partitions))
    ).toPandas()

    # Merge NF models into model_lookup now (after Phase 1 closure capture is done)
    model_lookup.update(nf_model_lookup)

    nf_results = []
    for partition in sorted(nf_partitions):
        group_df = nf_pdf[nf_pdf["PARTITION_COLUMN"] == partition].copy()
        result = _run_inference_impl(group_df, model_lookup)
        if not result.empty:
            nf_results.append(result)

    if nf_results:
        nf_combined = pd.concat(nf_results, ignore_index=True)
        predictions_parts.append(spark.createDataFrame(nf_combined, schema=prediction_schema))

    del nf_pdf

# Combine all predictions
if len(predictions_parts) == 1:
    predictions_df = predictions_parts[0]
elif len(predictions_parts) > 1:
    predictions_df = predictions_parts[0].unionByName(predictions_parts[1])
else:
    predictions_df = spark.createDataFrame([], schema=prediction_schema)

# Check for partitions that produced no predictions (inference failures)
expected_partitions = set(model_lookup.keys())
actual_partitions = set(
    row["PARTITION_COLUMN"]
    for row in predictions_df.select("PARTITION_COLUMN").distinct().collect()
)
missing_partitions = expected_partitions - actual_partitions

# Collect inference errors for pipeline_errors table (will be written at end)
inference_errors = []

if missing_partitions:
    print(f"\n{'='*60}")
    print(f"INFERENCE WARNINGS: {len(missing_partitions)} of {len(expected_partitions)} partitions produced no predictions")
    print(f"{'='*60}")
    for p in sorted(missing_partitions):
        print(f"  - {p}")
        model_info = model_lookup.get(p, {})
        inference_errors.append({
            "stage": "inference",
            "partition": p,
            "model_type": model_info.get("model_type", "unknown"),
            "severity": "error",
            "message": "Partition produced no predictions (inference failed)",
        })
    print(f"{'='*60}")
    print(f"Continuing with {len(actual_partitions)} successful partitions...")

print(f"Generated predictions for {len(actual_partitions)} partitions")

# COMMAND ----------

# MAGIC %md
# MAGIC # Post-Processing
# MAGIC
# MAGIC APE, demand indicators, null masking, row filtering via `applyInPandas` (distributed).

# COMMAND ----------

# Build eval schema: prediction columns + indicator columns per demand signal
eval_fields = list(prediction_schema.fields)
for signal in DEMAND_SIGNALS:
    affix = f"_{signal}_REFERENCED"
    eval_fields.extend([
        StructField(f"INDICATOR{affix}", StringType()),
        StructField(f"INDICATOR_FORECASTED{affix}", StringType()),
        StructField(f"VALIDATION_DISTANCE{affix}", DoubleType()),
        StructField(f"DIRECTIONAL_CORRECTNESS{affix}", DoubleType()),
    ])
eval_schema = StructType(eval_fields)

def eval_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """applyInPandas wrapper for apply_evaluation_pipeline."""
    result = apply_evaluation_pipeline(pdf, DEMAND_SIGNALS, MAX_HORIZON)
    # Ensure output matches eval_schema columns (add missing, drop extra)
    for field in eval_schema.fields:
        if field.name not in result.columns:
            result[field.name] = None
    return result[[f.name for f in eval_schema.fields]]

results_df = (
    predictions_df
    .groupBy("PARTITION_COLUMN")
    .applyInPandas(eval_udf, eval_schema)
)

results_in = results_df.toPandas()
print(f"Post-processing complete: {len(results_in)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction Sanity Checks
# MAGIC
# MAGIC Flag partitions with suspicious predictions: negative values, extreme APE (>500%),
# MAGIC flat predictions (<2 unique values), or poor directional correctness (<25%).

# COMMAND ----------

sanity_config = config.get("PREDICTION_SANITY_CHECKS", {})
results_in, sanity_report = run_prediction_sanity_checks(
    results_in,
    negative_check=sanity_config.get("NEGATIVE_PREDICTION_CHECK", True),
    extreme_ape_threshold=sanity_config.get("EXTREME_APE_THRESHOLD", 500),
    flat_check=sanity_config.get("FLAT_PREDICTION_CHECK", True),
    flat_min_unique=sanity_config.get("FLAT_MIN_UNIQUE_VALUES", 2),
    min_directional_correctness=sanity_config.get("MIN_DIRECTIONAL_CORRECTNESS", 0.25),
    max_validation_distance=sanity_config.get("MAX_VALIDATION_DISTANCE", 3.0),
)
print_sanity_check_report(sanity_report)
sanity_errors = sanity_report.get('errors_to_log', [])

if sanity_report['flagged_partitions'] and BREAK_ON_SANITY_FAILURE:
    raise ValueError(
        f"Sanity check failed: {len(sanity_report['flagged_partitions'])} partition(s) flagged. "
        f"Set break_on_sanity_failure=false to continue despite flagged partitions."
    )

# COMMAND ----------

# Exit if no predictions remain (all partitions failed or all rows filtered)
if results_in.empty:
    print("No predictions to save after post-processing.")
    # Write inference errors before exiting
    if inference_errors:
        write_pipeline_errors(spark, CATALOG, SCHEMA, inference_errors, "04_inference", "N/A")
    dbutils.notebook.exit("Inference produced no predictions. Check logs above.")

# COMMAND ----------

# MAGIC %md
# MAGIC # Add Versioning Columns
# MAGIC
# MAGIC Stamp each prediction row with version, timestamp, data lineage, and `is_latest=True`.
# MAGIC Downstream queries can filter on `is_latest` to get the most recent predictions.

# COMMAND ----------

inference_timestamp = datetime.now()

# MODEL_VERSION already set during inference - no need to re-add
results_in["PREDICTIONS_VERSION"] = f"V_{inference_timestamp.strftime('%Y%m%d_%H%M%S')}"
results_in["LOAD_TS"] = inference_timestamp
results_in["DATA_TABLE"] = DATA_TABLE
results_in["DATA_VERSION"] = DATA_VERSION
results_in["MODEL_SOURCE"] = MODEL_SOURCE
results_in["is_latest"] = True  # Will update previous rows to False after write

# COMMAND ----------

# MAGIC %md
# MAGIC # Write to Delta Table
# MAGIC
# MAGIC Marks all previous predictions as `is_latest=false`, then appends new predictions.
# MAGIC Creates the table on first run. Enables Iceberg compatibility if configured.

# COMMAND ----------

save_predictions = spark.createDataFrame(results_in)

# Check if table exists
table_exists = spark.catalog.tableExists(PREDICTIONS_TABLE)

if table_exists:
    # Maintain is_latest flag: mark all previous prediction rows as stale,
    # then append new predictions with is_latest=True.
    # This lets downstream queries filter to current predictions without version logic.
    existing_cols = {c.name for c in spark.table(PREDICTIONS_TABLE).schema}
    if "is_latest" not in existing_cols:
        spark.sql(f"ALTER TABLE {PREDICTIONS_TABLE} ADD COLUMNS (is_latest BOOLEAN)")

    spark.sql(f"UPDATE {PREDICTIONS_TABLE} SET is_latest = false WHERE is_latest = true")

    # Append new predictions with is_latest = True
    save_predictions.write.format("delta").option("mergeSchema", "true").mode("append").saveAsTable(PREDICTIONS_TABLE)
    print(f"Appended {len(results_in)} predictions to {PREDICTIONS_TABLE} (set previous as is_latest=false)")
else:
    # Create new table
    save_predictions.write.format("delta").mode("overwrite").saveAsTable(PREDICTIONS_TABLE)
    print(f"Created {PREDICTIONS_TABLE} with {len(results_in)} predictions")

if ENABLE_ICEBERG:
    enable_iceberg(spark, PREDICTIONS_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Partition Coverage Check
# MAGIC
# MAGIC Verify that this prediction run covers all partitions from the previous run.
# MAGIC Missing partitions may indicate training failures or data quality issues.

# COMMAND ----------

predicted_partitions = set(results_in["PARTITION_COLUMN"].unique())

if table_exists:
    previous_partitions = set(
        row["PARTITION_COLUMN"]
        for row in spark.table(PREDICTIONS_TABLE)
        .filter(F.col("is_latest") == False)
        .select("PARTITION_COLUMN")
        .distinct()
        .collect()
    )
    missing_partitions_coverage = previous_partitions - predicted_partitions
    if missing_partitions_coverage:
        print(f"WARNING: {len(missing_partitions_coverage)} partition(s) predicted last run but not this run:")
        for p in sorted(missing_partitions_coverage):
            print(f"  - {p}")
            inference_errors.append({
                "stage": "inference_coverage",
                "partition": p,
                "model_type": "unknown",
                "severity": "warning",
                "message": "Partition predicted in previous run but missing from current run",
            })
    else:
        print(f"Coverage OK: all {len(previous_partitions)} previously predicted partitions are covered")
else:
    print("First prediction run -- no coverage comparison available")

# COMMAND ----------

# MAGIC %md
# MAGIC # Per-Horizon MAPE Summary
# MAGIC
# MAGIC Compute HORIZON_MAPE per partition using `calculate_horizon_metrics` (same function
# MAGIC used by notebook 03). This gives per-horizon Q+1 through Q+6 MAPE values and an
# MAGIC overall HORIZON_MAPE that is directly comparable to the evaluation results.

# COMMAND ----------

horizon_rows = []
for partition in results_in["PARTITION_COLUMN"].unique():
    part_df = results_in[results_in["PARTITION_COLUMN"] == partition]
    segment = part_df["MARKET_SEGMENT"].iloc[0]
    model_used = part_df["MODEL_USED"].iloc[0]

    horizon_metrics = calculate_horizon_metrics(
        part_df, segment, demand_signals=DEMAND_SIGNALS, max_horizon=MAX_HORIZON
    )

    row = {"PARTITION_COLUMN": partition, "MODEL_USED": model_used}
    for h in range(1, 7):  # fixed Q1-Q6 columns; NaN for horizons beyond HORIZON_END
        row[f"Q{h}_MAPE"] = horizon_metrics.get(f"Q+{h} MAPE", np.nan)

    q_vals = [row[f"Q{h}_MAPE"] for h in range(HORIZON_START, HORIZON_END + 1)
              if pd.notna(row.get(f"Q{h}_MAPE"))]
    row["HORIZON_MAPE"] = np.mean(q_vals) if q_vals else np.nan
    row["prediction_count"] = int(part_df["APE"].notna().sum())
    horizon_rows.append(row)

partition_summary = pd.DataFrame(horizon_rows).sort_values("PARTITION_COLUMN")

# COMMAND ----------

# MAGIC %md
# MAGIC # Log to MLflow
# MAGIC
# MAGIC Resumes the shared training run (same parent run as notebooks 02, 03, and 05) and logs
# MAGIC inference params, metrics, artifacts, and Delta table lineage for traceability.
# MAGIC Params and metrics are prefixed with `inference_` to avoid conflicts with earlier stages.

# COMMAND ----------

mlflow.end_run()
print(f"Resuming training run: {training_run_id}")

with mlflow.start_run(run_id=training_run_id) as inference_run:
    inference_run_id = inference_run.info.run_id

    # Log data quality errors to pipeline_errors table (if any)
    if dq_errors_to_log:
        write_pipeline_errors(spark, CATALOG, SCHEMA, dq_errors_to_log, "04_inference", inference_run_id)
        print(f"Logged {len(dq_errors_to_log)} data quality errors to pipeline_errors table")
        mlflow.log_metric("inference_partitions_excluded_dq", len(dq_errors_to_log))

    # Log params (prefixed to avoid conflicts with training/eval params)
    mlflow.log_params({
        "inference_model_source": MODEL_SOURCE,
        "inference_predictions_version": results_in['PREDICTIONS_VERSION'].iloc[0],
        "inference_predictions_table": PREDICTIONS_TABLE,
    })

    # Log metrics (prefixed)
    mlflow.log_metric("inference_total_predictions", len(results_in))
    mlflow.log_metric("inference_partitions_predicted", results_in["PARTITION_COLUMN"].nunique())
    mlflow.log_metric("inference_partitions_failed", len(inference_errors))
    inference_mape = results_in["APE"].mean()
    if not np.isnan(inference_mape):
        mlflow.log_metric("inference_mape", inference_mape)
    inference_horizon_mape = partition_summary["HORIZON_MAPE"].mean()
    if not np.isnan(inference_horizon_mape):
        mlflow.log_metric("inference_horizon_mape", inference_horizon_mape)
    if inference_errors:
        mlflow.set_tag("has_inference_errors", "true")

    # Log sanity check results
    if sanity_errors:
        write_pipeline_errors(spark, CATALOG, SCHEMA, sanity_errors, "04_inference", inference_run_id)
        mlflow.log_metric("inference_partitions_flagged_sanity", len(sanity_report['flagged_partitions']))
        mlflow.log_metric("inference_negative_prediction_rows", sanity_report['summary']['total_negative_rows'])

    # Log artifacts
    mlflow.log_artifact(CONFIG_PATH, "config")

    # Segment summary (partition-level with per-horizon MAPE)
    mlflow.log_table(partition_summary, artifact_file="tables/inference_partition_summary.json")

    # Predictions sample (first 100 rows per segment)
    # Sample up to 100 rows per segment (not 100 total) for MLflow artifact logging
    preds_sample = results_in.groupby("MARKET_SEGMENT").head(100)
    mlflow.log_table(preds_sample, artifact_file="tables/inference_predictions_sample.json")

    mlflow.set_tag("predictions_table", PREDICTIONS_TABLE)
    mlflow.set_tag("inference_complete", "true")

    # Run description (renders at top of MLflow run page)
    n_partitions = results_in["PARTITION_COLUMN"].nunique()
    run_desc = (
        f"**Inference** | {n_partitions} partitions predicted, "
        f"{len(results_in)} total rows\n\n"
        f"Failures: {len(inference_errors)} | "
        f"Model source: {MODEL_SOURCE}\n\n"
        f"Output: `{PREDICTIONS_TABLE}`"
    )
    mlflow.set_tag("mlflow.note.content", run_desc)

    # Log Delta tables as MLflow datasets (lineage tracking)
    log_delta_lineage([
        (DATA_TABLE, "inference_data"),
        (SCORING_TABLE, "champion_models"),
        (PREDICTIONS_TABLE, "predictions_output"),
    ])

print(f"Inference run ID: {inference_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Inference Summary

# COMMAND ----------

print(f"Predictions version: {results_in['PREDICTIONS_VERSION'].iloc[0]}")
print(f"Data version: {DATA_VERSION}")
print(f"Model source: {MODEL_SOURCE}")
print(f"Output table: {PREDICTIONS_TABLE}")
print(f"Total predictions: {len(results_in)}")

# COMMAND ----------

# Display partition summary (computed earlier, before MLflow logging)
display_cols = ["PARTITION_COLUMN", "MODEL_USED", "HORIZON_MAPE",
                "Q1_MAPE", "Q2_MAPE", "Q3_MAPE", "Q4_MAPE", "Q5_MAPE", "Q6_MAPE",
                "prediction_count"]
display_df = partition_summary[[c for c in display_cols if c in partition_summary.columns]].copy()
for col in display_df.select_dtypes(include=[np.number]).columns:
    display_df[col] = display_df[col].round(2)

print("Partition-level summary:")
display(display_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Error Logging
# MAGIC
# MAGIC Write inference failures to the `pipeline_errors` Delta table for alerting and diagnostics dashboards.

# COMMAND ----------

# Write inference errors to pipeline_errors table
if inference_errors:
    write_pipeline_errors(spark, CATALOG, SCHEMA, inference_errors, "04_inference", inference_run_id)

