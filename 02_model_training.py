# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training Notebook
# MAGIC
# MAGIC Trains Prophet and XGBoost models for capacity forecasting, one model per partition (MARKET_SEGMENT + SCENARIO).
# MAGIC Each model is logged to MLflow with partition-specific tags and registered to Unity Catalog.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os

notebook_path = os.getcwd()
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Libraries

# COMMAND ----------

import gc
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import cloudpickle

import mlflow
from mlflow.pyfunc import PythonModel
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, BinaryType, DoubleType,
    TimestampType, BooleanType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Modules
# MAGIC
# MAGIC Pipeline utilities from `modeling_dev/`: data loading, MLflow helpers, model logging.

# COMMAND ----------

from modeling_dev.data_utils import (
    load_latest_version, enable_iceberg, write_pipeline_errors,
    truncate_partition_name, is_valid_metric,
    orchestrate_dq_checks,
)
from modeling_dev.mlflow_utils import (
    get_or_create_experiment, log_delta_lineage, log_models_to_mlflow_parallel,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Functions
# MAGIC
# MAGIC Loads train_model() dispatcher from 01_training_functions notebook.
# MAGIC Edit that notebook to customize model training behavior.

# COMMAND ----------

# MAGIC %run ./01_training_functions

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration
# MAGIC
# MAGIC Widget parameters control pipeline behavior (catalog, schema, model types, toggles).
# MAGIC These are set via the notebook UI or passed in as job parameters from `databricks.yml`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widget Parameters

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("model_name_prefix", "forecasting")
dbutils.widgets.text("experiment_name", "/Shared/dab-pipeline-forecasting")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib")
dbutils.widgets.text("config_path", "configs/model_config.yaml")

# Delta table output (MMF pattern - for future brand-level scaling)
dbutils.widgets.text("scoring_table", "scoring_output")

# MLflow logging toggle (set to true to log models to nested MLflow runs)
dbutils.widgets.dropdown("log_to_mlflow", "true", ["true", "false"])

# MLflow Model Registry toggle (only applies if log_to_mlflow=true)
# Set to false to log models without registering to Model Registry (recommended for 50+ models)
dbutils.widgets.dropdown("register_to_registry", "true", ["true", "false"])

# Partition column configuration (comma-separated list of columns to partition by)
# Default: MARKET_SEGMENT,SCENARIO (matches AMD approach - one model per segment+scenario)
# Alternative: MARKET_SEGMENT (simpler - one model per segment)
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO")

# Synthetic data toggle (enables MA imputation for masked/synthetic data)
dbutils.widgets.dropdown("is_synthetic_data", "true", ["true", "false"])

# Data quality check toggle (MMF-style pre-training validation)
dbutils.widgets.dropdown("data_quality_check", "true", ["true", "false"])
dbutils.widgets.dropdown("enable_iceberg", "true", ["true", "false"])

# Segment filter (optional - leave empty to train all segments)
dbutils.widgets.text("segment_filter", "")

# Run name prefix (customizable for job/experiment identification)
dbutils.widgets.text("run_name_prefix", "pipeline")

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME_PREFIX = dbutils.widgets.get("model_name_prefix")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}"
CONFIG_PATH = dbutils.widgets.get("config_path")

# Delta table name (MMF pattern)
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('scoring_table')}"

# MLflow toggles
LOG_TO_MLFLOW = dbutils.widgets.get("log_to_mlflow") == "true"
REGISTER_TO_REGISTRY = dbutils.widgets.get("register_to_registry") == "true"

# Partition columns (parsed from comma-separated widget)
PARTITION_COLS = [c.strip() for c in dbutils.widgets.get("partition_columns").split(",")]
print(f"Partitioning by: {PARTITION_COLS}")

# Synthetic data toggle
IS_SYNTHETIC_DATA = dbutils.widgets.get("is_synthetic_data") == "true"
print(f"Synthetic data imputation: {'enabled' if IS_SYNTHETIC_DATA else 'disabled'}")

# Data quality check toggle
DATA_QUALITY_CHECK = dbutils.widgets.get("data_quality_check") == "true"
ENABLE_ICEBERG = dbutils.widgets.get("enable_iceberg") == "true"
print(f"Data quality checks: {'enabled' if DATA_QUALITY_CHECK else 'disabled'}")

# Segment filter (optional)
SEGMENT_FILTER = dbutils.widgets.get("segment_filter").strip()
if SEGMENT_FILTER:
    print(f"Segment filter: {SEGMENT_FILTER}")

RUN_NAME_PREFIX = dbutils.widgets.get("run_name_prefix")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
with open("configs/variable_config.yaml", "r") as f:
    variable_config = yaml.safe_load(f)

# Model types to train (from YAML general.models)
MODEL_TYPES = config.get("general", {}).get("models", ["prophet", "xgboost"])
print(f"Model types (from YAML): {MODEL_TYPES}")

# Horizon range from config (for MLflow logging and consistency with eval/inference)
eval_config = config.get("EVALUATION", {})
HORIZON_RANGE = eval_config.get("horizon_range", [1, 6])
HORIZON_RANGE_STR = f"{HORIZON_RANGE[0]}-{HORIZON_RANGE[1]}"
print(f"Horizon range (from YAML): Q+{HORIZON_RANGE[0]} through Q+{HORIZON_RANGE[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Set MLflow Experiment
# MAGIC
# MAGIC Creates or reuses the MLflow experiment for this pipeline run.
# MAGIC All training runs (parent + nested per-model runs) are logged under this experiment.

# COMMAND ----------

EXPERIMENT_ID = get_or_create_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC # PyFunc Wrapper
# MAGIC
# MAGIC Thin wrapper that lets us log both Prophet and XGBoost to MLflow with a consistent interface.
# MAGIC The wrapper just delegates to model.predict() - caller is responsible for data prep.

# COMMAND ----------

class ForecastingModel(PythonModel):
    """MLflow PyFunc wrapper that provides a unified predict() interface for all model types.
    Delegates to self.model.predict() -- each model is a predictor wrapper with its own logic."""

    def __init__(self, model=None, model_config: dict = None):
        self.model = model
        self.model_config = model_config or {}

    def predict(self, _context, model_input: pd.DataFrame) -> np.ndarray:
        return self.model.predict(model_input)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data
# MAGIC
# MAGIC Load the latest version of the horizon data table and create the `PARTITION_COLUMN`
# MAGIC (concatenation of configured partition columns, e.g. `SEGMENT_A_MOF_UPSIDE`).
# MAGIC Each unique partition value gets its own model during parallel training.

# COMMAND ----------

SELECT_COLUMNS = list(variable_config["SELECT_COLUMNS"])
# Ensure partition columns are in the query even if not listed in variable_config
for col in PARTITION_COLS:
    if col not in SELECT_COLUMNS:
        SELECT_COLUMNS.append(col)

df_raw, DATA_VERSION = load_latest_version(spark, DATA_TABLE, SELECT_COLUMNS, is_synthetic_data=IS_SYNTHETIC_DATA)
print(f"Using data version: {DATA_VERSION}")

# Create PARTITION_COLUMN from configured partition columns
df_raw = df_raw.withColumn("PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLS]))

# Apply segment filter if specified
if SEGMENT_FILTER:
    pre_filter_count = df_raw.select("PARTITION_COLUMN").distinct().count()
    df_raw = df_raw.filter(F.col("MARKET_SEGMENT") == SEGMENT_FILTER)
    post_filter_count = df_raw.select("PARTITION_COLUMN").distinct().count()
    print(f"Segment filter applied: {SEGMENT_FILTER}")
    print(f"  Partitions: {pre_filter_count} -> {post_filter_count}")

# COMMAND ----------

# Warn if >10% nulls in critical columns (likely indicates data pipeline issue)
critical_cols = ["BB", "PARTITION_COLUMN", "QUARTER_PARSED"]
total_rows = df_raw.count()
for col in critical_cols:
    null_count = df_raw.filter(F.col(col).isNull()).count()
    null_rate = null_count / total_rows if total_rows > 0 else 0
    if null_rate > 0.1:
        print(f"WARNING: {col} has {null_rate:.1%} null values ({null_count}/{total_rows})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks (MMF-style)
# MAGIC
# MAGIC Validates data before training: nulls, negatives, insufficient periods.

# COMMAND ----------

dq_errors_to_log = []  # Store data quality errors for logging after MLflow run starts

if DATA_QUALITY_CHECK:
    df_raw, dq_errors_to_log, excluded_partitions = orchestrate_dq_checks(
        spark, df_raw, config, stage="data_quality_training"
    )

# COMMAND ----------

segments = [row["MARKET_SEGMENT"] for row in df_raw.select("MARKET_SEGMENT").distinct().collect()]
n_partitions = df_raw.select("PARTITION_COLUMN").distinct().count()
print(f"Found {len(segments)} segments, {n_partitions} partitions")

# COMMAND ----------

df_raw.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parallel Training with applyInPandas
# MAGIC
# MAGIC Training runs in parallel across Spark workers using `groupBy().applyInPandas()`.
# MAGIC Each partition (MARKET_SEGMENT + SCENARIO) trains on a separate worker.
# MAGIC Models are serialized and returned as a DataFrame, then registered to MLflow on the driver.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Schemas
# MAGIC
# MAGIC Output schemas for the parallel training function and the scoring Delta table.
# MAGIC
# MAGIC We have these separate because `train_parallel` will produce everything in training_result_schema, after which we add mlflow and training specific info on top.

# COMMAND ----------

# Closure capture for config (avoids duplicating JSON per row)
training_config = {"config": config}

_base_fields = [
    StructField("PARTITION_COLUMN", StringType(), True),
    StructField("MODEL_USED", StringType(), True),
    StructField("model_binary", BinaryType(), True),
    StructField("model_config", StringType(), True),
    StructField("mape", DoubleType(), True),
    StructField("sample_input", StringType(), True),
    StructField("training_metrics", StringType(), True),
    StructField("used_params", StringType(), True),
    StructField("error", StringType(), True),
]

training_result_schema = StructType(
    _base_fields + [StructField("predictions_sample", StringType(), True)]
)

scoring_schema = StructType(_base_fields + [
    StructField("LOAD_TS", TimestampType(), True),
    StructField("run_id", StringType(), True),
    StructField("nested_run_id", StringType(), True),
    StructField("DATA_TABLE", StringType(), True),
    StructField("DATA_VERSION", StringType(), True),
    StructField("is_champion", BooleanType(), True),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallel Training Function
# MAGIC
# MAGIC This function runs on each Spark worker for one partition. For each partition it:
# MAGIC 1. Reads model type from the `_model_type` column and loads segment-specific config from YAML
# MAGIC 2. Dispatches to `train_model()` (defined in `01_training_functions`)
# MAGIC 3. Serializes the trained model with cloudpickle
# MAGIC 4. Collects training metrics (MAPE, grid search results, feature selection)
# MAGIC 5. Builds a sample input for MLflow signature logging
# MAGIC 6. Returns everything as a single-row DataFrame matching `training_result_schema`
# MAGIC
# MAGIC The config is passed via closure capture (`training_config` dict) so each worker gets one copy.

# COMMAND ----------

def train_parallel(pdf: pd.DataFrame) -> pd.DataFrame:
    """applyInPandas function that trains one model per partition. Reads _model_type column to
    dispatch to train_prophet/train_xgboost/etc., serializes the model with cloudpickle, and
    returns metrics + binary in a single-row DataFrame matching training_result_schema."""
    cfg = training_config["config"]
    # _model_type is a marker column injected before applyInPandas (Spark can't pass extra args to UDFs)
    model_type = pdf["_model_type"].iloc[0]
    pdf = pdf.drop(columns=["_model_type"])
    partition = pdf["PARTITION_COLUMN"].iloc[0] if "PARTITION_COLUMN" in pdf.columns else None
    segment = pdf["MARKET_SEGMENT"].iloc[0]

    # Step 1: Load segment-specific config (regressors, grid search settings)
    seg_config = load_segment_config(cfg, segment)

    # Step 2: Skip if this model type isn't configured for this segment
    seg_training_config = get_segment_training_config(cfg, segment)
    if model_type.lower() not in seg_training_config["models"]:
        return pd.DataFrame([_error_result(partition, model_type, f"SKIPPED: {model_type} not configured for segment {segment}")])

    try:
        # Step 3: Train the model (dispatches to train_prophet/train_xgboost/etc.)
        model, preds, metrics, params = train_model(model_type, pdf.copy(), segment, seg_config, cfg)

        # ---- Package model output for MLflow + Delta ----

        # Step 4: Prepare MLflow metadata (sample input for signature, model config dict)
        pyfunc_config = model.pyfunc_config
        sample_input_df = model.sample_input(preds)
        sample_json = sample_input_df.to_json(date_format="iso")
        model_config = {"model_type": model_type.upper(), "segment": segment, "partition": partition, **pyfunc_config}

        # Step 5: Convert metrics/params to JSON-safe types (numpy types break json.dumps)
        metrics_serializable = {k: (v if not isinstance(v, float) or not np.isnan(v) else None) for k, v in metrics.items()}
        params_serializable = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in params.items()}

        # Sample of predictions for MLflow artifact (first 20 rows)
        preds_sample_cols = ["QUARTER_PARSED", "BILLING", "BILLING_PREDICTED", "MARKET_SEGMENT", "QUARTER"]
        preds_sample_cols = [c for c in preds_sample_cols if c in preds.columns]
        predictions_sample_json = preds[preds_sample_cols].head(20).to_json(date_format="iso")

        # Step 6: Wrap in ForecastingModel PyFunc and serialize with cloudpickle
        pyfunc_model = ForecastingModel(model=model, model_config=model_config)

        # Step 7: Return single-row DataFrame with model binary + all metadata
        return pd.DataFrame([{
            "PARTITION_COLUMN": partition,
            "MODEL_USED": model_type.upper(),
            "model_binary": cloudpickle.dumps(pyfunc_model),
            "model_config": json.dumps(model_config),
            "mape": float(metrics.get("mape", np.nan)),
            "sample_input": sample_json,
            "predictions_sample": predictions_sample_json,
            "training_metrics": json.dumps(metrics_serializable, default=str),
            "used_params": json.dumps(params_serializable, default=str),
            "error": None,
        }])

    except Exception as e:
        return pd.DataFrame([_error_result(partition, model_type, str(e))])

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Training
# MAGIC
# MAGIC Standard models (Prophet, XGBoost, etc.) train via a single `applyInPandas` pass across
# MAGIC all (partition x model_type) groups. NeuralForecast models train as global models on the driver.

# COMMAND ----------

train_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

cfg = training_config["config"]
nf_model_types = [m for m in MODEL_TYPES if cfg.get("models", {}).get(m.lower(), {}).get("type") == "neuralforecast"]
non_nf_model_types = [m for m in MODEL_TYPES if m not in nf_model_types]

if non_nf_model_types:
    print(f"Standard models (parallel on workers): {non_nf_model_types}")
if nf_model_types:
    print(f"NeuralForecast models (global on driver): {nf_model_types}")
print(f"Training {n_partitions} partitions...")

# --- Phase 1: Standard models via applyInPandas ---
# Cross-join partitions with model types, then group by (PARTITION_COLUMN, _model_type)
# so all partition x model combinations run in a single applyInPandas pass.
if non_nf_model_types:
    model_types_df = spark.createDataFrame(
        [(m,) for m in non_nf_model_types], ["_model_type"]
    )
    n_groups = n_partitions * len(non_nf_model_types)
    print(f"  Training {n_groups} groups ({n_partitions} partitions x {len(non_nf_model_types)} model types)...")

    trained_models_df = (
        df_raw
        .crossJoin(model_types_df)
        .repartition(n_groups, "PARTITION_COLUMN", "_model_type")
        .groupBy("PARTITION_COLUMN", "_model_type")
        .applyInPandas(train_parallel, training_result_schema)
    )
    trained_models = [row.asDict() for row in trained_models_df.collect()]
else:
    trained_models = []

# --- Phase 2: NeuralForecast models via global training on driver ---
if nf_model_types:
    print(f"  Collecting data to driver for NeuralForecast global training...")
    all_partitions_pdf = df_raw.toPandas()

    for model_type in nf_model_types:
        print(f"  Training {model_type} (global)...")
        nf_results = train_neuralforecast_global(all_partitions_pdf, model_type, cfg)
        trained_models.extend(nf_results)
        del nf_results
        gc.collect()

    del all_partitions_pdf
    gc.collect()

    # Lightning creates lightning_logs/ during NF training; clean up to avoid
    # accumulating checkpoints across runs on the same cluster.
    import shutil
    if os.path.isdir("lightning_logs"):
        shutil.rmtree("lightning_logs", ignore_errors=True)
        print("  Cleaned up lightning_logs/")

print(f"Training complete. Collected {len(trained_models)} model results.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Errors
# MAGIC
# MAGIC Report any partitions where model training failed. These are logged to the
# MAGIC `pipeline_errors` Delta table at the end of the notebook for alerting.

# COMMAND ----------

training_errors = [row for row in trained_models if row["error"]]
if training_errors:
    print(f"\n{'='*60}")
    print(f"TRAINING ERRORS: {len(training_errors)} of {len(trained_models)} models failed")
    print(f"{'='*60}")
    for row in training_errors:
        print(f"  [{row['PARTITION_COLUMN']}] {row['MODEL_USED']}: {row['error']}")
    print(f"{'='*60}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Gate
# MAGIC
# MAGIC Verify every partition has at least one successfully trained model.

# COMMAND ----------

expected_partitions = set(df_raw.select("PARTITION_COLUMN").distinct().toPandas()["PARTITION_COLUMN"])
successful_partitions = set(
    row["PARTITION_COLUMN"] for row in trained_models
    if row["model_binary"] is not None and not row["error"]
)
missing_partitions = expected_partitions - successful_partitions

if missing_partitions:
    # Separate real failures from expected skips (partitions with too few rows to train)
    insufficient_data = set()
    for row in trained_models:
        if row["PARTITION_COLUMN"] in missing_partitions and row["error"]:
            if "Insufficient training data" in row["error"]:
                insufficient_data.add(row["PARTITION_COLUMN"])
    real_failures = missing_partitions - insufficient_data

    if insufficient_data:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(insufficient_data)} partition(s) skipped (insufficient training data)")
        print(f"{'='*60}")
        for p in sorted(insufficient_data):
            print(f"  - {p}")
        print(f"{'='*60}\n")

    if real_failures:
        print(f"\n{'='*60}")
        print(f"QUALITY GATE FAILED: {len(real_failures)} partition(s) have no successfully trained models")
        print(f"{'='*60}")
        for p in sorted(real_failures):
            print(f"  - {p}")
        print(f"{'='*60}")
        raise ValueError(
            f"Quality gate failed: {len(real_failures)} partition(s) have no successfully trained models: "
            f"{sorted(real_failures)[:5]}{'...' if len(real_failures) > 5 else ''}"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC # MLflow Logging

# COMMAND ----------

mlflow.end_run() # in case job auto created a run

with mlflow.start_run(
    run_name=f"{RUN_NAME_PREFIX}_{train_timestamp}",
    experiment_id=EXPERIMENT_ID
) as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # --- Nested MLflow runs (if enabled) ---
    results = []
    nested_run_ids = {}

    if LOG_TO_MLFLOW:
        artifact_msg = " + artifacts"
        registry_msg = " + Model Registry" if REGISTER_TO_REGISTRY else ""
        print(f"Logging {len(trained_models)} models to MLflow (params/metrics{artifact_msg}{registry_msg})...\n")

        results, nested_run_ids = log_models_to_mlflow_parallel(
            trained_models,
            MODEL_NAME_PREFIX,
            CATALOG,
            SCHEMA,
            DATA_TABLE,
            DATA_VERSION,
            parent_run_id=run_id,
            experiment_id=EXPERIMENT_ID,
            horizon_range_str=HORIZON_RANGE_STR,
            experiment_name=EXPERIMENT_NAME,
            log_artifacts=True,
            register_to_registry=REGISTER_TO_REGISTRY,
        )

        print(f"\n{len(results)} models logged to MLflow nested runs.")
    else:
        print(f"Skipping nested MLflow runs (log_to_mlflow=false). Models stored in Delta only.")

    # --- Summary metrics ---
    mlflow.log_metric("models_trained", len(trained_models))
    mlflow.log_metric("models_logged", len(results))
    mlflow.log_metric("training_errors", len(training_errors))
    mlflow.set_tag("scoring_table", SCORING_TABLE)
    if REGISTER_TO_REGISTRY:
        mlflow.log_metric("models_registered", len([r for r in results if r.get("registered_model_name")]))

    # Run description (renders at top of MLflow run page)
    n_registered = len([r for r in results if r.get("registered_model_name")])
    run_desc = (
        f"**Training** | {len(trained_models)} models trained, "
        f"{len(results)} logged, {len(training_errors)} errors\n\n"
        f"Data: `{DATA_TABLE}` v{DATA_VERSION}\n\n"
        f"Scoring table: `{SCORING_TABLE}`\n\n"
        f"Registered: {n_registered}"
    )
    mlflow.set_tag("mlflow.note.content", run_desc)

    # --- Artifacts ---
    mlflow.log_artifact(CONFIG_PATH, "config")
    mlflow.log_dict(config, "config/model_config.json")

    if results:
        summary_records = [{
            "partition": r["partition"],
            "model": r["MODEL_USED"],
            "mape": r.get("mape"),
            "nested_run_id": r.get("nested_run_id"),
            "registered_model_name": r.get("registered_model_name"),
            "model_version": r.get("model_version"),
        } for r in results]
        summary_df_artifact = pd.DataFrame(summary_records)
        mlflow.log_table(summary_df_artifact, artifact_file="summary/training_summary.json")

    # --- Delta lineage ---
    log_delta_lineage([
        (DATA_TABLE, "training_data"),
        (SCORING_TABLE, "scoring_output"),
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC # Delta Table Writes
# MAGIC
# MAGIC Write model binaries and pipeline errors to Delta. Uses `run_id` and `nested_run_ids`
# MAGIC from the MLflow block above for lineage linkage.

# COMMAND ----------

# Data quality errors
if dq_errors_to_log:
    write_pipeline_errors(spark, CATALOG, SCHEMA, dq_errors_to_log, "02_model_training", run_id)
    print(f"Logged {len(dq_errors_to_log)} data quality errors to pipeline_errors table")

# Build scoring records and write to Delta in batches to limit peak memory.
# NF model results carry unserialized model_binary (raw pyfunc objects sharing a
# single copy of the global model bytes). Serialize just-in-time per batch, write,
# then free.  Standard model results already have pre-serialized bytes.
# is_champion=False here -- notebook 03 sets the actual champion after evaluation.
_SCORING_BATCH_SIZE = 100
load_ts = datetime.now()
total_written = 0

for batch_start in range(0, len(trained_models), _SCORING_BATCH_SIZE):
    batch_end = min(batch_start + _SCORING_BATCH_SIZE, len(trained_models))
    batch_records = []

    for i in range(batch_start, batch_end):
        row = trained_models[i]
        binary = row["model_binary"]
        if binary is not None and not isinstance(binary, (bytes, bytearray)):
            binary = cloudpickle.dumps(binary)
        batch_records.append({
            "PARTITION_COLUMN": row["PARTITION_COLUMN"],
            "MODEL_USED": row["MODEL_USED"],
            "model_binary": binary,
            "model_config": row["model_config"],
            "mape": row["mape"],
            "sample_input": row["sample_input"],
            "training_metrics": row["training_metrics"],
            "used_params": row["used_params"],
            "error": str(row["error"]) if row["error"] else None,
            "LOAD_TS": load_ts,
            "run_id": run_id,
            "nested_run_id": nested_run_ids.get(i),
            "DATA_TABLE": DATA_TABLE,
            "DATA_VERSION": DATA_VERSION,
            "is_champion": False,
        })

    batch_df = spark.createDataFrame(batch_records, schema=scoring_schema)
    batch_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(SCORING_TABLE)
    total_written += len(batch_records)

    # Free serialized bytes; replace with sentinel so downstream `is not None`
    # checks (quality gate, training summary) still pass.
    del batch_records, batch_df
    for i in range(batch_start, batch_end):
        if trained_models[i]["model_binary"] is not None:
            trained_models[i]["model_binary"] = b"WRITTEN"
    gc.collect()

print(f"Appended {total_written} models to {SCORING_TABLE}")

if ENABLE_ICEBERG:
    enable_iceberg(spark, SCORING_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC # Training Summary
# MAGIC
# MAGIC Per-model results table showing partition, model type, MAPE, and registry info.
# MAGIC This is the same data logged to MLflow as `summary/training_summary.json`.

# COMMAND ----------

print(f"Run ID: {run_id}")
print(f"Data version: {DATA_VERSION}")
print(f"Scoring table: {SCORING_TABLE}")
print(f"Total models trained: {len(trained_models)}")

successful_models = [row for row in trained_models if row["model_binary"] is not None and not row["error"]]

# Build partition -> segment lookup from the source data
partition_to_segment = {
    r["PARTITION_COLUMN"]: r["MARKET_SEGMENT"]
    for r in df_raw.select("PARTITION_COLUMN", "MARKET_SEGMENT").distinct().collect()
}

if successful_models:
    summary_data = []
    for row in successful_models:
        partition = row["PARTITION_COLUMN"] or ""
        segment = partition_to_segment.get(partition, "")
        nested_rid = nested_run_ids.get(trained_models.index(row))
        row_data = {
            "MARKET_SEGMENT": segment,
            "Partition": truncate_partition_name(partition, max_len=50),
            "Model": row["MODEL_USED"],
            "MAPE (%)": round(row["mape"], 2) if is_valid_metric(row.get("mape")) else None,
            "Nested Run": nested_rid[:8] + "..." if nested_rid else None,
        }
        summary_data.append(row_data)
    summary_df = pd.DataFrame(summary_data).sort_values(["Partition", "Model"])
    display(summary_df)
else:
    summary_df = None
    print("No models successfully trained.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segment-Level Summary
# MAGIC
# MAGIC Aggregate training MAPE by MARKET_SEGMENT.

# COMMAND ----------

if summary_df is not None:
    segment_summary = summary_df.groupby(["MARKET_SEGMENT", "Model"]).agg(
        avg_mape=("MAPE (%)", "mean"),
        partition_count=("Partition", "count")
    ).reset_index()
    segment_summary.columns = ["MARKET_SEGMENT", "Model", "Avg MAPE (%)", "Partition Count"]
    segment_summary["Avg MAPE (%)"] = segment_summary["Avg MAPE (%)"].round(2)
    segment_summary = segment_summary.sort_values(["MARKET_SEGMENT", "Model"])

    print(f"\nSegment-level summary ({len(segment_summary)} rows):")
    display(segment_summary)
else:
    print("No segment summary available (no models trained successfully).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Error Logging
# MAGIC
# MAGIC Write training errors to the `pipeline_errors` Delta table.
# MAGIC Downstream notebooks and dashboards can query this table for alerting and diagnostics.

# COMMAND ----------

if training_errors:
    pipeline_errors = []
    for row in training_errors:
        pipeline_errors.append({
            "stage": "training",
            "partition": row["PARTITION_COLUMN"],
            "model_type": row["MODEL_USED"],
            "severity": "error",
            "message": row["error"],
        })
    write_pipeline_errors(spark, CATALOG, SCHEMA, pipeline_errors, "02_model_training", run_id)

# COMMAND ----------

# Pass run_id to downstream tasks so NB03/04/05 can use it directly instead of
# rediscovering it via find_latest_training_run. Prevents race conditions when two
# pipeline runs execute concurrently in the same MLflow experiment.
try:
    dbutils.jobs.taskValues.set(key="training_run_id", value=run_id)
    print(f"Set task value training_run_id={run_id}")
except Exception:
    pass  # not running in a job (manual notebook execution)

# COMMAND ----------

# MAGIC %md
# MAGIC **Next step:** Run the evaluation notebook (`03_model_evaluation.py`) to compare models and set `@champion` aliases.
# MAGIC Then run inference notebook (`04_inference.py`) to generate predictions.