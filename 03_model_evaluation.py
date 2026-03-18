# Databricks notebook source
# MAGIC %md
# MAGIC # Model Evaluation
# MAGIC
# MAGIC Evaluates trained models per partition, selects champions by HORIZON_MAPE,
# MAGIC and logs results to MLflow.
# MAGIC
# MAGIC The `eval_source` widget controls where models are loaded from:
# MAGIC - `mlflow`: Models in MLflow Model Registry (default)
# MAGIC - `delta`: Model binaries in a Delta table (brand-level scale)

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

import yaml
from datetime import datetime

import numpy as np
import pandas as pd

import mlflow

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Modules

# COMMAND ----------

from modeling_dev.metric_utils import calculate_horizon_metrics, apply_evaluation_pipeline, create_backtest_windows
from modeling_dev.data_utils import (
    load_latest_version, enable_iceberg, write_pipeline_errors,
    truncate_partition_name,
    orchestrate_dq_checks, prepare_for_scoring,
    get_next_metrics_version, align_schema_for_delta,
)
from modeling_dev.mlflow_utils import (
    setup_mlflow_client, load_models_from_latest_run_parallel,
    load_models_from_delta, log_delta_lineage,
    set_champion_aliases_parallel, clear_stale_champion_aliases,
    update_nested_runs_parallel,
)
from modeling_dev.chart_utils import create_horizon_ape_chart, create_champion_summary_chart

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("model_name_prefix", "forecasting")
dbutils.widgets.text("experiment_name", "/Shared/dab-pipeline-forecasting")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib")
dbutils.widgets.dropdown("eval_source", "mlflow", ["mlflow", "delta"])
dbutils.widgets.text("scoring_table", "scoring_output")
dbutils.widgets.text("config_path", "configs/model_config.yaml")
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO")
dbutils.widgets.dropdown("is_synthetic_data", "true", ["true", "false"])
dbutils.widgets.dropdown("enable_iceberg", "true", ["true", "false"])
dbutils.widgets.dropdown("data_quality_check", "true", ["true", "false"])
dbutils.widgets.dropdown("log_mlflow_charts", "true", ["true", "false"])
dbutils.widgets.text("horizon_range", "1,6")

# Rolling window backtesting: 1 = disabled, >1 = evaluate across multiple temporal windows
dbutils.widgets.text("backtest_windows", "1")
dbutils.widgets.text("backtest_stride", "1")

# Quality gate toggle: break pipeline if APE thresholds are exceeded
dbutils.widgets.dropdown("break_on_ape_failure", "false", ["true", "false"])

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME_PREFIX = dbutils.widgets.get("model_name_prefix")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}"
EVAL_SOURCE = dbutils.widgets.get("eval_source")
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('scoring_table')}"
CONFIG_PATH = dbutils.widgets.get("config_path")

PARTITION_COLS = [c.strip() for c in dbutils.widgets.get("partition_columns").split(",")]

IS_SYNTHETIC_DATA = dbutils.widgets.get("is_synthetic_data") == "true"
ENABLE_ICEBERG = dbutils.widgets.get("enable_iceberg") == "true"
DATA_QUALITY_CHECK = dbutils.widgets.get("data_quality_check") == "true"
LOG_MLFLOW_CHARTS = dbutils.widgets.get("log_mlflow_charts") == "true"
BREAK_ON_APE_FAILURE = dbutils.widgets.get("break_on_ape_failure") == "true"
BACKTEST_WINDOWS = int(dbutils.widgets.get("backtest_windows"))
BACKTEST_STRIDE = int(dbutils.widgets.get("backtest_stride"))

horizon_range_str = dbutils.widgets.get("horizon_range")
HORIZON_START, HORIZON_END = [int(x.strip()) for x in horizon_range_str.split(",")]
MAX_HORIZON = HORIZON_END

print(f"Eval source: {EVAL_SOURCE}")
print(f"Data table: {DATA_TABLE}")
print(f"Partitioning by: {PARTITION_COLS}")
print(f"Horizon range: Q+{HORIZON_START} through Q+{HORIZON_END}")
if BACKTEST_WINDOWS > 1:
    print(f"Backtesting: {BACKTEST_WINDOWS} windows, stride={BACKTEST_STRIDE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
with open("configs/variable_config.yaml", "r") as f:
    variable_config = yaml.safe_load(f)

quality_gates = config.get("QUALITY_GATES", {})
MAX_CHAMPION_MAPE = quality_gates.get("MAX_CHAMPION_MAPE", 200)
MIN_CHAMPION_MAPE = quality_gates.get("MIN_CHAMPION_MAPE", 1)
print(f"Quality gates: MAX_MAPE={MAX_CHAMPION_MAPE}%, MIN_MAPE={MIN_CHAMPION_MAPE}%")

# NeuralForecast model binaries are too large for Spark gRPC (~1.4GB vs 128MB limit).
# Mirror notebook 02: NF models evaluate on the driver, standard models via applyInPandas.
nf_model_types = set()
for m_name, m_cfg in config.get("models", {}).items():
    if m_cfg.get("type") == "neuralforecast":
        nf_model_types.add(m_name.lower())

# COMMAND ----------

experiment = mlflow.set_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Load Data

# COMMAND ----------

SELECT_COLUMNS = list(variable_config["SELECT_COLUMNS"])
for col in PARTITION_COLS:
    if col not in SELECT_COLUMNS:
        SELECT_COLUMNS.append(col)

df_raw, DATA_VERSION = load_latest_version(spark, DATA_TABLE, SELECT_COLUMNS, is_synthetic_data=IS_SYNTHETIC_DATA)
print(f"Data version: {DATA_VERSION}")

df_raw = df_raw.withColumn("PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLS]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

dq_errors_to_log = []
excluded_partitions = set()

if DATA_QUALITY_CHECK:
    df_raw, dq_errors_to_log, excluded_partitions = orchestrate_dq_checks(
        spark, df_raw, config, stage="data_quality_eval"
    )

# COMMAND ----------

partitions = [row["PARTITION_COLUMN"] for row in df_raw.select("PARTITION_COLUMN").distinct().collect()]
print(f"Found {len(partitions)} partitions for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backtest Windows

# COMMAND ----------

def _build_eval_windows(data_df, backtest_windows, backtest_stride, max_horizon):
    """Build evaluation windows. Returns single fallback entry when backtesting is disabled."""
    fallback = [{'window_idx': None, 'eval_start_quarter': None,
                 'eval_end_quarter': None, 'eval_quarters': None}]
    if backtest_windows > 1:
        quarter_pdf = data_df.select("QUARTER").distinct().toPandas()
        windows = create_backtest_windows(
            quarter_pdf, backtest_windows, stride_quarters=backtest_stride,
            min_train_quarters=6, max_horizon=max_horizon
        )
        return windows if windows else fallback
    return fallback

eval_windows = _build_eval_windows(df_raw, BACKTEST_WINDOWS, BACKTEST_STRIDE, MAX_HORIZON)
print(f"Evaluation windows: {len(eval_windows)}")
for w in eval_windows:
    if w['window_idx'] is not None:
        print(f"  Window {w['window_idx']}: {w['eval_start_quarter']} to {w['eval_end_quarter']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Load Models
# MAGIC
# MAGIC Load trained models from notebook 02. Path depends on `eval_source` widget.

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

# MAGIC %md
# MAGIC ### MLflow Path

# COMMAND ----------

if EVAL_SOURCE == "mlflow":
    print("Loading models from latest training run...")

    models, latest_parent_id = load_models_from_latest_run_parallel(
        client, EXPERIMENT_NAME, SCORING_TABLE, CATALOG, SCHEMA, MODEL_NAME_PREFIX,
        parent_run_id=_job_training_run_id,
    )

    # Load model artifacts serially on driver. Workers cannot authenticate to MLflow
    # artifact storage, and parallel bulk download stalls at segment scale.
    print("Loading model artifacts on driver (serial)...")
    for model_info in models:
        if model_info["model_type"] in nf_model_types:
            continue  # NF models are lazy-loaded in Phase 2
        if model_info["pyfunc_model"] is None and model_info.get("artifact_uri"):
            loaded = mlflow.pyfunc.load_model(model_info["artifact_uri"])
            model_info["pyfunc_model"] = loaded.unwrap_python_model()

    for model_info in models:
        pyfunc = model_info["pyfunc_model"]
        entry = {
            "PARTITION_COLUMN": model_info["partition"],
            "_model_type": model_info["model_type"],
            "_model": pyfunc,
            "_model_config": pyfunc.model_config if pyfunc is not None else {},
            "_model_name": model_info["registered_name"],
            "_version": model_info["version"],
            "_nested_run_id": model_info["nested_run_id"],
        }
        if model_info.get("artifact_uri"):
            entry["_artifact_uri"] = model_info["artifact_uri"]
        model_data.append(entry)

    if not model_data:
        dbutils.notebook.exit("No models found in latest training run")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delta Path

# COMMAND ----------

if EVAL_SOURCE == "delta":
    print(f"Loading models from {SCORING_TABLE}...")

    delta_models, latest_run_id = load_models_from_delta(
        spark, SCORING_TABLE,
        validate_data_table=DATA_TABLE,
        catalog=CATALOG, schema=SCHEMA, model_name_prefix=MODEL_NAME_PREFIX,
    )

    if not delta_models:
        dbutils.notebook.exit("No models found")

    for model_info in delta_models:
        is_nf = model_info["model_type"] in nf_model_types
        model_data.append({
            "PARTITION_COLUMN": model_info["PARTITION_COLUMN"],
            "_model_type": model_info["model_type"],
            "_model": model_info["model"] if is_nf else None,
            "_model_config": model_info["model_config"],
            "_model_name": model_info["model_name"],
            "_version": model_info["version"],
            "_nested_run_id": model_info["nested_run_id"],
        })
    del delta_models

# COMMAND ----------

print(f"\nLoaded {len(model_data)} models via {EVAL_SOURCE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Parallel Evaluation
# MAGIC
# MAGIC Evaluate all models using `applyInPandas`. Models are shipped to workers
# MAGIC via closure capture (`model_lookup` dict) or Delta binary join.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Lookup
# MAGIC
# MAGIC Dictionary mapping `(partition, model_type)` to model objects. Referenced inside
# MAGIC `evaluate_model()` and shipped to workers via closure capture.

# COMMAND ----------

model_lookup = {}
nf_model_lookup = {}
for model_info in model_data:
    key = (model_info["PARTITION_COLUMN"], model_info["_model_type"])
    entry = {
        "model": model_info["_model"],
        "config": model_info["_model_config"],
        "model_name": model_info["_model_name"],
        "version": model_info["_version"],
        "nested_run_id": model_info.get("_nested_run_id"),
        "artifact_uri": model_info.get("_artifact_uri"),
    }
    if model_info["_model_type"] in nf_model_types:
        nf_model_lookup[key] = entry
    else:
        model_lookup[key] = entry

print(f"Prepared {len(model_lookup)} standard + {len(nf_model_lookup)} NF models for evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation DataFrame
# MAGIC
# MAGIC Join raw data with model keys so each group contains the rows for one
# MAGIC `(partition, model_type)` combination. For the Delta path, also join the
# MAGIC serialized model binary so each worker gets only the model it needs.

# COMMAND ----------

# Split: standard models go through applyInPandas, NF models evaluate on driver
non_nf_model_data = [m for m in model_data if m["_model_type"] not in nf_model_types]
nf_model_data = [m for m in model_data if m["_model_type"] in nf_model_types]

non_nf_eval_keys = [(m["PARTITION_COLUMN"], m["_model_type"]) for m in non_nf_model_data]
nf_eval_keys = [(m["PARTITION_COLUMN"], m["_model_type"]) for m in nf_model_data]

if nf_eval_keys:
    print(f"NeuralForecast models ({len(nf_eval_keys)}): driver path")
if non_nf_eval_keys:
    non_nf_types = sorted(set(k[1] for k in non_nf_eval_keys))
    print(f"Standard models ({len(non_nf_eval_keys)}): applyInPandas path ({non_nf_types})")

df_with_keys = None
if non_nf_eval_keys:
    eval_keys_df = spark.createDataFrame(non_nf_eval_keys, ["PARTITION_COLUMN", "_model_type"])
    df_with_keys = df_raw.join(eval_keys_df, on="PARTITION_COLUMN", how="inner")

    # Delta path: join model binary from scoring table (standard models only)
    if EVAL_SOURCE == "delta":
        scoring_binary_df = (
            spark.table(SCORING_TABLE)
            .filter(F.col("run_id") == latest_run_id)
            .filter(F.col("model_binary").isNotNull())
            .select(
                "PARTITION_COLUMN",
                F.lower(F.col("MODEL_USED")).alias("_model_type"),
                F.col("model_binary").alias("_model_binary"),
            )
        )
        if nf_model_types:
            scoring_binary_df = scoring_binary_df.filter(
                ~F.col("_model_type").isin(list(nf_model_types))
            )
        df_with_keys = df_with_keys.join(
            scoring_binary_df, on=["PARTITION_COLUMN", "_model_type"], how="inner"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Output Schema

# COMMAND ----------

eval_result_schema = StructType([
    StructField("PARTITION_COLUMN", StringType()),
    StructField("MARKET_SEGMENT", StringType()),
    StructField("MODEL_USED", StringType()),
    StructField("model_name", StringType()),
    StructField("version", StringType()),
    StructField("nested_run_id", StringType()),
    StructField("MAPE", DoubleType()),
    StructField("HORIZON_MAPE", DoubleType()),
    StructField("Q1_MAPE", DoubleType()),
    StructField("Q2_MAPE", DoubleType()),
    StructField("Q3_MAPE", DoubleType()),
    StructField("Q4_MAPE", DoubleType()),
    StructField("Q5_MAPE", DoubleType()),
    StructField("Q6_MAPE", DoubleType()),
    StructField("prediction_count", DoubleType()),
    StructField("BACKTEST_WINDOW", IntegerType()),
    StructField("WINDOW_START_QUARTER", StringType()),
    StructField("WINDOW_END_QUARTER", StringType()),
    StructField("error", StringType()),
])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Function

# COMMAND ----------

DEMAND_SIGNALS = config.get("DEMAND_SIGNALS", ["MOF", "MOF_PLUS_BUFFER", "SELLIN", "SR"])


def evaluate_model(group_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate one (partition, model_type) group across all backtest windows.

    Predicts on full data, then computes per-horizon MAPE for each window.
    Returns one row per window. When backtesting is disabled, returns a single
    row with null window fields (identical to pre-backtest behavior).
    """
    partition = group_df["PARTITION_COLUMN"].iloc[0]
    segment = group_df["MARKET_SEGMENT"].iloc[0]
    model_type = group_df["_model_type"].iloc[0]
    key = (partition, model_type)

    def _make_result(**overrides):
        r = {
            "PARTITION_COLUMN": partition, "MARKET_SEGMENT": segment,
            "MODEL_USED": model_type.upper(),
            "model_name": None, "version": None, "nested_run_id": None,
            "MAPE": None, "HORIZON_MAPE": None,
            **{f"Q{h}_MAPE": None for h in range(1, 7)},
            "prediction_count": None,
            "BACKTEST_WINDOW": None, "WINDOW_START_QUARTER": None,
            "WINDOW_END_QUARTER": None, "error": None,
        }
        r.update(overrides)
        return r

    # ---- Look up model ----
    if key not in model_lookup:
        return pd.DataFrame([_make_result(error=f"Model not found for {key}")])

    info = model_lookup[key]
    base_fields = {
        "model_name": info["model_name"],
        "version": str(info["version"]) if info["version"] is not None else None,
        "nested_run_id": info.get("nested_run_id"),
    }

    # ---- Resolve model object ----
    if "_model_binary" in group_df.columns and group_df["_model_binary"].iloc[0] is not None:
        import cloudpickle
        model = cloudpickle.loads(group_df["_model_binary"].iloc[0])
        group_df = group_df.drop(columns=["_model_binary"])
    elif info["model"] is not None:
        model = info["model"]
    else:
        return pd.DataFrame([_make_result(error=f"No model binary for {key}", **base_fields)])

    try:
        df = prepare_for_scoring(group_df, segment, variable_config)
        df["BILLING_PREDICTED"] = model.predict(None, df)

        # APE, demand signal indicators, null-masking, row filtering
        df = apply_evaluation_pipeline(df, DEMAND_SIGNALS, MAX_HORIZON)

        results = []
        for window in eval_windows:
            w_df = df
            if window['eval_quarters'] is not None:
                w_df = df[df['QUARTER'].isin(window['eval_quarters'])]

            valid_ape = w_df["APE"].dropna()
            w_result = _make_result(
                **base_fields,
                MAPE=valid_ape.mean() if len(valid_ape) > 0 else np.nan,
                prediction_count=float(len(valid_ape)),
                BACKTEST_WINDOW=window['window_idx'],
                WINDOW_START_QUARTER=window['eval_start_quarter'],
                WINDOW_END_QUARTER=window['eval_end_quarter'],
            )

            horizon_metrics = calculate_horizon_metrics(
                w_df, segment, demand_signals=DEMAND_SIGNALS, max_horizon=MAX_HORIZON
            )
            for h in range(1, 7):
                w_result[f"Q{h}_MAPE"] = horizon_metrics.get(f"Q+{h} MAPE", np.nan)

            q_vals = [w_result[f"Q{h}_MAPE"] for h in range(HORIZON_START, HORIZON_END + 1)
                      if pd.notna(w_result.get(f"Q{h}_MAPE"))]
            w_result["HORIZON_MAPE"] = np.mean(q_vals) if q_vals else np.nan

            results.append(w_result)

        return pd.DataFrame(results)

    except Exception as e:
        return pd.DataFrame([_make_result(error=str(e), **base_fields)])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Parallel Evaluation

# COMMAND ----------

all_metrics_parts = []

# --- Phase 1: Standard models via applyInPandas ---
if df_with_keys is not None:
    n_eval_groups = df_with_keys.select("PARTITION_COLUMN", "_model_type").distinct().count()
    if n_eval_groups == 0:
        mlflow_partitions = sorted({m["PARTITION_COLUMN"] for m in non_nf_model_data})
        data_partitions = sorted([
            r["PARTITION_COLUMN"]
            for r in df_raw.select("PARTITION_COLUMN").distinct().collect()
        ])
        mlflow_only = sorted(set(mlflow_partitions) - set(data_partitions))
        data_only = sorted(set(data_partitions) - set(mlflow_partitions))
        raise ValueError(
            f"Inner join returned 0 rows — no overlap between MLflow partition values "
            f"and '{DATA_TABLE}'.\n"
            f"  MLflow partitions ({len(mlflow_partitions)}): {mlflow_partitions}\n"
            f"  Data table partitions ({len(data_partitions)}): {data_partitions}\n"
            f"  In MLflow only: {mlflow_only}\n"
            f"  In data table only: {data_only}\n"
            f"  Likely causes:\n"
            f"    1. NB03 data_table widget ('{DATA_TABLE}') differs from what NB02 trained on\n"
            f"    2. catalog/schema/scoring_table widgets differ between NB02 and NB03, "
            f"causing find_latest_training_run to pick up a stale run with different partitions\n"
            f"    3. NB03 was run with stale/default widget values instead of the job params"
        )
    print(f"Evaluating {n_eval_groups} standard model(s) via applyInPandas...")

    metrics_result_df = (
        df_with_keys
        .repartition(n_eval_groups, "PARTITION_COLUMN", "_model_type")
        .groupBy("PARTITION_COLUMN", "_model_type")
        .applyInPandas(evaluate_model, eval_result_schema)
    )
    all_metrics_parts.append(metrics_result_df.toPandas())

# --- Phase 2: NeuralForecast models on driver (too large for gRPC) ---
if nf_eval_keys:
    print(f"Evaluating {len(nf_eval_keys)} NeuralForecast model(s) on driver...")
    nf_partitions = list({k[0] for k in nf_eval_keys})
    nf_pdf = df_raw.filter(F.col("PARTITION_COLUMN").isin(nf_partitions)).toPandas()

    # Temporarily add NF models to lookup; lazy-load deferred ones
    _std_lookup = model_lookup.copy()
    for key, info in nf_model_lookup.items():
        if info["model"] is None and info.get("artifact_uri"):
            print(f"  Loading deferred model: {key[0]} {key[1].upper()}...")
            loaded = mlflow.pyfunc.load_model(info["artifact_uri"])
            info["model"] = loaded.unwrap_python_model()
            info["config"] = info["model"].model_config
    model_lookup.update(nf_model_lookup)

    for partition, model_type in nf_eval_keys:
        group_df = nf_pdf[nf_pdf["PARTITION_COLUMN"] == partition].copy()
        group_df["_model_type"] = model_type
        result = evaluate_model(group_df)
        all_metrics_parts.append(result)

    # Restore to standard-only (prevents accidental future closure capture)
    model_lookup.clear()
    model_lookup.update(_std_lookup)
    del nf_pdf

all_metrics = pd.concat(all_metrics_parts, ignore_index=True) if all_metrics_parts else pd.DataFrame()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Errors

# COMMAND ----------

errors = all_metrics[all_metrics["error"].notna()]
if not errors.empty:
    print("Errors during evaluation:")
    for _, row in errors.iterrows():
        print(f"  [{truncate_partition_name(row['PARTITION_COLUMN'])}] {row['MODEL_USED']}: {row['error']}")

all_metrics = all_metrics[all_metrics["error"].isna()].drop(columns=["error"])
print(f"\nCollected metrics for {len(all_metrics)} rows")

# COMMAND ----------

if all_metrics.empty:
    dbutils.notebook.exit("No metrics collected")

# COMMAND ----------

all_metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ### Backtest Aggregation

# COMMAND ----------

STABILITY_PENALTY = config.get("EVALUATION", {}).get("BACKTEST", {}).get("stability_penalty", 0.5)

if BACKTEST_WINDOWS > 1:
    agg_dict = {
        'HORIZON_MAPE': ['mean', 'std', 'count'],
        'MAPE': ['mean', 'std'],
        'model_name': 'first',
        'version': 'first',
        'nested_run_id': 'first',
        'prediction_count': 'sum',
    }
    for q in range(1, 7):
        agg_dict[f'Q{q}_MAPE'] = 'mean'

    backtest_agg = all_metrics.groupby(['PARTITION_COLUMN', 'MODEL_USED']).agg(agg_dict).reset_index()

    # Flatten MultiIndex columns
    backtest_agg.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in backtest_agg.columns
    ]
    backtest_agg = backtest_agg.rename(columns={
        'HORIZON_MAPE_mean': 'BT_MEAN_HORIZON_MAPE',
        'HORIZON_MAPE_std': 'BT_STD_HORIZON_MAPE',
        'HORIZON_MAPE_count': 'BT_N_WINDOWS',
        'MAPE_mean': 'MAPE',
        'MAPE_std': 'BT_STD_MAPE',
        'model_name_first': 'model_name',
        'version_first': 'version',
        'nested_run_id_first': 'nested_run_id',
        'prediction_count_sum': 'prediction_count',
    })
    for q in range(1, 7):
        backtest_agg = backtest_agg.rename(columns={f'Q{q}_MAPE_mean': f'Q{q}_MAPE'})

    # Also propagate MARKET_SEGMENT
    seg_map = all_metrics.groupby('PARTITION_COLUMN')['MARKET_SEGMENT'].first()
    backtest_agg['MARKET_SEGMENT'] = backtest_agg['PARTITION_COLUMN'].map(seg_map)

    # Stability penalty: score = mean_mape * (1 + penalty * CV)
    backtest_agg['BT_CV'] = backtest_agg['BT_STD_HORIZON_MAPE'] / backtest_agg['BT_MEAN_HORIZON_MAPE'].replace(0, np.nan)
    backtest_agg['BT_CV'] = backtest_agg['BT_CV'].fillna(0)
    backtest_agg['BT_SELECTION_SCORE'] = backtest_agg['BT_MEAN_HORIZON_MAPE'] * (1 + STABILITY_PENALTY * backtest_agg['BT_CV'])
    backtest_agg['HORIZON_MAPE'] = backtest_agg['BT_MEAN_HORIZON_MAPE']

    print(f"Backtest aggregation: {len(backtest_agg)} model(s) across {backtest_agg['BT_N_WINDOWS'].iloc[0]} windows")
    print(f"Stability penalty: {STABILITY_PENALTY}")

# COMMAND ----------

# Switch selection source based on backtesting mode
if BACKTEST_WINDOWS > 1:
    all_metrics_for_selection = backtest_agg
    selection_col = "BT_SELECTION_SCORE"
else:
    all_metrics_for_selection = all_metrics
    selection_col = "HORIZON_MAPE"

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Select Champions
# MAGIC
# MAGIC Pick the best model per partition. When backtesting is enabled, uses
# MAGIC stability-penalized score; otherwise uses HORIZON_MAPE.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Metrics to Delta

# COMMAND ----------

metrics_to_save = all_metrics.copy()
metrics_to_save["LOAD_TS"] = datetime.now()
metrics_to_save["eval_source"] = EVAL_SOURCE
metrics_to_save["DATA_TABLE"] = DATA_TABLE
metrics_to_save["DATA_VERSION"] = DATA_VERSION
metrics_to_save["is_champion"] = False

if EVAL_SOURCE == "delta" and model_data:
    metrics_to_save["run_id"] = model_data[0]["_version"]
else:
    metrics_to_save["run_id"] = None

metrics_table_name = f"{CATALOG}.{SCHEMA}.predictions_metrics"

# COMMAND ----------

new_version = get_next_metrics_version(spark, metrics_table_name)
metrics_to_save["METRICS_VERSION"] = new_version

metrics_spark_df = align_schema_for_delta(
    spark.createDataFrame(metrics_to_save), metrics_table_name, spark
)

if new_version == "V_1":
    metrics_spark_df.write.mode("overwrite").saveAsTable(metrics_table_name)
else:
    metrics_spark_df.write.option("mergeSchema", "true").mode("append").saveAsTable(metrics_table_name)

if ENABLE_ICEBERG:
    enable_iceberg(spark, metrics_table_name)
print(f"Saved metrics as {new_version} to {metrics_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion Selection

# COMMAND ----------

valid_metrics = all_metrics_for_selection[
    all_metrics_for_selection["MAPE"].notna() & (all_metrics_for_selection["MAPE"] > 0)
].copy()

if valid_metrics.empty:
    dbutils.notebook.exit("No valid metrics")

# Fall back to MAPE if primary selection column is unavailable
if selection_col not in valid_metrics.columns or valid_metrics[selection_col].isna().all():
    if "HORIZON_MAPE" in valid_metrics.columns and not valid_metrics["HORIZON_MAPE"].isna().all():
        selection_col = "HORIZON_MAPE"
    else:
        selection_col = "MAPE"
    print(f"Primary selection metric unavailable, falling back to {selection_col}")
else:
    print(f"Selecting champions by {selection_col} (Q+{HORIZON_START} through Q+{HORIZON_END})")

best_idx = valid_metrics.groupby("PARTITION_COLUMN")[selection_col].idxmin()
valid_metrics["is_champion"] = False
valid_metrics.loc[best_idx, "is_champion"] = True

# Warn about partitions with no champion
expected = set(all_metrics_for_selection["PARTITION_COLUMN"].unique())
got = set(valid_metrics[valid_metrics["is_champion"]]["PARTITION_COLUMN"].unique())
missing = expected - got
if missing:
    print(f"\nWARNING: {len(missing)} partition(s) have no champion model:")
    for p in sorted(missing):
        print(f"  - {truncate_partition_name(p)}")
    if BREAK_ON_APE_FAILURE:
        raise ValueError(
            f"Quality gate failed: {len(missing)} partition(s) have no champion model: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}"
        )
    else:
        print("CONTINUING (break_on_ape_failure=false)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Champion (MLflow)

# COMMAND ----------

if EVAL_SOURCE == "mlflow":
    champion_rows = [valid_metrics.loc[idx].to_dict() for idx in best_idx]

    # Clear stale @champion aliases from non-champion models so inference
    # notebook only finds the correct champion per partition
    champion_names = {row['model_name'] for row in champion_rows if row.get('model_name')}
    stale_names = [
        name for name in valid_metrics['model_name'].dropna().unique()
        if name not in champion_names
    ]
    if stale_names:
        clear_stale_champion_aliases(client, stale_names)

    set_champion_aliases_parallel(client, champion_rows)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Champion (Delta)

# COMMAND ----------

if EVAL_SOURCE == "delta":
    champion_keys = [(valid_metrics.loc[idx, 'PARTITION_COLUMN'], valid_metrics.loc[idx, 'MODEL_USED']) for idx in best_idx]

    for partition, model_used in champion_keys:
        mape_val = valid_metrics.loc[valid_metrics['PARTITION_COLUMN'] == partition].iloc[0][selection_col]
        print(f"[{truncate_partition_name(partition)}] Champion: {model_used} ({selection_col}: {mape_val:.2f}%)")

    eval_run_id = model_data[0]["_version"] if model_data else None

    scoring_cols = {c.name for c in spark.table(SCORING_TABLE).schema}
    if "is_champion" not in scoring_cols:
        spark.sql(f"ALTER TABLE {SCORING_TABLE} ADD COLUMNS (is_champion BOOLEAN)")

    # Reset is_champion for this run, then set champions
    spark.sql(f"""
        UPDATE {SCORING_TABLE}
        SET is_champion = false
        WHERE run_id = '{eval_run_id}'
    """)

    if champion_keys:
        champion_pairs = ", ".join([f"('{partition}', '{mtype}')" for partition, mtype in champion_keys])
        spark.sql(f"""
            UPDATE {SCORING_TABLE}
            SET is_champion = true
            WHERE run_id = '{eval_run_id}'
              AND (PARTITION_COLUMN, MODEL_USED) IN ({champion_pairs})
        """)

    print(f"\nUpdated {SCORING_TABLE}: {len(champion_keys)} champion(s) set")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quality Gates

# COMMAND ----------

champions = valid_metrics[valid_metrics["is_champion"]].copy()

# Flag suspiciously good models
suspicious = champions[champions[selection_col] < MIN_CHAMPION_MAPE]
if not suspicious.empty:
    print(f"WARNING: {len(suspicious)} champion(s) with suspiciously low {selection_col} (< {MIN_CHAMPION_MAPE}%):")
    for _, row in suspicious.iterrows():
        print(f"  [{truncate_partition_name(row['PARTITION_COLUMN'])}] {row['MODEL_USED']}: {row[selection_col]:.2f}%")
    print("  May indicate data leakage or overfitting.\n")

# Reject models above threshold
bad_models = champions[champions[selection_col] > MAX_CHAMPION_MAPE]
if not bad_models.empty:
    print(f"WARNING: {len(bad_models)} champion(s) exceed {MAX_CHAMPION_MAPE}% {selection_col}:")
    for _, row in bad_models.iterrows():
        print(f"  [{truncate_partition_name(row['PARTITION_COLUMN'])}] {row['MODEL_USED']}: {row[selection_col]:.1f}%")
    if BREAK_ON_APE_FAILURE:
        raise ValueError(
            f"Quality gate failed: {len(bad_models)} champion(s) exceed {MAX_CHAMPION_MAPE}% {selection_col}"
        )
    else:
        print("CONTINUING (break_on_ape_failure=false)")

if suspicious.empty and bad_models.empty:
    print(f"Quality gates passed: {len(champions)} champion(s) within [{MIN_CHAMPION_MAPE}%, {MAX_CHAMPION_MAPE}%]")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update Metrics Table with Champion Flag

# COMMAND ----------

SINGLE_QUOTE = "'"
champion_conditions = " OR ".join(
    f"(PARTITION_COLUMN = '{row['PARTITION_COLUMN'].replace(SINGLE_QUOTE, SINGLE_QUOTE * 2)}' AND MODEL_USED = '{row['MODEL_USED']}')"
    for _, row in champions.iterrows()
)
spark.sql(f"""
    UPDATE {metrics_table_name}
    SET is_champion = true
    WHERE METRICS_VERSION = '{new_version}'
      AND ({champion_conditions})
""")

print(f"Updated {metrics_table_name}: {len(champions)} champion(s) flagged in {new_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # MLflow Logging
# MAGIC
# MAGIC Resume the training parent run and log evaluation params, metrics, artifacts,
# MAGIC and charts. Update nested runs with per-model eval metrics.

# COMMAND ----------

if EVAL_SOURCE == "mlflow":
    training_run_id = latest_parent_id
elif _job_training_run_id:
    training_run_id = _job_training_run_id
else:
    training_run_id = latest_run_id
print(f"Resuming training run: {training_run_id}")

# Pre-compute chart before opening MLflow run
champion_chart_fig = None
if LOG_MLFLOW_CHARTS and not champions.empty:
    try:
        champion_chart_fig = create_champion_summary_chart(champions, (HORIZON_START, HORIZON_END))
    except Exception as chart_err:
        print(f"Warning: could not create champion summary chart: {chart_err}")

mlflow.end_run()
with mlflow.start_run(run_id=training_run_id) as eval_run:
    eval_run_id = eval_run.info.run_id

    # Data quality errors
    if dq_errors_to_log:
        write_pipeline_errors(spark, CATALOG, SCHEMA, dq_errors_to_log, "03_model_evaluation", eval_run_id)
        mlflow.log_metric("partitions_excluded_dq", len(dq_errors_to_log))

    # Params
    eval_params = {
        "eval_source": EVAL_SOURCE,
        "data_table": DATA_TABLE,
        "data_version": DATA_VERSION,
        "horizon_range": f"{HORIZON_START}-{HORIZON_END}",
    }
    if BACKTEST_WINDOWS > 1:
        eval_params["backtest_windows"] = BACKTEST_WINDOWS
        eval_params["backtest_stride"] = BACKTEST_STRIDE
        eval_params["stability_penalty"] = STABILITY_PENALTY
    mlflow.log_params(eval_params)

    # Metrics
    mlflow.log_metric("models_evaluated", len(all_metrics))
    mlflow.log_metric("champions_selected", len(champions))
    mlflow.log_metric("avg_champion_mape", champions[selection_col].mean() if not champions.empty else 0)

    if BACKTEST_WINDOWS > 1 and not champions.empty:
        if 'BT_CV' in champions.columns:
            mlflow.log_metric("avg_champion_cv", champions['BT_CV'].mean())
        if 'BT_STD_HORIZON_MAPE' in champions.columns:
            mlflow.log_metric("avg_champion_std_mape", champions['BT_STD_HORIZON_MAPE'].mean())

    if not champions.empty:
        for q in range(HORIZON_START, HORIZON_END + 1):
            col = f"Q{q}_MAPE"
            if col in champions.columns:
                mlflow.log_metric(f"avg_champion_q{q}_mape", champions[col].mean())

    # Artifacts
    mlflow.log_artifact(CONFIG_PATH, "config")
    mlflow.log_table(metrics_to_save, artifact_file="tables/predictions_metrics.json")
    mlflow.log_table(champions, artifact_file="tables/champions.json")

    if champion_chart_fig is not None:
        import matplotlib.pyplot as plt
        mlflow.log_figure(champion_chart_fig, "charts/champion_horizon_summary.png")
        plt.close(champion_chart_fig)

    # Tags and lineage
    mlflow.set_tag("scoring_table", SCORING_TABLE)
    mlflow.set_tag("metrics_table", metrics_table_name)

    avg_mape = champions[selection_col].mean() if not champions.empty else 0
    mlflow.set_tag("mlflow.note.content", (
        f"**Evaluation** | {len(all_metrics)} models evaluated, "
        f"{len(champions)} champions selected\n\n"
        f"Avg champion MAPE: {avg_mape:.2f}%\n\n"
        f"Horizons: Q+{HORIZON_START} to Q+{HORIZON_END}"
    ))

    log_delta_lineage([
        (DATA_TABLE, "evaluation_data"),
        (SCORING_TABLE, "scoring_input"),
        (metrics_table_name, "metrics_output"),
    ])

print(f"Evaluation run ID: {eval_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Update Nested Runs

# COMMAND ----------

eval_logged = update_nested_runs_parallel(
    valid_metrics,
    client,
    experiment_id=EXPERIMENT_ID,
    horizon_start=HORIZON_START,
    horizon_end=HORIZON_END,
    log_charts=LOG_MLFLOW_CHARTS,
    chart_func=create_horizon_ape_chart,
    backtest_windows=BACKTEST_WINDOWS,
)

print(f"Updated {eval_logged} MLflow nested runs with eval metrics")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Evaluation Results

# COMMAND ----------

display_cols = ['PARTITION_COLUMN', 'MODEL_USED', 'version', 'HORIZON_MAPE', 'Q1_MAPE', 'Q2_MAPE', 'Q3_MAPE', 'Q4_MAPE', 'Q5_MAPE', 'Q6_MAPE', 'prediction_count', 'is_champion']
if BACKTEST_WINDOWS > 1:
    display_cols = ['PARTITION_COLUMN', 'MODEL_USED', 'version', 'BT_SELECTION_SCORE', 'BT_MEAN_HORIZON_MAPE', 'BT_STD_HORIZON_MAPE', 'BT_CV', 'BT_N_WINDOWS', 'Q1_MAPE', 'Q2_MAPE', 'Q3_MAPE', 'Q4_MAPE', 'Q5_MAPE', 'Q6_MAPE', 'prediction_count', 'is_champion']
display_df = valid_metrics[[c for c in display_cols if c in valid_metrics.columns]].copy()
display_df = display_df.sort_values(['PARTITION_COLUMN', 'HORIZON_MAPE'])

for col in display_df.select_dtypes(include=[np.number]).columns:
    display_df[col] = display_df[col].round(2)

display(display_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Errors

# COMMAND ----------

pipeline_errors = []

if not errors.empty:
    for _, row in errors.iterrows():
        pipeline_errors.append({
            "stage": "evaluation",
            "partition": row["PARTITION_COLUMN"],
            "model_type": row["MODEL_USED"],
            "severity": "error",
            "message": row["error"],
        })

if not suspicious.empty:
    for _, row in suspicious.iterrows():
        pipeline_errors.append({
            "stage": "quality_gate",
            "partition": row["PARTITION_COLUMN"],
            "model_type": row["MODEL_USED"],
            "severity": "warning",
            "message": f"Suspiciously low {selection_col}: {row[selection_col]:.2f}% (threshold: {MIN_CHAMPION_MAPE}%)",
        })

if pipeline_errors:
    write_pipeline_errors(spark, CATALOG, SCHEMA, pipeline_errors, "03_model_evaluation", eval_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC
# MAGIC **Next step:** Run inference (`04_inference.py`) to generate predictions using champion models.
