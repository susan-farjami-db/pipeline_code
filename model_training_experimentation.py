# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training Experimentation
# MAGIC
# MAGIC Test `train_model()` on a single partition before full parallel execution.
# MAGIC Same code path as the production pipeline -- no duplicated training logic.
# MAGIC Supports any model type defined in `model_config.yaml`.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./01_training_functions

# COMMAND ----------

from pyspark.sql import functions as F
from modeling_dev.data_utils import load_latest_version

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib")
dbutils.widgets.text("config_path", "configs/model_config.yaml")
dbutils.widgets.dropdown("is_synthetic_data", "true", ["true", "false"])
dbutils.widgets.dropdown("grid_search", "false", ["true", "false"])
dbutils.widgets.text("model_types", "prophet,xgboost", "Model types (comma-separated)")
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO", "Columns to partition on (comma-separated)")
dbutils.widgets.text("specific_partition", "", "Specific partition value (leave empty for first)")

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}"
CONFIG_PATH = dbutils.widgets.get("config_path")
IS_SYNTHETIC_DATA = dbutils.widgets.get("is_synthetic_data") == "true"
USE_GRID_SEARCH = dbutils.widgets.get("grid_search") == "true"
MODEL_TYPES = [m.strip() for m in dbutils.widgets.get("model_types").split(",") if m.strip()]
PARTITION_COLUMNS = [c.strip() for c in dbutils.widgets.get("partition_columns").split(",") if c.strip()]
SPECIFIC_PARTITION = dbutils.widgets.get("specific_partition").strip()

print(f"Data table: {DATA_TABLE}")
print(f"Synthetic data: {'enabled' if IS_SYNTHETIC_DATA else 'disabled'}")
print(f"Grid search: {'enabled' if USE_GRID_SEARCH else 'disabled'}")
print(f"Model types: {MODEL_TYPES}")
print(f"Partition columns: {PARTITION_COLUMNS}")
print(f"Specific partition: {SPECIFIC_PARTITION if SPECIFIC_PARTITION else '(first available)'}")

# COMMAND ----------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

SELECT_COLUMNS = VAR_CONFIG.get("SELECT_COLUMNS", [])

df_spark, DATA_VERSION = load_latest_version(
    spark, DATA_TABLE, SELECT_COLUMNS, is_synthetic_data=IS_SYNTHETIC_DATA
)
print(f"Data version: {DATA_VERSION}")

df_spark = df_spark.withColumn(
    "PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLUMNS])
)

df_spark.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Select Partition

# COMMAND ----------

partitions = sorted(
    [row["PARTITION_COLUMN"] for row in df_spark.select("PARTITION_COLUMN").distinct().collect()]
)
print(f"Available partitions ({len(partitions)}):")
for p in partitions:
    print(f"  {p}")

if SPECIFIC_PARTITION and SPECIFIC_PARTITION in partitions:
    TEST_PARTITION = SPECIFIC_PARTITION
elif SPECIFIC_PARTITION:
    print(f"\nWARNING: '{SPECIFIC_PARTITION}' not found, using first available")
    TEST_PARTITION = partitions[0]
else:
    TEST_PARTITION = partitions[0]

df_test = df_spark.filter(F.col("PARTITION_COLUMN") == TEST_PARTITION)
df = df_test.toPandas().reset_index(drop=True)

if "PARTITION_COLUMN" in df.columns:
    df = df.drop(columns=["PARTITION_COLUMN"])

MARKET_SEGMENT = df["MARKET_SEGMENT"].iloc[0]
print(f"\nSelected partition: {TEST_PARTITION}")
print(f"Segment: {MARKET_SEGMENT}, Rows: {len(df)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preview

# COMMAND ----------

validation_horizon, prediction_horizon = extract_horizons(df)
print(f"Validation horizon: {validation_horizon}")
print(f"Prediction horizon: {prediction_horizon}")
print(f"Training rows:      {len(df) - validation_horizon - prediction_horizon}")
print(f"Validation rows:    {validation_horizon}")
print(f"Test rows:          {prediction_horizon}")
print()

seg_config = load_segment_config(config, MARKET_SEGMENT)
n_features = get_feature_count(config, MARKET_SEGMENT)
print(f"Available regressors: {seg_config['all_regressors']}")
print(f"n_features for RFE:   {n_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Models
# MAGIC
# MAGIC Calls `train_model()` from `01_training_functions` for each selected model type.

# COMMAND ----------

# Override grid search settings based on widget
for seg_name, seg_conf in config.get("segments", {}).items():
    if "grid_search" in seg_conf:
        seg_conf["grid_search"] = {k: USE_GRID_SEARCH for k in seg_conf["grid_search"]}
general_conf = config.get("general", {})
if "grid_search" in general_conf:
    general_conf["grid_search"] = {k: USE_GRID_SEARCH for k in general_conf["grid_search"]}

# Reload seg_config after grid search override
seg_config = load_segment_config(config, MARKET_SEGMENT)

results = {}
for model_type in MODEL_TYPES:
    print(f"\n{'=' * 60}")
    print(f"Training {model_type.upper()} on {TEST_PARTITION}")
    print(f"{'=' * 60}")
    try:
        model, predictions, metrics, params = train_model(
            model_type, df.copy(), MARKET_SEGMENT, seg_config, config
        )
        results[model_type] = {
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
            "params": params,
        }
        mape = metrics.get("mape", float("nan"))
        regressors = metrics.get("regressors", [])
        print(f"\nValidation MAPE: {mape:.2f}%")
        if regressors:
            print(f"Selected features ({len(regressors)}): {regressors}")
        if metrics.get("dropped_features"):
            print(f"Dropped features: {metrics['dropped_features']}")
        if metrics.get("grid_search_enabled"):
            gs_combos = metrics.get("grid_search_combos", 0) or metrics.get("grid_search_combos_tried", 0)
            print(f"Grid search combos: {gs_combos}")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        results[model_type] = {"error": str(e)}

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictions

# COMMAND ----------

for model_type, result in results.items():
    if "error" in result:
        print(f"\n{model_type.upper()}: FAILED - {result['error']}")
        continue
    print(f"\n--- {model_type.upper()} ---")
    display(result["predictions"])

# COMMAND ----------

# MAGIC %md
# MAGIC # Compare Models

# COMMAND ----------

print(f"Partition: {TEST_PARTITION}")
print(f"Segment:   {MARKET_SEGMENT}")
print(f"Rows:      {len(df)}")
print(f"Grid:      {'on' if USE_GRID_SEARCH else 'off'}")
print()

for model_type, result in results.items():
    if "error" in result:
        print(f"  {model_type.upper():<20s} FAILED: {result['error']}")
    else:
        mape = result["metrics"].get("mape", float("nan"))
        n_reg = len(result["metrics"].get("regressors", []))
        suffix = f"({n_reg} features)" if n_reg else "(univariate)"
        print(f"  {model_type.upper():<20s} MAPE: {mape:>8.2f}%  {suffix}")

valid = {k: v for k, v in results.items() if "error" not in v}
if len(valid) > 1:
    best = min(valid, key=lambda k: valid[k]["metrics"].get("mape", float("inf")))
    print(f"\nBest: {best.upper()} ({valid[best]['metrics']['mape']:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Detailed Metrics
# MAGIC
# MAGIC Parameters and diagnostics returned by each training handler.

# COMMAND ----------

for model_type, result in results.items():
    if "error" in result:
        continue
    print(f"\n{'=' * 40}")
    print(f"{model_type.upper()} - Parameters & Metrics")
    print(f"{'=' * 40}")

    metrics = result["metrics"]
    params = result["params"]

    # Skip transient keys (used internally by train_parallel)
    display_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}

    print("\nMetrics:")
    for k, v in display_metrics.items():
        print(f"  {k}: {v}")

    print("\nUsed params:")
    for k, v in params.items():
        print(f"  {k}: {v}")
