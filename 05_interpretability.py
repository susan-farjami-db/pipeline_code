# Databricks notebook source
# MAGIC %md
# MAGIC # Model Interpretability Notebook
# MAGIC
# MAGIC Analyzes trained models to extract feature importances, SHAP values, Prophet components,
# MAGIC and other interpretability artifacts.
# MAGIC
# MAGIC **interpretability_scope widget:**
# MAGIC - `champions`: Analyze only champion models (one per partition)
# MAGIC - `all`: Analyze all models and compare champion vs candidates
# MAGIC
# MAGIC **run_id widget:**
# MAGIC - Empty: Use latest training run
# MAGIC - Specified: Analyze models from specific training run

# COMMAND ----------

# MAGIC %md
# MAGIC ## Note on Provenance
# MAGIC
# MAGIC This notebook is entirely net-new -- no code was migrated from the original AMD codebase.
# MAGIC The original `GENERATE_PREDICTIONS_ALL_QUARTER.ipynb` had no interpretability or model
# MAGIC explanation functionality.

# COMMAND ----------

# MAGIC %pip install shap matplotlib seaborn -r requirements.txt

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
from typing import Any

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import xgboost  # noqa: F401 -- eagerly import before threaded model loading to avoid circular import race

from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Modules
# MAGIC
# MAGIC Pipeline utilities from `modeling_dev/`: data loading, MLflow helpers, metric name sanitization.

# COMMAND ----------

from modeling_dev.data_utils import (
    load_latest_version, enable_iceberg,
    truncate_partition_name, is_valid_metric,
)
from modeling_dev.mlflow_utils import (
    setup_mlflow_client,
    load_models_from_latest_run_parallel,
    load_models_from_delta,
    get_nested_runs,
    resolve_champion_and_mape_from_runs,
    sanitize_metric_name,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Standard pipeline widgets plus interpretability-specific settings: analysis scope
# MAGIC (champions only vs all models), SHAP sample size, and optional run ID filter.

# COMMAND ----------

# Standard pipeline widgets (same as 02/03/04)
dbutils.widgets.text("catalog", "dash_workshop")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("model_name_prefix", "forecasting")
dbutils.widgets.text("experiment_name", "/Shared/dab-pipeline-forecasting")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib")
dbutils.widgets.text("config_path", "configs/model_config.yaml")
dbutils.widgets.text("scoring_table", "scoring_output")

# Dual-path model loading (same as 03_model_evaluation)
dbutils.widgets.dropdown("eval_source", "mlflow", ["mlflow", "delta"])

# Interpretability-specific widgets
dbutils.widgets.text("run_id", "")  # Optional: analyze specific training run (empty = latest)
dbutils.widgets.dropdown("interpretability_scope", "champions", ["champions", "all"])

# Partition column configuration
dbutils.widgets.text("partition_columns", "MARKET_SEGMENT,SCENARIO")

# Synthetic data toggle
dbutils.widgets.dropdown("is_synthetic_data", "true", ["true", "false"])
dbutils.widgets.dropdown("enable_iceberg", "true", ["true", "false"])

# SHAP configuration
dbutils.widgets.text("shap_sample_size", "100")  # Max samples for SHAP (performance)
dbutils.widgets.dropdown("log_shap_values", "false", ["true", "false"])

# Display toggle (plots are still created and logged to MLflow regardless)
dbutils.widgets.dropdown("display_plots", "false", ["true", "false"])

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME_PREFIX = dbutils.widgets.get("model_name_prefix")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}"
CONFIG_PATH = dbutils.widgets.get("config_path")
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('scoring_table')}"
EVAL_SOURCE = dbutils.widgets.get("eval_source")

# Interpretability settings
RUN_ID = dbutils.widgets.get("run_id").strip() or None
INTERPRETABILITY_SCOPE = dbutils.widgets.get("interpretability_scope")

PARTITION_COLS = [c.strip() for c in dbutils.widgets.get("partition_columns").split(",")]
IS_SYNTHETIC_DATA = dbutils.widgets.get("is_synthetic_data") == "true"
ENABLE_ICEBERG = dbutils.widgets.get("enable_iceberg") == "true"

SHAP_SAMPLE_SIZE = int(dbutils.widgets.get("shap_sample_size"))
LOG_SHAP_VALUES = dbutils.widgets.get("log_shap_values") == "true"
DISPLAY_PLOTS = dbutils.widgets.get("display_plots") == "true"

# Output table name
INTERPRETABILITY_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}_interpretability"

print(f"Eval source: {EVAL_SOURCE}")
print(f"Scope: {INTERPRETABILITY_SCOPE}")
print(f"Run ID filter: {RUN_ID or 'latest'}")
print(f"SHAP sample size: {SHAP_SAMPLE_SIZE}")
print(f"Output table: {INTERPRETABILITY_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
with open("configs/variable_config.yaml", "r") as f:
    variable_config = yaml.safe_load(f)

# Interpretability thresholds from config
interp_config = config.get("INTERPRETABILITY", {})
TOP_N_FEATURES = interp_config.get("TOP_N_FEATURES", 10)
TOP_N_SHAP_DEPENDENCE = interp_config.get("TOP_N_SHAP_DEPENDENCE", 3)
TOP_N_SHAP_FEATURES = interp_config.get("TOP_N_SHAP_FEATURES", 5)
TOP_N_PROPHET_PARTITIONS = interp_config.get("TOP_N_PROPHET_PARTITIONS", 5)

# COMMAND ----------

experiment = mlflow.set_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id

client = setup_mlflow_client()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Load Models
# MAGIC
# MAGIC Loads models from Delta or MLflow based on `eval_source` widget.
# MAGIC Filters by `run_id` if specified, otherwise uses latest run.
# MAGIC Filters by `interpretability_scope` (champions only or all models).

# COMMAND ----------

_KNOWN_WRAPPERS = ("ForecastingModel", "ProphetPredictor", "SklearnPredictor", "StatsForecastPredictor")

def unwrap_model(model, max_depth=5):
    """Unwrap nested model objects: ForecastingModel (PyFunc) -> Predictor wrapper -> raw model.
    Raw model is what SHAP/Prophet component analysis needs (e.g., XGBRegressor, Prophet)."""
    for _ in range(max_depth):
        if not hasattr(model, "model"):
            break
        if type(model).__name__ not in _KNOWN_WRAPPERS:
            break
        model = model.model
    return model

# COMMAND ----------

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

model_data = []

if EVAL_SOURCE == "delta":
    print(f"Loading models from {SCORING_TABLE}...")

    delta_models, resolved_run_id = load_models_from_delta(
        spark, SCORING_TABLE,
        champions_only=(INTERPRETABILITY_SCOPE == "champions"),
        run_id=RUN_ID,
        catalog=CATALOG, schema=SCHEMA, model_name_prefix=MODEL_NAME_PREFIX,
    )
    RUN_ID = resolved_run_id
    training_run_id = _job_training_run_id if _job_training_run_id else resolved_run_id

    if not delta_models:
        dbutils.notebook.exit("No models found in scoring table")

    for model_info in delta_models:
        mc = model_info["model_config"]
        raw_model = unwrap_model(model_info["model"])
        model_data.append({
            "PARTITION_COLUMN": model_info["PARTITION_COLUMN"],
            "model_type": model_info["model_type"],
            "model": raw_model,
            "model_config": mc,
            "is_champion": model_info["is_champion"],
            "mape": model_info["mape"],
            "nested_run_id": model_info["nested_run_id"],
            "handler_type": mc.get("handler_type", "sklearn_api") if mc else "sklearn_api",
        })

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Path

# COMMAND ----------

if EVAL_SOURCE == "mlflow":
    # Load all models from run artifacts (same as notebook 03), then filter by scope
    print("Loading models from latest training run...")
    models, training_run_id = load_models_from_latest_run_parallel(
        client, EXPERIMENT_NAME, SCORING_TABLE, CATALOG, SCHEMA, MODEL_NAME_PREFIX,
        parent_run_id=_job_training_run_id,
    )

    # Build champion/MAPE lookup from nested runs (0 extra API calls)
    _exp_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    nested_df = get_nested_runs(_exp_id, training_run_id)
    run_lookup = resolve_champion_and_mape_from_runs(nested_df)

    for model_info in models:
        nrid = model_info.get("nested_run_id")
        info = run_lookup.get(nrid, {})
        is_champion = info.get("is_champion", False)

        if INTERPRETABILITY_SCOPE == "champions" and not is_champion:
            continue

        # Lazy-load artifact on driver (all models are deferred)
        if model_info["pyfunc_model"] is None and model_info.get("artifact_uri"):
            loaded = mlflow.pyfunc.load_model(model_info["artifact_uri"])
            model_info["pyfunc_model"] = loaded.unwrap_python_model()

        pyfunc = model_info["pyfunc_model"]
        mc = pyfunc.model_config if pyfunc is not None else {}
        model_data.append({
            "PARTITION_COLUMN": model_info["partition"],
            "model_type": model_info["model_type"],
            "model": unwrap_model(pyfunc),
            "model_config": mc,
            "is_champion": is_champion,
            "mape": info.get("mape"),
            "nested_run_id": nrid,
            "handler_type": mc.get("handler_type", "sklearn_api"),
        })

    if not model_data:
        dbutils.notebook.exit("No models found in latest training run")

print(f"\nLoaded {len(model_data)} models")

if not model_data:
    dbutils.notebook.exit("No models found")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Interpretability Functions
# MAGIC
# MAGIC Handler-specific functions for extracting interpretability artifacts:
# MAGIC - **sklearn_api**: SHAP TreeExplainer (or KernelExplainer fallback), feature_importances_
# MAGIC - **prophet**: Component decomposition, regressor coefficients, changepoint analysis
# MAGIC - **statsforecast**: Model parameters, seasonal pattern description
# MAGIC - **neuralforecast**: Descriptive stub (no feature-level interpretability)

# COMMAND ----------

def interpret_sklearn_api(
    model: Any,
    model_config: dict,
    X_sample: pd.DataFrame,
    partition: str,
    create_plots: bool = True,
) -> dict:
    """
    Extract interpretability for sklearn-API models (XGBoost, LightGBM, CatBoost, etc.)

    Returns dict with:
        - feature_importances: dict of feature -> importance
        - shap_values: optional array of SHAP values
        - shap_summary_fig: matplotlib figure
        - interpretation_type: "tree" or "kernel"
    """
    import shap

    result = {
        "feature_importances": {},
        "shap_values": None,
        "shap_summary_fig": None,
        "feature_importance_fig": None,
        "shap_dependence_figs": {},
        "interpretation_type": None,
        "top_features": [],
        "error": None,
    }

    regressors = model_config.get("regressors", [])
    available_regs = [r for r in regressors if r in X_sample.columns]

    if not available_regs:
        result["error"] = "No regressors available in sample data"
        return result

    # NOTE: This uses raw data, not the adjusted data from training.
    # adjust_regressors() replaces below-threshold regressor values during training,
    # so SHAP values here are approximate -- they reflect feature importance
    # on clean data rather than the exact training distribution.
    X = X_sample[available_regs].copy().fillna(0)

    if len(X) == 0:
        result["error"] = "No data samples available"
        return result

    # Native feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        result["feature_importances"] = dict(zip(available_regs, importances.tolist()))
    elif hasattr(model, "coef_"):
        # Linear models (Ridge, etc.)
        coefs = model.coef_ if model.coef_.ndim == 1 else model.coef_[0]
        result["feature_importances"] = dict(zip(available_regs, np.abs(coefs).tolist()))

    if result["feature_importances"]:
        sorted_features = sorted(result["feature_importances"].items(), key=lambda x: -x[1])
        result["top_features"] = [feat_item[0] for feat_item in sorted_features[:TOP_N_SHAP_FEATURES]]

        if create_plots:
            top_n = sorted_features[:TOP_N_FEATURES]
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_names = [feat_item[0] for feat_item in top_n]
            feat_vals = [feat_item[1] for feat_item in top_n]
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
            ax.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1])
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature Importance: {truncate_partition_name(partition, 30)}")
            ax.grid(True, alpha=0.3, axis='x')
            for bar, val in zip(ax.patches, feat_vals[::-1]):
                ax.text(bar.get_width() + max(feat_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va='center', fontsize=9)
            plt.tight_layout()
            result["feature_importance_fig"] = fig
            plt.close(fig)

    # Try TreeExplainer first (fast, exact for tree models like XGBoost/LightGBM).
    # Falls back to KernelExplainer for non-tree models (slower, uses 50-sample background).
    try:
        try:
            explainer = shap.TreeExplainer(model)
            result["interpretation_type"] = "tree"
        except Exception:
            background = shap.sample(X, min(50, len(X)))
            explainer = shap.KernelExplainer(model.predict, background)
            result["interpretation_type"] = "kernel"

        shap_values = explainer.shap_values(X)
        result["shap_values"] = shap_values

        if create_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X, show=False, plot_type="bar")
            plt.title(f"SHAP Feature Importance: {truncate_partition_name(partition, 30)}")
            plt.tight_layout()
            result["shap_summary_fig"] = fig
            plt.close(fig)

            # Get indices of top N features by mean absolute SHAP value (descending)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-TOP_N_SHAP_DEPENDENCE:][::-1]
            top_features_for_dep = [available_regs[i] for i in top_indices]

            for feat in top_features_for_dep:
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    feat_idx = available_regs.index(feat)
                    shap.dependence_plot(feat_idx, shap_values, X, show=False, ax=ax)
                    plt.title(f"SHAP Dependence: {feat}")
                    plt.tight_layout()
                    result["shap_dependence_figs"][feat] = fig
                    plt.close(fig)
                except Exception:
                    pass

    except Exception as e:
        result["error"] = f"SHAP failed: {str(e)}"

    return result

# COMMAND ----------

def _extract_prophet_coefficients_fallback(model, result):
    """Fallback: extract coefficients from Prophet internals (pre-1.0 or if public API fails)."""
    for reg_name in list(model.extra_regressors.keys()):
        reg_data = model.extra_regressors[reg_name]
        coef = None
        if hasattr(model, "params") and "beta" in model.params:
            # beta = fitted regressor coefficients, indexed by position in train_component_cols
            beta = model.params["beta"]
            # Some Prophet versions store beta as 2D array; extract first row
            if hasattr(beta, "ndim") and beta.ndim > 1:
                beta = beta[0]
            # Map regressor name -> beta index via train_component_cols column order
            col_name = reg_data.get("col_name", reg_name)
            if hasattr(model, "train_component_cols"):
                cols = model.train_component_cols
                if col_name in cols.columns:
                    col_idx = cols.columns.get_loc(col_name)
                    if col_idx < len(beta):
                        coef = float(beta[col_idx])
        result["regressor_coefficients"][reg_name] = {
            "coefficient": coef,
            "mode": reg_data.get("mode", "additive"),
            "prior_scale": reg_data.get("prior_scale"),
        }


def interpret_prophet(
    model: Any,
    model_config: dict,
    partition: str,
    create_plots: bool = True,
) -> dict:
    """
    Extract interpretability for Prophet models.

    Returns dict with:
        - components: dict with trend, seasonality breakdown
        - regressor_coefficients: dict of regressor -> (coef, mode)
        - changepoints: list of detected changepoint dates
        - component_fig: matplotlib figure
    """
    result = {
        "components": {},
        "regressor_coefficients": {},
        "changepoints": [],
        "changepoint_magnitudes": [],
        "component_fig": None,
        "top_regressors": [],
        "error": None,
    }

    try:
        # Try public API first (regressor_coefficients()), fall back to internal arrays.
        # Internal path: beta = regressor coefficients array, delta = additional effects per changepoint.
        # This fallback is needed for older Prophet versions that lack the public API.
        if hasattr(model, "extra_regressors") and model.extra_regressors:
            if hasattr(model, "regressor_coefficients"):
                try:
                    coef_df = model.regressor_coefficients()
                    for _, row in coef_df.iterrows():
                        reg_name = row["regressor"]
                        reg_data = model.extra_regressors.get(reg_name, {})
                        result["regressor_coefficients"][reg_name] = {
                            "coefficient": float(row["coef"]),
                            "mode": reg_data.get("mode", str(row.get("regressor_mode", "additive"))),
                            "prior_scale": reg_data.get("prior_scale"),
                        }
                except Exception:
                    _extract_prophet_coefficients_fallback(model, result)
            else:
                _extract_prophet_coefficients_fallback(model, result)

            # Sort regressors by absolute coefficient
            if result["regressor_coefficients"]:
                sorted_regs = sorted(
                    result["regressor_coefficients"].items(),
                    key=lambda x: abs(x[1]["coefficient"]) if x[1]["coefficient"] else 0,
                    reverse=True
                )
                result["top_regressors"] = [r[0] for r in sorted_regs[:5]]

        # Changepoints
        if hasattr(model, "changepoints") and model.changepoints is not None:
            result["changepoints"] = [str(cp) for cp in model.changepoints.tolist()]

            # Changepoint magnitudes
            if hasattr(model, "params") and "delta" in model.params:
                deltas = model.params["delta"]
                if hasattr(deltas, "tolist"):
                    result["changepoint_magnitudes"] = deltas.tolist()
                else:
                    result["changepoint_magnitudes"] = list(deltas) if deltas is not None else []

        # Component decomposition - requires history
        if hasattr(model, "history") and model.history is not None:
            try:
                # Decompose historical period only (periods=0 = no future dates)
                future = model.make_future_dataframe(periods=0, freq='QS')

                # Prophet's predict() requires regressor + saturation columns even for decomposition
                regressors = model_config.get("regressors", [])
                for reg in regressors:
                    if reg not in future.columns:
                        future[reg] = 0

                future['floor'] = 0
                future['cap'] = model_config.get("cap", 1e9)

                forecast = model.predict(future)

                # Extract components
                result["components"]["trend"] = {
                    "mean": float(forecast["trend"].mean()),
                    "std": float(forecast["trend"].std()),
                    "min": float(forecast["trend"].min()),
                    "max": float(forecast["trend"].max()),
                }

                if "quarterly" in forecast.columns:
                    result["components"]["quarterly_seasonality"] = {
                        "mean": float(forecast["quarterly"].mean()),
                        "amplitude": float(forecast["quarterly"].max() - forecast["quarterly"].min()),
                    }

                # Component plot
                if create_plots:
                    try:
                        fig = model.plot_components(forecast)
                        result["component_fig"] = fig
                    except Exception:
                        pass

            except Exception as e:
                result["error"] = f"Component extraction failed: {str(e)}"

    except Exception as e:
        result["error"] = f"Prophet interpretation failed: {str(e)}"

    return result

# COMMAND ----------

def interpret_statsforecast(
    model: Any,
    model_config: dict,
    model_type: str,
    partition: str
) -> dict:
    """
    Extract interpretability for StatsForecast models (univariate).

    Returns dict with:
        - model_params: extracted parameters
        - description: human-readable description
    """
    result = {
        "model_params": {},
        "description": "",
        "error": None,
    }

    try:
        model_type_lower = model_type.lower()
        hyperparams = model_config.get("hyperparameters", {})

        if "seasonalnaive" in model_type_lower:
            season_length = hyperparams.get("season_length", 4)
            result["model_params"]["season_length"] = season_length
            result["model_params"]["model_class"] = "SeasonalNaive"
            result["description"] = (
                f"SeasonalNaive baseline with season_length={season_length}. "
                f"Forecasts by repeating the value from {season_length} periods ago "
                f"(same quarter last year for quarterly data). No learned parameters."
            )

        elif "autoarima" in model_type_lower:
            season_length = hyperparams.get("season_length", 4)
            result["model_params"]["season_length"] = season_length
            result["model_params"]["model_class"] = "AutoARIMA"

            # Try to extract fitted ARIMA order
            order = None
            seasonal_order = None

            if hasattr(model, "models_") and model.models_:
                try:
                    fitted_model = list(model.models_.values())[0] if isinstance(model.models_, dict) else model.models_[0]
                    if hasattr(fitted_model, "model_"):
                        arima_model = fitted_model.model_
                        order = getattr(arima_model, "order", None)
                        seasonal_order = getattr(arima_model, "seasonal_order", None)
                except Exception:
                    pass

            if order:
                result["model_params"]["order"] = list(order) if hasattr(order, "__iter__") else order
            if seasonal_order:
                result["model_params"]["seasonal_order"] = list(seasonal_order) if hasattr(seasonal_order, "__iter__") else seasonal_order

            order_str = f"order={order}" if order else "order=auto-selected"
            result["description"] = (
                f"AutoARIMA with season_length={season_length}. "
                f"Automatically selects optimal (p,d,q) ARIMA parameters. "
                f"Fitted {order_str}. Univariate model (no external regressors)."
            )
        else:
            result["model_params"]["model_class"] = model_type
            result["description"] = f"StatsForecast model: {model_type}. Univariate baseline model."

    except Exception as e:
        result["error"] = f"StatsForecast interpretation failed: {str(e)}"

    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Load Sample Data for SHAP
# MAGIC
# MAGIC Load the same horizon data as training and convert to pandas for SHAP and Prophet
# MAGIC component analysis. Sample size is capped per partition (via `shap_sample_size` widget)
# MAGIC to keep SHAP computation tractable.

# COMMAND ----------

SELECT_COLUMNS = variable_config["SELECT_COLUMNS"] + [
    "SELLIN_MINUS6_MONTH", "MOF_MINUS6_MONTH", "SR_MINUS6_MONTH", "MOF_PLUS_BUFFER_MINUS6_MONTH"
]
for col in PARTITION_COLS:
    if col not in SELECT_COLUMNS:
        SELECT_COLUMNS.append(col)

df_raw, DATA_VERSION = load_latest_version(spark, DATA_TABLE, SELECT_COLUMNS, is_synthetic_data=IS_SYNTHETIC_DATA)
df_raw = df_raw.withColumn("PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in PARTITION_COLS]))

# Convert to pandas for interpretability calculations
data_pdf = df_raw.toPandas()
print(f"Loaded {len(data_pdf)} rows for interpretability analysis (data version: {DATA_VERSION})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Interpretability Analysis
# MAGIC
# MAGIC Iterates over loaded models and dispatches to the appropriate handler:
# MAGIC - **sklearn_api**: SHAP values (TreeExplainer or KernelExplainer fallback), `feature_importances_`
# MAGIC - **prophet**: Component decomposition, regressor coefficients, changepoint detection
# MAGIC - **statsforecast**: Model parameter extraction, human-readable description

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')

_INTERP_MAX_WORKERS = 2  # Keep low to avoid OOM on serverless (Prophet figures are memory-heavy)

_CREATE_PLOTS = EVAL_SOURCE != "delta"

def _interpret_one(model_info):
    """Run interpretability analysis for a single model (thread-safe)."""
    partition = model_info["PARTITION_COLUMN"]
    model_config = model_info["model_config"] or {}
    model_type = model_info.get("model_type") or model_config.get("model_type", "unknown").lower()
    handler_type = model_info.get("handler_type") or model_config.get("handler_type", "sklearn_api")
    model = model_info["model"]
    is_champion = model_info["is_champion"]

    partition_data = data_pdf[data_pdf["PARTITION_COLUMN"] == partition].copy()
    if len(partition_data) > SHAP_SAMPLE_SIZE:
        partition_data = partition_data.sample(n=SHAP_SAMPLE_SIZE, random_state=42)

    result = {
        "PARTITION_COLUMN": partition,
        "MODEL_USED": model_type.upper(),
        "handler_type": handler_type,
        "is_champion": is_champion,
        "mape": model_info.get("mape"),
        "nested_run_id": model_info.get("nested_run_id"),
        "timestamp": datetime.now(),
    }

    # sklearn_api: SHAP values + feature importance (XGBoost, LightGBM, CatBoost, etc.)
    if handler_type == "sklearn_api":
        interp = interpret_sklearn_api(model, model_config, partition_data, partition, create_plots=_CREATE_PLOTS)
        result["feature_importances"] = interp["feature_importances"]
        result["interpretation_type"] = interp["interpretation_type"]
        result["top_features"] = interp["top_features"]
        result["feature_importance_fig"] = interp["feature_importance_fig"]
        result["shap_summary_fig"] = interp["shap_summary_fig"]
        result["shap_dependence_figs"] = interp["shap_dependence_figs"]
        result["shap_values"] = interp["shap_values"] if LOG_SHAP_VALUES else None
        result["error"] = interp["error"]

    # prophet: regressor coefficients + trend/seasonality decomposition
    elif handler_type == "prophet":
        interp = interpret_prophet(model, model_config, partition, create_plots=_CREATE_PLOTS)
        result["regressor_coefficients"] = interp["regressor_coefficients"]
        result["changepoints"] = interp["changepoints"]
        result["changepoint_magnitudes"] = interp["changepoint_magnitudes"]
        result["components"] = interp["components"]
        result["top_regressors"] = interp["top_regressors"]
        result["component_fig"] = interp["component_fig"]
        result["error"] = interp["error"]

    # statsforecast: univariate baseline (SeasonalNaive, AutoARIMA) -- limited interpretability
    elif handler_type == "statsforecast":
        interp = interpret_statsforecast(model, model_config, model_type, partition)
        result["model_params"] = interp["model_params"]
        result["description"] = interp["description"]
        result["error"] = interp["error"]

    # neuralforecast: deep learning models -- no interpretability available
    elif handler_type == "neuralforecast":
        result["description"] = (
            f"Deep learning model ({model_type}). "
            "NeuralForecast models (NHITS, etc.) are univariate deep learning models. "
            "Feature-level interpretability is not available."
        )

    else:
        result["error"] = f"Unknown handler type: {handler_type}"

    error_str = f"{partition}/{model_type}: {result['error']}" if result.get("error") else None
    return result, error_str

interpretability_results = []
analysis_errors = []

print(f"Running interpretability analysis on {len(model_data)} models ({_INTERP_MAX_WORKERS} threads)...")

with ThreadPoolExecutor(max_workers=_INTERP_MAX_WORKERS) as executor:
    futures = {executor.submit(_interpret_one, m): m for m in model_data}
    for future in as_completed(futures):
        try:
            result, error = future.result()
            interpretability_results.append(result)
            if error:
                analysis_errors.append(error)
            status = "champion" if result["is_champion"] else "candidate"
            print(f"  Done: {truncate_partition_name(result['PARTITION_COLUMN'], 35)} / {result['MODEL_USED'].lower()} ({status})")
        except Exception as exc:
            failed_entry = futures[future]
            partition = failed_entry.get("PARTITION_COLUMN", "unknown")
            msg = f"{truncate_partition_name(partition, 35)}: {exc}"
            analysis_errors.append(msg)
            print(f"  Error: {msg}")

print(f"\nCompleted interpretability analysis for {len(interpretability_results)} models")
if analysis_errors:
    print(f"Warnings/errors: {len(analysis_errors)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Per-Model Visualizations
# MAGIC
# MAGIC Displays SHAP summary plots (sklearn_api), Prophet component decompositions,
# MAGIC and SHAP dependence plots. Scope controlled by `interpretability_scope` widget
# MAGIC (`champions` = champion models only, `all` = all models).

# COMMAND ----------

_SHOW_ALL = INTERPRETABILITY_SCOPE == "all"

def _scope_filter(r):
    """Return True if this result should be displayed (based on scope widget)."""
    return _SHOW_ALL or r["is_champion"]

def _status_label(r):
    return "champion" if r["is_champion"] else "candidate"

# COMMAND ----------

# Native feature importance bar charts (sklearn_api)
sklearn_fi_display = [r for r in interpretability_results if r["handler_type"] == "sklearn_api" and r.get("feature_importance_fig") and _scope_filter(r)]

for r in sklearn_fi_display:
    print(f"\n{truncate_partition_name(r['PARTITION_COLUMN'], 40)} -- {r['MODEL_USED']} ({_status_label(r)})")
    if DISPLAY_PLOTS:
        display(r["feature_importance_fig"])

if not sklearn_fi_display:
    print("No sklearn_api feature importance plots to display.")

# COMMAND ----------

# SHAP summary bar plots (sklearn_api)
sklearn_shap_display = [r for r in interpretability_results if r["handler_type"] == "sklearn_api" and r.get("shap_summary_fig") and _scope_filter(r)]

for r in sklearn_shap_display:
    print(f"\n{truncate_partition_name(r['PARTITION_COLUMN'], 40)} -- {r['MODEL_USED']} ({_status_label(r)})")
    if DISPLAY_PLOTS:
        display(r["shap_summary_fig"])

if not sklearn_shap_display:
    print("No sklearn_api SHAP plots to display.")

# COMMAND ----------

# SHAP dependence plots for top features (sklearn_api)
for r in sklearn_shap_display:
    dep_figs = r.get("shap_dependence_figs", {})
    if dep_figs:
        print(f"\n{truncate_partition_name(r['PARTITION_COLUMN'], 40)} -- {r['MODEL_USED']} dependence plots:")
        for feat, fig in dep_figs.items():
            if DISPLAY_PLOTS:
                display(fig)

# COMMAND ----------

# Prophet component decomposition plots
prophet_display = [r for r in interpretability_results if r["handler_type"] == "prophet" and r.get("component_fig") and _scope_filter(r)]

for r in prophet_display:
    print(f"\n{truncate_partition_name(r['PARTITION_COLUMN'], 40)} -- PROPHET ({_status_label(r)})")
    if DISPLAY_PLOTS:
        display(r["component_fig"])

if not prophet_display:
    print("No Prophet component plots to display.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # MLflow Logging
# MAGIC
# MAGIC Log interpretability artifacts to the training run (same parent run as notebooks 02 and 03):
# MAGIC 1. Parent run: summary metrics, comparison charts
# MAGIC 2. Nested (child) runs: per-model charts, feature importances, SHAP values

# COMMAND ----------

# Log per-model interpretability artifacts directly to child (nested) runs.
# This runs BEFORE the parent-run summary cell so that figures are freed
# one-by-one here rather than all at once when mlflow.start_run() opens the
# parent run. Keeping ~1000 figures alive until the summary cell causes
# DRIVER_UNREACHABLE (OOM) with large partition counts.
updated_runs = 0
for result in interpretability_results:
    nested_run_id = result.get("nested_run_id")
    if not nested_run_id:
        continue

    try:
        with mlflow.start_run(run_id=nested_run_id):
            # Feature importances as metrics (sklearn_api)
            for feat, imp in result.get("feature_importances", {}).items():
                feat_safe = sanitize_metric_name(feat)
                if is_valid_metric(imp):
                    mlflow.log_metric(f"fi_{feat_safe}", imp)

            # Native feature importance chart
            if result.get("feature_importance_fig"):
                mlflow.log_figure(result["feature_importance_fig"], "interpretability/feature_importance.png")

            # SHAP summary chart
            if result.get("shap_summary_fig"):
                mlflow.log_figure(result["shap_summary_fig"], "interpretability/shap_summary.png")

            # SHAP dependence plots
            for feat, fig in result.get("shap_dependence_figs", {}).items():
                feat_safe = sanitize_metric_name(feat, max_len=30)
                mlflow.log_figure(fig, f"interpretability/shap_dep_{feat_safe}.png")

            # Prophet component decomposition
            if result.get("component_fig"):
                mlflow.log_figure(result["component_fig"], "interpretability/prophet_components.png")

            # Regressor coefficients (prophet)
            for reg, coef_info in result.get("regressor_coefficients", {}).items():
                reg_safe = sanitize_metric_name(reg)
                if coef_info.get("coefficient") is not None and is_valid_metric(coef_info["coefficient"]):
                    mlflow.log_metric(f"coef_{reg_safe}", coef_info["coefficient"])

            mlflow.set_tag("interpretability_complete", "true")

        updated_runs += 1
    except Exception as e:
        print(f"  Warning: could not update nested run {nested_run_id}: {e}")
    finally:
        # Close figures immediately after logging to free driver memory.
        # SHAP dependence figures hold internal references to shap_values arrays,
        # so closing them also releases those arrays.
        for fig_key in ["shap_summary_fig", "feature_importance_fig", "component_fig"]:
            if result.get(fig_key):
                plt.close(result[fig_key])
                result[fig_key] = None
        for fig in result.get("shap_dependence_figs", {}).values():
            plt.close(fig)
        result["shap_dependence_figs"] = {}

print(f"Updated {updated_runs} nested MLflow runs with interpretability artifacts")

# COMMAND ----------

import gc
gc.collect()

# COMMAND ----------

# Resume the training parent run (same as notebooks 02 and 03).
# Runs after per-model logging so figures are already freed before
# mlflow.start_run() allocates additional client state.
mlflow.end_run()
print(f"Resuming training run: {training_run_id}")

with mlflow.start_run(run_id=training_run_id):
    # Log interpretability summary to parent run
    sklearn_models = [r for r in interpretability_results if r["handler_type"] == "sklearn_api"]
    prophet_models = [r for r in interpretability_results if r["handler_type"] == "prophet"]
    statsforecast_models = [r for r in interpretability_results if r["handler_type"] == "statsforecast"]

    mlflow.log_metric("interp_models_analyzed", len(interpretability_results))
    mlflow.log_metric("interp_n_sklearn", len(sklearn_models))
    mlflow.log_metric("interp_n_prophet", len(prophet_models))
    mlflow.log_metric("interp_n_statsforecast", len(statsforecast_models))

    if sklearn_models:
        feature_counts = [len(r.get("feature_importances", {})) for r in sklearn_models if r.get("feature_importances")]
        if feature_counts:
            mlflow.log_metric("interp_avg_features_used", np.mean(feature_counts))

    mlflow.set_tag("interpretability_complete", "true")

    # Run description (renders at top of MLflow run page)
    run_desc = (
        f"**Interpretability** | {len(interpretability_results)} models analyzed\n\n"
        f"sklearn: {len(sklearn_models)} | prophet: {len(prophet_models)} | "
        f"statsforecast: {len(statsforecast_models)}"
    )
    mlflow.set_tag("mlflow.note.content", run_desc)

print(f"Logged interpretability summary to training run: {training_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Write to Delta Table
# MAGIC
# MAGIC Persist interpretability results (feature importances, SHAP metadata, Prophet components,
# MAGIC changepoints, StatsForecast parameters) to the `_interpretability` Delta table
# MAGIC for downstream dashboards and auditing.

# COMMAND ----------

# Prepare DataFrame for Delta
delta_records = []

for result in interpretability_results:
    record = {
        "PARTITION_COLUMN": result["PARTITION_COLUMN"],
        "MODEL_USED": result["MODEL_USED"],
        "handler_type": result["handler_type"],
        "is_champion": result["is_champion"],
        "mape": float(result["mape"]) if result.get("mape") is not None else None,
        "nested_run_id": result.get("nested_run_id"),
        "interpretability_run_id": training_run_id,
        "training_run_id": training_run_id,
        "timestamp": result["timestamp"],

        # Feature importances (JSON)
        "feature_importances": json.dumps(result.get("feature_importances", {})),
        "interpretation_type": result.get("interpretation_type"),
        "top_features": json.dumps(result.get("top_features", [])),

        # Prophet-specific
        "regressor_coefficients": json.dumps(result.get("regressor_coefficients", {})),
        "changepoints": json.dumps(result.get("changepoints", [])),
        "components": json.dumps(result.get("components", {})),
        "top_regressors": json.dumps(result.get("top_regressors", [])),

        # StatsForecast-specific
        "model_params": json.dumps(result.get("model_params", {})),
        "model_description": result.get("description"),

        # Metadata
        "error": result.get("error"),
        "DATA_TABLE": DATA_TABLE,
        "DATA_VERSION": DATA_VERSION,
    }
    delta_records.append(record)

interp_df = pd.DataFrame(delta_records)

# COMMAND ----------

# Write to Delta
interp_spark_df = spark.createDataFrame(interp_df)

# Check if table exists
table_exists = spark.catalog.tableExists(INTERPRETABILITY_TABLE)

if table_exists:
    # Append with schema merge
    interp_spark_df.write.format("delta").option("mergeSchema", "true").mode("append").saveAsTable(INTERPRETABILITY_TABLE)
    print(f"Appended {len(interp_df)} records to {INTERPRETABILITY_TABLE}")
else:
    # Create new table
    interp_spark_df.write.format("delta").mode("overwrite").saveAsTable(INTERPRETABILITY_TABLE)
    print(f"Created {INTERPRETABILITY_TABLE} with {len(interp_df)} records")

if ENABLE_ICEBERG:
    enable_iceberg(spark, INTERPRETABILITY_TABLE)
    print("Enabled Iceberg compatibility")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Interpretability Summary
# MAGIC
# MAGIC Final summary showing models analyzed, breakdown by handler type (sklearn_api/prophet/statsforecast),
# MAGIC and any analysis errors encountered.

# COMMAND ----------

print(f"Interpretability Analysis Complete")
print(f"{'='*60}")
print(f"Models analyzed: {len(interpretability_results)}")
print(f"  - Champions: {len([r for r in interpretability_results if r['is_champion']])}")
print(f"  - Candidates: {len([r for r in interpretability_results if not r['is_champion']])}")
print(f"Scope: {INTERPRETABILITY_SCOPE}")
print(f"Output table: {INTERPRETABILITY_TABLE}")
print(f"Training run: {training_run_id}")
if analysis_errors:
    print(f"Errors: {len(analysis_errors)}")
print(f"{'='*60}")

# COMMAND ----------

# Summary by handler type
summary_df = pd.DataFrame([
    {
        "PARTITION_COLUMN": truncate_partition_name(r["PARTITION_COLUMN"], 30),
        "MODEL_USED": r["MODEL_USED"],
        "handler_type": r["handler_type"],
        "is_champion": r["is_champion"],
        "mape": round(r["mape"], 2) if r.get("mape") else None,
        "n_features": len(r.get("feature_importances", {})) if r.get("feature_importances") else None,
        "n_regressors": len(r.get("regressor_coefficients", {})) if r.get("regressor_coefficients") else None,
        "n_changepoints": len(r.get("changepoints", [])) if r.get("changepoints") else None,
        "error": "Yes" if r.get("error") else None,
    }
    for r in interpretability_results
])

if DISPLAY_PLOTS:
    display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top Features by Handler Type
# MAGIC
# MAGIC Aggregated feature importances across all sklearn_api models (bar chart of top 10 features),
# MAGIC plus a Prophet regressor coefficient heatmap showing how coefficients vary across partitions.

# COMMAND ----------

# sklearn_api: Aggregate feature importances across all models
sklearn_results = [r for r in interpretability_results if r["handler_type"] == "sklearn_api" and r.get("feature_importances")]

if sklearn_results:
    # Aggregate feature importances
    feature_totals = {}
    feature_counts = {}
    for r in sklearn_results:
        for feat, imp in r.get("feature_importances", {}).items():
            feature_totals[feat] = feature_totals.get(feat, 0) + imp
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    feature_avg = {f: feature_totals[f] / feature_counts[f] for f in feature_totals}
    sorted_features = sorted(feature_avg.items(), key=lambda x: -x[1])[:TOP_N_FEATURES]

    fig, ax = plt.subplots(figsize=(12, 6))
    feat_names = [feat_item[0] for feat_item in sorted_features]
    feat_vals = [feat_item[1] for feat_item in sorted_features]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
    bars = ax.barh(feat_names[::-1], feat_vals[::-1], color=colors[::-1])
    ax.set_xlabel("Average Importance")
    ax.set_title(f"Aggregated Feature Importance ({len(sklearn_results)} sklearn_api models)")
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, feat_vals[::-1]):
        ax.text(bar.get_width() + max(feat_vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    if DISPLAY_PLOTS:
        display(fig)
    plt.close(fig)
else:
    print("No sklearn_api models with feature importances.")

# COMMAND ----------

# Prophet: Regressor coefficients
prophet_results = [r for r in interpretability_results if r["handler_type"] == "prophet" and r.get("regressor_coefficients")]

if prophet_results:
    # Build a coefficient matrix: partitions x regressors
    coef_data = {}
    all_regs = set()
    for r in prophet_results:
        partition = truncate_partition_name(r["PARTITION_COLUMN"], 25)
        coeffs = r.get("regressor_coefficients", {})
        coef_data[partition] = {}
        for reg, info in coeffs.items():
            if info.get("coefficient") is not None:
                coef_data[partition][reg] = info["coefficient"]
                all_regs.add(reg)

    if all_regs:
        all_regs = sorted(all_regs)
        coef_df = pd.DataFrame(coef_data).T.reindex(columns=all_regs).fillna(0)

        # Heatmap of coefficients across partitions
        fig, ax = plt.subplots(figsize=(max(10, len(all_regs) * 1.2), max(4, len(coef_df) * 0.5)))
        sns.heatmap(coef_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Coefficient"})
        ax.set_title("Prophet Regressor Coefficients by Partition")
        ax.set_ylabel("Partition")
        ax.set_xlabel("Regressor")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if DISPLAY_PLOTS:
            display(fig)
        plt.close(fig)

        # Per-partition bar chart for top partitions (by max absolute coefficient)
        max_abs = coef_df.abs().max(axis=1).sort_values(ascending=False)
        top_partitions = max_abs.head(TOP_N_PROPHET_PARTITIONS).index.tolist()

        n_plots = len(top_partitions)
        if n_plots > 0:
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), sharey=False)
            if n_plots == 1:
                axes = [axes]

            for ax, partition in zip(axes, top_partitions):
                row = coef_df.loc[partition].sort_values()
                colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in row.values]
                row.plot.barh(ax=ax, color=colors)
                ax.set_title(partition, fontsize=10)
                ax.set_xlabel("Coefficient")
                ax.axvline(0, color="black", linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='x')

            plt.suptitle("Prophet Regressor Coefficients (top partitions by magnitude)", fontsize=12, y=1.02)
            plt.tight_layout()
            if DISPLAY_PLOTS:
                display(fig)
            plt.close(fig)
else:
    print("No Prophet models with regressor coefficients.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Heatmap (sklearn_api)
# MAGIC
# MAGIC Cross-partition heatmap showing how feature importance varies across market segments
# MAGIC for sklearn_api champion models. Highlights which features drive predictions in each segment.

# COMMAND ----------

# Cross-partition feature importance heatmap
sklearn_champs = [r for r in interpretability_results if r["handler_type"] == "sklearn_api" and r["is_champion"] and r.get("feature_importances")]

if sklearn_champs:
    fi_data = {}
    all_features = set()
    for r in sklearn_champs:
        partition = truncate_partition_name(r["PARTITION_COLUMN"], 25)
        fi_data[partition] = r["feature_importances"]
        all_features.update(r["feature_importances"].keys())

    all_features = sorted(all_features)
    fi_df = pd.DataFrame(fi_data).T.reindex(columns=all_features).fillna(0)

    fig, ax = plt.subplots(figsize=(max(10, len(all_features) * 1.2), max(4, len(fi_df) * 0.5)))
    sns.heatmap(fi_df, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Importance"})
    ax.set_title("Feature Importance Across Partitions (sklearn_api champions)")
    ax.set_ylabel("Partition")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if DISPLAY_PLOTS:
        display(fig)
    plt.close(fig)
else:
    print("No sklearn_api champion models with feature importances.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet Changepoint Analysis
# MAGIC
# MAGIC Timeline visualization of detected trend changepoints across Prophet champion models.
# MAGIC Marker size reflects magnitude, color indicates direction (green = positive shift, red = negative).

# COMMAND ----------

# Changepoint timeline visualization
prophet_cp = [r for r in interpretability_results if r["handler_type"] == "prophet" and r.get("changepoints") and r["is_champion"]]

if prophet_cp:
    fig, ax = plt.subplots(figsize=(14, max(3, len(prophet_cp) * 0.6)))

    for i, r in enumerate(prophet_cp):
        partition = truncate_partition_name(r["PARTITION_COLUMN"], 25)
        changepoints = r["changepoints"]
        magnitudes = r.get("changepoint_magnitudes", [])

        cp_dates = pd.to_datetime(changepoints, errors="coerce")
        cp_dates = cp_dates.dropna()

        if len(cp_dates) > 0:
            # Size markers by magnitude if available
            if magnitudes and len(magnitudes) >= len(cp_dates):
                # scale delta magnitudes to marker sizes; clamp [20, 200] so markers stay visible
                sizes = [max(20, min(200, abs(m) * 500)) for m in magnitudes[:len(cp_dates)]]
                colors_cp = ["#e74c3c" if m < 0 else "#2ecc71" for m in magnitudes[:len(cp_dates)]]
            else:
                sizes = [40] * len(cp_dates)
                colors_cp = ["#3498db"] * len(cp_dates)

            ax.scatter(cp_dates, [i] * len(cp_dates), s=sizes, c=colors_cp, alpha=0.7, edgecolors="black", linewidth=0.5)

    ax.set_yticks(range(len(prophet_cp)))
    ax.set_yticklabels([truncate_partition_name(r["PARTITION_COLUMN"], 25) for r in prophet_cp])
    ax.set_xlabel("Date")
    ax.set_title("Prophet Changepoints by Partition (size = magnitude, green = positive, red = negative)")
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if DISPLAY_PLOTS:
        display(fig)
    plt.close(fig)
else:
    print("No Prophet champion models with changepoint data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC **Next:** Review the interpretability results in MLflow and the Delta table.
