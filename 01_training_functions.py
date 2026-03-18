# Databricks notebook source
# MAGIC %md
# MAGIC # Training Functions
# MAGIC
# MAGIC Core training logic for the capacity forecasting pipeline. This file is loaded
# MAGIC via `%run ./01_training_functions` from `02_model_training.py`.
# MAGIC
# MAGIC ## File Layout
# MAGIC
# MAGIC The file is organized so the most important code comes first:
# MAGIC
# MAGIC 1. **Imports** - Package installation check, standard imports, local imports
# MAGIC 2. **Dispatcher** - `train_model()` routes to the correct handler based on config
# MAGIC 3. **Prophet Handler** - `_train_prophet()` -- the original AMD Prophet training logic
# MAGIC 4. **Sklearn Handler** - `_train_sklearn_api()` -- the original AMD XGBoost logic, generalized
# MAGIC 5. **Predictor Wrappers** - Thin wrappers for inference-time prediction
# MAGIC 6. **Additional Handlers** - StatsForecast, NeuralForecast serialization stubs, global NF training
# MAGIC
# MAGIC ## Train/Val/Test Split
# MAGIC
# MAGIC - Training data: `allData[:-validation_horizon-prediction_horizon]`
# MAGIC - Validation data: `allData[-validation_horizon-prediction_horizon:-prediction_horizon]`
# MAGIC - Test data: `allData[-prediction_horizon:]`
# MAGIC
# MAGIC Hyperparameter tuning evaluates on VALIDATION data only (no data leakage).
# MAGIC `03_model_evaluation.py` and `04_inference.py` use the full inference metric
# MAGIC (APE + indicators + remove_unwanted_rows) for champion selection and reporting.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Code Mapping: Original -> Refactored
# MAGIC
# MAGIC This notebook contains the same training logic as the original
# MAGIC `GENERATE_PREDICTIONS_ALL_QUARTER.ipynb`, reorganized into testable functions
# MAGIC with config-driven parameters.
# MAGIC
# MAGIC ### Where did the original code go?
# MAGIC
# MAGIC | Original AMD Code | New Location |
# MAGIC |---|---|
# MAGIC | `set_market_segments_model_params()` (hardcoded segment dicts) | `configs/model_config.yaml` + `configs/variable_config.yaml` |
# MAGIC | `feature_selection()` | `modeling_dev/modeling_utils.py::feature_selection()` (identical logic) |
# MAGIC | `adjust_regressors()` | `modeling_dev/modeling_utils.py::adjust_regressors()` (identical logic) |
# MAGIC | `if PREFERRED_MODEL == 'PROPHET':` block | `_train_prophet()` in this file (called via `train_model()` dispatcher) |
# MAGIC | `if PREFERRED_MODEL == 'XGBOOST':` block | `_train_sklearn_api()` in this file (called via `train_model()` dispatcher) |
# MAGIC | `FORECASTINGMODEL_XGBOOST_PROPHET` class | `ForecastingModel` in `02_model_training.py` |
# MAGIC | `calculate_indicators()` | `modeling_dev/metric_utils.py::calculate_indicators()` |
# MAGIC | `calculate_mape()` | `modeling_dev/metric_utils.py::calculate_mape()` |
# MAGIC | `remove_unwanted_rows()` | `modeling_dev/metric_utils.py::remove_unwanted_rows()` |
# MAGIC | `categorize_percentage_difference()` | `modeling_dev/metric_utils.py::categorize_percentage_difference()` |
# MAGIC | `make_verification_columns()` | `modeling_dev/metric_utils.py::make_verification_columns()` |
# MAGIC | Column renames (`MOF_0_MONTH` -> `MOF`, etc.) | `modeling_dev/modeling_utils.py::apply_standard_renames()` |
# MAGIC | Demand signal fallback (`if demand_signal not in columns`) | `modeling_dev/modeling_utils.py::ensure_demand_signals()` |
# MAGIC | SELLIN exclusion for MOBILE/AIB/EMBEDDED | `modeling_dev/modeling_utils.py::filter_regressors()` + YAML config |
# MAGIC | `param_lookup.pkl` (pickled config) | `configs/model_config.yaml` (human-readable) |
# MAGIC
# MAGIC ### What's net new? (not in the original)
# MAGIC
# MAGIC | Feature | Purpose |
# MAGIC |---|---|
# MAGIC | **Grid search for Prophet** | Original used fixed `train_params`. Now optionally searches over changepoint/seasonality parameters. |
# MAGIC | **Generic sklearn dispatcher** | Original only supported XGBoost. Now any `fit(X,y)/predict(X)` model works via YAML config (LightGBM, CatBoost, etc.). |
# MAGIC | **`train_model()` dispatcher** | Replaces the `if PREFERRED_MODEL == ...` branching. Adding a new sklearn model requires only a YAML entry. |
# MAGIC | **Predictor wrappers** | Thin classes (`ProphetPredictor`, `SklearnPredictor`) that standardize inference across model types. |
# MAGIC | **StatsForecast handler** | Univariate baselines (SeasonalNaive, AutoARIMA) for sanity-checking regressor models. |
# MAGIC | **NeuralForecast handler** | Deep learning (NHITS, RNN, AutoLSTM). Global training on driver via `train_neuralforecast_global()`. |
# MAGIC | **Per-segment config overrides** | YAML can override grid search, feature count, etc. per segment. Original was one-size-fits-all. |
# MAGIC | **sklearn `RandomizedSearchCV`** | Original XGBoost used a manual `itertools.product` loop. Now uses sklearn's built-in search with `PredefinedSplit`. |
# MAGIC | **`ffill/bfill` on Prophet future regressors** | Defensive fill for missing regressor values in future periods. Original did not do this. |

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

# MAGIC %md
# MAGIC We may be running this notebook from inside notebook 02 where we would have already
# MAGIC installed packages and restarted Python. Prevent duplication by checking whether
# MAGIC select packages are installed.

# COMMAND ----------

# Auto-install dependencies only if not already available (allows %run without restart issues)
def _check_and_install():
    import subprocess
    import sys
    # Check a few key packages to see if install is needed
    for pkg in ['prophet', 'xgboost', 'sklearn']:
        try:
            __import__(pkg)
        except ImportError:
            print("Installing dependencies from requirements.txt...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'])
            return True
    return False

if _check_and_install():
    dbutils.library.restartPython()

# COMMAND ----------

# stdlib
import gc
import sys
import os
import io
import json
import shutil
import tarfile
import tempfile
import itertools
import importlib
from typing import Tuple

# third-party
import numpy as np
import pandas as pd
import yaml
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit

# local
notebook_path = os.getcwd()
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

with open("configs/variable_config.yaml", "r") as f:
    VAR_CONFIG = yaml.safe_load(f)
DISPLAY_COLUMNS_BASE = VAR_CONFIG.get("DISPLAY_COLUMNS_BASE", [])

from modeling_dev.modeling_utils import (
    load_segment_config,
    get_segment_training_config,
    feature_selection,
    adjust_regressors,
    ensure_demand_signals,
    split_train_val_test,
    apply_standard_renames,
    extract_horizons,
    get_feature_count,
    filter_regressors,
)
from modeling_dev.metric_utils import calculate_mape, calculate_ape, remove_unwanted_rows, calculate_indicators
from modeling_dev.data_utils import prepare_training_data


def _error_result(partition, model_type, error_msg):
    """Build a standardized error row for failed training attempts."""
    return {
        "PARTITION_COLUMN": partition, "MODEL_USED": model_type.upper(),
        "model_binary": None, "model_config": None, "mape": None,
        "sample_input": None, "predictions_sample": None,
        "training_metrics": None, "used_params": None, "error": error_msg,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training Dispatcher
# MAGIC
# MAGIC Entry point for all model training. Receives a model type string and routes to
# MAGIC the appropriate handler function.
# MAGIC
# MAGIC **Handler types:**
# MAGIC - `prophet` - Facebook Prophet with external regressors (original AMD model)
# MAGIC - `sklearn_api` - Any model with `fit(X, y)` / `predict(X)` (XGBoost, LightGBM, CatBoost, etc.)
# MAGIC - `statsforecast` - Nixtla univariate models (SeasonalNaive, AutoARIMA)
# MAGIC
# MAGIC To add a new sklearn-compatible model, just add to `model_config.yaml` -- no code changes needed.

# COMMAND ----------

def train_model(
    model_type: str,
    df: pd.DataFrame,
    segment: str,
    seg_config: dict,
    config: dict
) -> Tuple:
    """
    Generic model training dispatcher. Routes to appropriate handler based on
    the "type" field in each model's config entry.

    Replaces the original monolithic if/elif chain:
        if PREFERRED_MODEL == 'PROPHET': ...
        elif PREFERRED_MODEL == 'XGBOOST': ...
    Now config["models"]["xgboost"]["type"] = "sklearn_api" routes to
    _train_sklearn_api(), etc.

    Note: NeuralForecast models bypass this dispatcher entirely -- they are
    trained as global models on the driver via train_neuralforecast_global().

    To add a new sklearn-compatible model:
        1. Add entry to models section in model_config.yaml
        2. Include in model_types widget (e.g., "prophet,xgboost,lightgbm")

    To add a model with a different interface:
        1. Write a new _train_xxx() handler function
        2. Add the handler type to the dispatcher below

    Args:
        model_type: Model name (e.g., "prophet", "xgboost", "lightgbm")
        df: DataFrame with segment data
        segment: Segment name
        seg_config: Segment-specific config
        config: Global config dict

    Returns:
        Tuple of (model, predictions_df, metrics_dict, used_params_dict)
    """
    model_type_lower = model_type.lower()
    model_conf = config.get("models", {}).get(model_type_lower, {})

    # Determine handler type from config, with fallback for backwards compatibility
    if model_conf:
        handler = model_conf.get("type", "sklearn_api")
    else:
        # Fallback: prophet uses prophet handler, everything else uses sklearn_api
        handler = "prophet" if model_type_lower == "prophet" else "sklearn_api"

    if handler == "prophet":
        return _train_prophet(df, segment, seg_config, config, model_type_lower, model_conf)
    elif handler == "sklearn_api":
        return _train_sklearn_api(df, segment, seg_config, config, model_type_lower, model_conf)
    elif handler == "statsforecast":
        return _train_statsforecast(df, segment, seg_config, config, model_type_lower, model_conf)
    else:
        raise ValueError(f"Unknown model handler type: {handler}. Supported: prophet, sklearn_api, statsforecast")

# COMMAND ----------

# MAGIC %md
# MAGIC # Prophet Handler
# MAGIC
# MAGIC This is the original AMD Prophet model, migrated from `GENERATE_PREDICTIONS_ALL_QUARTER.ipynb`.
# MAGIC
# MAGIC **Original code location:** `GENERATE_PREDICTIONS_ALL_QUARTER.ipynb`, Cell 12
# MAGIC (`FORECASTINGMODEL_XGBOOST_PROPHET.predict()`, the `if PREFERRED_MODEL == 'PROPHET':` branch)
# MAGIC
# MAGIC **What changed vs. the original:**
# MAGIC - Extracted from the monolithic predict() method into a standalone function
# MAGIC - Grid search over Prophet hyperparameters is NEW (original used fixed train_params)
# MAGIC - Future dataframe regressors now get ffill/bfill (defensive, original did not)
# MAGIC - Everything else (regressor adjustment, RFE feature selection, cap/floor, quarterly
# MAGIC   seasonality, column renames) is identical to the original
# MAGIC
# MAGIC **Training flow (same as original):**
# MAGIC 1. Rename columns to Prophet format (QUARTER_PARSED -> ds, BB -> y)
# MAGIC 2. Adjust regressors (replace values below 10% of main regressor)
# MAGIC 3. Exclude SELLIN for MOBILE, AIB, EMBEDDED segments
# MAGIC 4. RFE feature selection to pick top regressors
# MAGIC 5. Replace zero regressors with BILLING_MOVING_AVERAGE_2Q
# MAGIC 6. Set cap (10x max BB) and floor (0) for logistic growth
# MAGIC 7. Fit Prophet with quarterly seasonality (period=91.3125 days, Fourier order=5)
# MAGIC 8. Grid search (if enabled) over changepoint_prior_scale, seasonality_prior_scale,
# MAGIC    seasonality_mode, n_changepoints
# MAGIC 9. Return model, predictions, validation MAPE, and parameters

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet Helper Functions

# COMMAND ----------

def _build_prophet_model(params, used_regressors, prophet_settings, prophet_data):
    """Build, configure, and fit a Prophet model.

    Args:
        params: Dict with changepoint_prior_scale, seasonality_prior_scale,
                seasonality_mode, n_changepoints
        used_regressors: Dict/iterable of regressor names
        prophet_settings: Dict with growth, uncertainty_samples, QUARTERLY_SEASONALITY
        prophet_data: DataFrame with ds, y, floor, cap, and regressor columns

    Returns:
        Fitted Prophet model
    """
    model = Prophet(
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.001),
        seasonality_prior_scale=params.get("seasonality_prior_scale", 0.01),
        seasonality_mode=params.get("seasonality_mode", "additive"),
        n_changepoints=params.get("n_changepoints", 25),
        growth=prophet_settings.get("growth", "logistic"),
        uncertainty_samples=prophet_settings.get("uncertainty_samples", 0),
    )

    for regressor in used_regressors:
        model.add_regressor(regressor)

    quarterly = prophet_settings.get("QUARTERLY_SEASONALITY", {})
    model.add_seasonality(
        name="quarterly",
        period=quarterly.get("period", 91.3125),
        fourier_order=quarterly.get("fourier_order", 5),
    )

    fit_cols = ["ds", "y", "floor", "cap"] + list(used_regressors)
    model.fit(prophet_data[fit_cols])
    return model


def _build_prophet_future(model, periods, allData, used_regressors, cap):
    """Build future dataframe, merge regressors, and predict.

    Args:
        model: Fitted Prophet model
        periods: Number of future periods (validation_horizon + prediction_horizon)
        allData: Full DataFrame with ds and regressor columns
        used_regressors: Iterable of regressor column names
        cap: Logistic growth cap value

    Returns:
        Prophet forecast DataFrame
    """
    future = model.make_future_dataframe(periods=periods, freq="Q")
    regressor_list = list(used_regressors)
    future = pd.merge(future, allData[["ds"] + regressor_list], on="ds", how="left")
    future[regressor_list] = future[regressor_list].ffill().bfill()
    future["floor"] = 0
    future["cap"] = cap
    return model.predict(future)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prophet Training Function

# COMMAND ----------

def _train_prophet(
    df: pd.DataFrame,
    segment: str,
    seg_config: dict,
    config: dict,
    model_type: str,
    model_conf: dict
) -> Tuple[Prophet, pd.DataFrame, dict, dict]:
    """
    Train Prophet model for a single segment.

    Args:
        df: DataFrame with segment data
        segment: Segment name (e.g., 'MOBILE', 'SERVER/WORKSTATION')
        seg_config: Segment-specific config from load_segment_config()
        config: Global config dict
        model_type: Model type string (e.g., "prophet")
        model_conf: Model-specific config from config["models"]["prophet"]

    Returns:
        model: Trained Prophet model
        predictions: DataFrame with predictions
        metrics: Dict with mape, validation_horizon, etc.
        used_params: Dict of parameters used
    """
    regressors_weights = seg_config["regressors_weights"]
    all_regressors = seg_config["all_regressors"]
    train_params = model_conf.get("hyperparameters", {})

    # Data prep with Prophet column names (ds, y)
    allData = df.rename(columns={'QUARTER_PARSED': 'ds', 'BB': 'y'})
    allData, validation_horizon, prediction_horizon = prepare_training_data(allData, sort_col='ds')

    min_rows = validation_horizon + prediction_horizon + 1
    if len(allData) < min_rows:
        raise ValueError(
            f"Insufficient training data: {len(allData)} rows, need at least {min_rows} "
            f"(val={validation_horizon}, pred={prediction_horizon})"
        )

    allData = adjust_regressors(allData, all_regressors, 'BILLING_MOVING_AVERAGE_2Q', segment)
    regressors_to_use = filter_regressors(all_regressors, segment, config.get("EXCLUDE_SELLIN_PROPHET", []))

    # Feature selection
    common_columns = ['MARKET_SEGMENT', 'QUARTER']
    target_column = ['y']
    n_features = get_feature_count(config, segment)

    top_regressors = feature_selection(
        allData, validation_horizon, prediction_horizon,
        common_columns, regressors_to_use, target_column, n_features
    )

    if not top_regressors:
        raise ValueError(f"No features selected for segment {segment}")

    # Partition regressors into selected (used in training) vs. dropped (kept for display only)
    remaining_regressors = [reg for reg in all_regressors if reg not in top_regressors]
    used_regressors = {key: regressors_weights[key] for key in top_regressors if key in regressors_weights}
    not_used_regressors = {key: regressors_weights[key] for key in remaining_regressors if key in regressors_weights}

    # Prepare training data (drop NaN target rows -- future periods may lack actuals)
    prophet_data = allData.iloc[:-(validation_horizon + prediction_horizon)].copy()
    prophet_data = prophet_data.dropna(subset=['y'])

    # Replace zero regressor values with MA baseline to avoid division issues in Prophet
    for reg in used_regressors.keys():
        prophet_data.loc[prophet_data[reg] == 0, reg] = prophet_data['BILLING_MOVING_AVERAGE_2Q']

    # Cap for logistic growth: 10x max observed value (headroom for future growth)
    cap = prophet_data['y'].max() * 10
    prophet_data['floor'] = 0
    prophet_data['cap'] = cap

    prophet_settings = model_conf.get("settings", {})

    # --------------------------------------------------------------------------
    # Prophet Model Training
    # Grid search controlled by segment_training_config in YAML.
    #
    # METRIC ALIGNMENT: Grid search uses the SAME metric as inference:
    # row-weighted APE after remove_unwanted_rows(). This ensures hyperparameter
    # selection optimizes for the same metric reported in inference.
    # --------------------------------------------------------------------------
    seg_training_config = get_segment_training_config(config, segment)
    USE_PROPHET_GRID_SEARCH = seg_training_config["grid_search"].get("prophet", True)

    # Grid search tracking (populated below if grid search runs)
    gs_combos = 0
    gs_combos_tried = 0
    gs_failures = 0
    gs_best_ape = np.nan

    model = None

    if USE_PROPHET_GRID_SEARCH:
        # Grid search over Prophet hyperparameters
        # Evaluates on VALIDATION data only (proper train/val split, no data leakage)
        prophet_grid = model_conf.get("grid", {
            "changepoint_prior_scale": [0.001, 0.01, 0.1, 0.5],
            "seasonality_prior_scale": [0.01, 0.1, 1.0],
            "seasonality_mode": ["additive", "multiplicative"],
            "n_changepoints": [10, 25],
        })

        # Slice validation window: last (val_horizon + pred_horizon) rows before test split
        val_start = -(validation_horizon + prediction_horizon)
        val_end = -prediction_horizon if prediction_horizon > 0 else None
        val_data = allData.iloc[val_start:val_end].copy()

        best_params = None
        best_model = None
        best_avg_ape = np.inf

        def prophet_param_combos(g, n_iter=None, random_state=42):
            """Yield parameter dicts. If n_iter < total combos, randomly sample."""
            keys = list(g.keys())
            all_combos = list(itertools.product(*[g[k] for k in keys]))
            if n_iter and 0 < n_iter < len(all_combos):
                rng = np.random.RandomState(random_state)
                indices = rng.choice(len(all_combos), size=n_iter, replace=False)
                all_combos = [all_combos[i] for i in indices]
            for values in all_combos:
                yield dict(zip(keys, values))

        # Count combinations for logging
        total_combos = _count_grid_combos(prophet_grid)
        n_iter = model_conf.get("n_iter", 0)
        n_trying = min(n_iter, total_combos) if n_iter > 0 else total_combos
        if n_iter > 0 and n_iter < total_combos:
            print(f"  Prophet: randomly sampling {n_trying} of {total_combos} parameter combinations")
        else:
            print(f"  Prophet: trying {total_combos} parameter combinations")

        grid_failures = 0
        for params in prophet_param_combos(prophet_grid, n_iter=n_iter):
            try:
                # Build candidate model with these hyperparameters
                candidate = _build_prophet_model(params, used_regressors, prophet_settings, prophet_data)

                # Generate predictions on validation window
                forecast = _build_prophet_future(candidate, validation_horizon + prediction_horizon, allData, used_regressors, cap)

                # Score: mean APE on validation rows
                eval_df = val_data.copy()
                eval_df = pd.merge(eval_df, forecast[['ds', 'yhat']], on='ds', how='left')
                eval_df = eval_df.rename(columns={'y': 'BILLING', 'yhat': 'BILLING_PREDICTED'})
                eval_df = apply_standard_renames(eval_df)

                eval_df['APE'] = calculate_ape(eval_df['BILLING'], eval_df['BILLING_PREDICTED'])
                valid_ape = eval_df['APE'][~np.isnan(eval_df['APE'])]
                avg_ape = valid_ape.mean() if len(valid_ape) > 0 else np.inf

                # Track best model by validation APE
                if avg_ape < best_avg_ape:
                    best_avg_ape = avg_ape
                    best_params = params
                    best_model = candidate
            except Exception:
                grid_failures += 1
                continue

        gs_combos = total_combos
        gs_combos_tried = n_trying
        gs_failures = grid_failures
        gs_best_ape = best_avg_ape if best_avg_ape != np.inf else np.nan

        if grid_failures > 0:
            print(f"  Prophet grid search: {grid_failures} of {total_combos} combinations failed")

        if best_model is not None:
            model = best_model
            train_params = {**train_params, **best_params}
            print(f"  Prophet grid search: best val APE={best_avg_ape:.2f}%, params={best_params}")

    # Default model (if grid search disabled or failed)
    if model is None or not USE_PROPHET_GRID_SEARCH:
        model = _build_prophet_model(train_params, used_regressors, prophet_settings, prophet_data)

    # Create future dataframe and predict
    forecast = _build_prophet_future(model, validation_horizon + prediction_horizon, allData, used_regressors, cap)

    # Ensure demand signals exist
    allData = ensure_demand_signals(allData)

    # Merge results
    fullResult = pd.merge(allData, forecast, on='ds', how='right')

    # Select display columns (base from config + Prophet-specific + regressors)
    # After merge, used regressors gain _x suffix (merge disambiguation); not_used keep original names
    regressor_cols = list(not_used_regressors.keys()) + [reg + '_x' for reg in list(used_regressors.keys())]
    displayCols = DISPLAY_COLUMNS_BASE + ['ds', 'y', 'yhat'] + regressor_cols
    displayCols = [c for c in displayCols if c in fullResult.columns]

    predictions = fullResult[displayCols].copy()

    # Rename Prophet columns + standard demand signal renames
    prophet_renames = {'ds': 'QUARTER_PARSED', 'y': 'BILLING', 'yhat': 'BILLING_PREDICTED'}
    predictions = predictions.rename(columns=prophet_renames)
    predictions = apply_standard_renames(predictions)

    used_params = {**train_params, **used_regressors}
    predictions['used_params'] = str(used_params)
    predictions['MODEL_USED'] = 'PROPHET'

    # --------------------------------------------------------------------------
    # Final Metrics: APE on validation data (proper train/val split)
    # --------------------------------------------------------------------------
    val_start = -(validation_horizon + prediction_horizon)
    val_end = -prediction_horizon if prediction_horizon > 0 else None
    val_predictions = predictions.iloc[val_start:val_end].copy()
    val_predictions['APE'] = calculate_ape(val_predictions['BILLING'], val_predictions['BILLING_PREDICTED'])

    valid_ape = val_predictions['APE'][~np.isnan(val_predictions['APE'])]
    mape = float(valid_ape.mean()) if len(valid_ape) > 0 else np.nan

    metrics = {
        "mape": mape,  # Validation APE (proper train/val split)
        "prediction_count": len(valid_ape),
        "validation_horizon": validation_horizon,
        "prediction_horizon": prediction_horizon,
        "n_regressors": len(used_regressors),
        "regressors": list(used_regressors.keys()),
        "dropped_features": remaining_regressors,
        "cap": cap,
        # Grid search tracking
        "grid_search_enabled": USE_PROPHET_GRID_SEARCH,
        "grid_search_combos": gs_combos,
        "grid_search_combos_tried": gs_combos_tried if USE_PROPHET_GRID_SEARCH else 0,
        "grid_search_failures": gs_failures,
        "grid_search_best_ape": gs_best_ape,
    }

    return ProphetPredictor(model, list(used_regressors.keys()), cap), predictions, metrics, used_params

# COMMAND ----------

# MAGIC %md
# MAGIC # Sklearn-API Handler (XGBoost, LightGBM, CatBoost, etc.)
# MAGIC
# MAGIC This is the original AMD XGBoost logic, generalized to support any sklearn-compatible model.
# MAGIC
# MAGIC **Original code location:** `GENERATE_PREDICTIONS_ALL_QUARTER.ipynb`, Cell 12
# MAGIC (`FORECASTINGMODEL_XGBOOST_PROPHET.predict()`, the `if PREFERRED_MODEL == 'XGBOOST':` branch)
# MAGIC
# MAGIC **What changed vs. the original:**
# MAGIC - Model class is now loaded dynamically from config (enables LightGBM, CatBoost, etc.)
# MAGIC - Grid search uses sklearn's RandomizedSearchCV/GridSearchCV with PredefinedSplit
# MAGIC   (original used a manual itertools.product loop)
# MAGIC - Grid is broader (more hyperparameters explored)
# MAGIC - eval_metric changed from rmse to mae (more stable for small-denomination quarters)
# MAGIC - Merge key simplified from all columns to just MARKET_SEGMENT + QUARTER + QUARTER_PARSED
# MAGIC - Everything else (regressor adjustment, SELLIN exclusion for AIB/EMBEDDED, RFE feature
# MAGIC   selection, chronological split, early stopping, empty-validation fallback) is identical

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sklearn Helper Functions

# COMMAND ----------

def _import_class(class_path: str):
    """
    Dynamically import a class from a module path string.

    Example: _import_class('xgboost.XGBRegressor') returns the XGBRegressor class.
    This allows adding new model types via YAML config without code changes.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _supports_early_stopping(model_class, default_params: dict) -> bool:
    """Check if a model class supports early_stopping_rounds parameter."""
    try:
        test_model = model_class(**default_params)
        return 'early_stopping_rounds' in test_model.get_params()
    except Exception:
        return False


def _count_grid_combos(grid: dict) -> int:
    """Count total parameter combinations in a grid dict."""
    total = 1
    for param_options in grid.values():
        total *= len(param_options)
    return total

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sklearn Training Function

# COMMAND ----------

def _train_sklearn_api(
    df: pd.DataFrame,
    segment: str,
    seg_config: dict,
    config: dict,
    model_type: str,
    model_conf: dict
) -> Tuple:
    """
    Generic trainer for sklearn-compatible models (XGBoost, LightGBM, CatBoost, etc.)

    Model class is dynamically loaded from config. All models using fit(X, y) / predict(X)
    interface can use this handler. Logic matches original AMD train_xgboost() flow.
    """
    # --------------------------------------------------------------------------
    # Model class and hyperparameters from config (generic) or XGBoost defaults
    # --------------------------------------------------------------------------
    if model_conf and "class" in model_conf:
        model_class = _import_class(model_conf["class"])
        default_params = model_conf.get("hyperparameters", {}).copy()
        grid = model_conf.get("grid", {})
        # CatBoost: force file writing off (Spark workers may lack write access to cwd).
        # Belt-and-suspenders: config already sets this, but sklearn.clone() inside
        # GridSearchCV can drop it if CatBoost's get_params() doesn't round-trip cleanly.
        if "catboost" in (model_conf.get("class") or "").lower():
            default_params["allow_writing_files"] = False
    else:
        # Fallback to XGBoost defaults for backwards compatibility
        model_class = xgb.XGBRegressor
        xgb_conf = config.get("models", {}).get("xgboost", {})
        default_params = xgb_conf.get("hyperparameters", {}).copy()
        grid = xgb_conf.get("grid", {})

    # Grid search toggle: segment config > model config > default True
    seg_training_config = get_segment_training_config(config, segment)
    USE_GRID_SEARCH = seg_training_config["grid_search"].get(model_type, model_conf.get("grid_search", True))

    all_regressors = seg_config["all_regressors"]

    # Data prep
    allData, validation_horizon, prediction_horizon = prepare_training_data(df)

    min_rows = validation_horizon + prediction_horizon + 1
    if len(allData) < min_rows:
        raise ValueError(
            f"Insufficient training data: {len(allData)} rows, need at least {min_rows} "
            f"(val={validation_horizon}, pred={prediction_horizon})"
        )

    allData = adjust_regressors(allData, all_regressors, 'BILLING_MOVING_AVERAGE_2Q', segment)
    regressors_to_use = filter_regressors(all_regressors, segment, config.get("EXCLUDE_SELLIN_XGBOOST", []))

    # Feature selection
    common_columns = ['MARKET_SEGMENT', 'QUARTER', 'QUARTER_PARSED']
    target_column = ['BB']
    n_features = get_feature_count(config, segment)

    regressors = feature_selection(
        allData, validation_horizon, prediction_horizon,
        common_columns, regressors_to_use, target_column, n_features
    )

    # Split data (same slicing as original: train / val / test)
    train_data, validate_data, test_data = split_train_val_test(
        allData, validation_horizon, prediction_horizon,
        common_columns + regressors + target_column
    )

    # Original did not drop NaN targets -- added as safety net for partitions
    # where future periods (with NaN BB) leak into the training slice
    train_data = train_data.dropna(subset=target_column)
    validate_data = validate_data.dropna(subset=target_column)

    # Edge case: if no validation rows exist, use last training row as pseudo-validation.
    # This only affects early stopping -- model still trains on all training data.
    if validate_data.empty and len(train_data) > 1:
        pseudo_val = train_data.tail(1).copy()
        train_data = train_data.iloc[:-1].copy()
        validate_data = pd.concat([validate_data, pseudo_val], ignore_index=True)

    X_train = train_data[regressors]
    y_train = train_data[target_column[0]]
    X_val = validate_data[regressors]
    y_val = validate_data[target_column[0]]

    has_val = len(X_val) > 0

    # --------------------------------------------------------------------------
    # Model Training
    # Set USE_GRID_SEARCH via config (or grid_search widget in training notebook).
    # When disabled, uses default_params from config.
    # --------------------------------------------------------------------------
    early_stopping_rounds = model_conf.get("early_stopping_rounds", 5)
    dropped_features = [r for r in regressors_to_use if r not in regressors]
    supports_es = _supports_early_stopping(model_class, default_params)

    # Grid search tracking (populated below if grid search runs)
    gs_combos = 0
    gs_combos_tried = 0
    cv_results_summary = None

    if USE_GRID_SEARCH and grid:
        n_iter = model_conf.get("n_iter", 50)

        # MAE scorer (negative because sklearn maximizes).
        # MAE is more stable than APE for hyperparameter tuning -- APE blows up on
        # small-denomination quarters. Downstream eval picks champion by APE.
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        # Base estimator with early stopping (matches original AMD early_stopping_rounds=5)
        if supports_es:
            base_estimator = model_class(
                early_stopping_rounds=early_stopping_rounds if has_val else None,
                **default_params
            )
        else:
            base_estimator = model_class(**default_params)

        if has_val:
            # PredefinedSplit convention: -1 = training fold, 0 = validation fold.
            # This tells RandomizedSearchCV to use our exact train/val split (no k-fold).
            X_combined = pd.concat([X_train, X_val], ignore_index=True)
            y_combined = pd.concat([y_train, y_val], ignore_index=True)
            test_fold = np.concatenate([np.full(len(X_train), -1), np.zeros(len(X_val))])
            cv = PredefinedSplit(test_fold)
        else:
            X_combined, y_combined = X_train, y_train
            cv = 2  # fallback to 2-fold CV if no validation set

        # Count total combinations for logging
        total_combos = _count_grid_combos(grid)

        # Use RandomizedSearchCV if n_iter specified and less than total, else GridSearchCV
        if n_iter > 0 and n_iter < total_combos:
            print(f"  {model_type.upper()}: RandomizedSearchCV with {n_iter} of {total_combos} combinations")
            search = RandomizedSearchCV(
                base_estimator,
                param_distributions=grid,
                n_iter=n_iter,
                scoring=mae_scorer,
                cv=cv,
                random_state=grid.get("random_state", [42])[0] if isinstance(grid.get("random_state"), list) else 42,
                n_jobs=1,
                refit=True
            )
        else:
            print(f"  {model_type.upper()}: GridSearchCV with {total_combos} combinations")
            search = GridSearchCV(
                base_estimator,
                param_grid=grid,
                scoring=mae_scorer,
                cv=cv,
                n_jobs=1,
                refit=True
            )

        # Pass eval_set for early stopping (only used when has_val=True)
        fit_params = {}
        if has_val and supports_es:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        search.fit(X_combined, y_combined, **fit_params)

        best_model = search.best_estimator_
        best_params = search.best_params_
        best_val_mae = -search.best_score_  # flip sign back (sklearn negates)

        gs_combos = total_combos
        gs_combos_tried = n_iter if (n_iter > 0 and n_iter < total_combos) else total_combos

        # Extract top 10 grid search results for MLflow artifact
        try:
            cv_df = pd.DataFrame(search.cv_results_)
            cv_df = cv_df.sort_values("rank_test_score").head(10)
            param_cols = [c for c in cv_df.columns if c.startswith("param_")]
            summary_cols = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
            cv_results_summary = cv_df[summary_cols].to_dict(orient="records")
        except Exception:
            cv_results_summary = None

        print(f"  Best val MAE: {best_val_mae:.2f}")
        print(f"  Best params: {best_params}")
        if has_val and hasattr(best_model, 'best_iteration'):
            print(f"  Best iteration: {best_model.best_iteration}")
    else:
        # No grid search -- use default parameters from config
        print(f"  {model_type.upper()}: using default parameters (grid search disabled)")

        # Build model with early stopping if supported
        if supports_es:
            best_model = model_class(
                early_stopping_rounds=early_stopping_rounds if has_val else None,
                **default_params,
            )
        else:
            best_model = model_class(**default_params)

        # Fit with eval_set if model supports early stopping
        if has_val and supports_es:
            best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            best_model.fit(X_train, y_train)

        best_val_mae = mean_absolute_error(y_val, best_model.predict(X_val)) if has_val else np.nan
        best_params = default_params.copy()
        print(f"  Val MAE: {best_val_mae:.2f}" if not np.isnan(best_val_mae) else "  No validation set")

    # Build combined predictions DataFrame: NaN for train (no predictions), model output for val/test
    X_test = test_data[regressors]

    train_data = train_data.copy()
    validate_data = validate_data.copy()
    test_data = test_data.copy()

    train_data['BILLING_PREDICTED'] = np.nan
    if has_val:
        validate_data['BILLING_PREDICTED'] = best_model.predict(X_val)
    test_data['BILLING_PREDICTED'] = best_model.predict(X_test)

    forecast = pd.concat([train_data, validate_data, test_data])

    # Ensure demand signals exist
    allData = ensure_demand_signals(allData)

    # Merge on identity columns only (original also included BB + regressors in
    # the merge key, but that's unnecessary within a single partition and risks
    # float mismatches dropping rows)
    fullResult = pd.merge(
        allData,
        forecast[['MARKET_SEGMENT', 'QUARTER', 'QUARTER_PARSED', 'BILLING_PREDICTED']],
        on=['MARKET_SEGMENT', 'QUARTER', 'QUARTER_PARSED'],
        how='left'
    )

    # Select display columns (base from config + model-specific + regressors)
    displayCols = DISPLAY_COLUMNS_BASE + ['QUARTER_PARSED', 'BB', 'BILLING_PREDICTED'] + all_regressors
    displayCols = [c for c in displayCols if c in fullResult.columns]

    predictions = fullResult[displayCols].copy()
    predictions = apply_standard_renames(predictions)

    # Store params
    used_params = best_params.copy() if best_params else {}
    used_params['regressors'] = regressors
    used_params['model_class'] = model_conf.get("class", "xgboost.XGBRegressor") if model_conf else "xgboost.XGBRegressor"
    predictions['used_params'] = json.dumps(used_params)
    predictions['MODEL_USED'] = model_type.upper()

    # --------------------------------------------------------------------------
    # Final Metrics: APE on validation data (proper train/val split)
    # --------------------------------------------------------------------------
    val_predictions = validate_data.copy()
    val_predictions['BILLING_PREDICTED'] = best_model.predict(X_val) if has_val else np.nan
    val_predictions['APE'] = calculate_ape(val_predictions['BB'], val_predictions['BILLING_PREDICTED'])

    valid_ape = val_predictions['APE'][~np.isnan(val_predictions['APE'])]
    mape = valid_ape.mean() if len(valid_ape) > 0 else np.nan

    metrics = {
        "mape": mape,  # Validation APE (proper train/val split)
        "prediction_count": len(valid_ape),
        "grid_search_val_mae": best_val_mae,
        "validation_horizon": validation_horizon,
        "prediction_horizon": prediction_horizon,
        "n_regressors": len(regressors),
        "regressors": regressors,
        "dropped_features": dropped_features,
        "best_iteration": getattr(best_model, "best_iteration", None),
        # Grid search tracking
        "grid_search_enabled": USE_GRID_SEARCH,
        "grid_search_combos": gs_combos,
        "grid_search_combos_tried": gs_combos_tried,
        "cv_results_summary": cv_results_summary,
    }

    return SklearnPredictor(best_model, regressors), predictions, metrics, used_params

# COMMAND ----------

# MAGIC %md
# MAGIC # Predictor Wrappers
# MAGIC
# MAGIC Each model type has a thin wrapper with a standard `predict(model_input) -> np.ndarray`
# MAGIC interface. This keeps model-type-specific inference logic here alongside the handlers,
# MAGIC so `ForecastingModel.predict()` in `02_model_training` is a single delegation call.
# MAGIC Wrappers for Prophet/sklearn expose `.model` (the raw model) for interpretability.

# COMMAND ----------

class ProphetPredictor:
    """Wraps a fitted Prophet model with inference-time data prep."""

    def __init__(self, model, regressors, cap):
        self.model = model
        self.regressors = regressors
        self.cap = cap

    @property
    def pyfunc_config(self):
        return {"handler_type": "prophet", "regressors": self.regressors, "cap": float(self.cap)}

    def sample_input(self, predictions):
        sample = pd.DataFrame({
            "QUARTER_PARSED": predictions["QUARTER_PARSED"].head(1).values,
            "BILLING": predictions["BILLING"].head(1).values if "BILLING" in predictions.columns else [0.0],
        })
        for reg in self.regressors:
            val = predictions[reg].iloc[0] if reg in predictions.columns else 0.0
            sample[reg] = [float(val) if pd.notna(val) else 0.0]
        return sample

    def predict(self, model_input):
        inp = model_input[['QUARTER_PARSED']].rename(columns={'QUARTER_PARSED': 'ds'}).copy()
        for r in self.regressors:
            inp[r] = model_input[r].values if r in model_input.columns else 0
        if self.regressors:
            # Defensive fill cascade: forward-fill gaps, backward-fill leading NaNs, zeros for anything remaining
            inp[self.regressors] = inp[self.regressors].ffill().bfill().fillna(0)
        inp['floor'], inp['cap'] = 0, int(self.cap)
        return self.model.predict(inp)["yhat"].values


class SklearnPredictor:
    """Wraps an sklearn-compatible model with inference-time regressor selection."""

    def __init__(self, model, regressors):
        self.model = model
        self.regressors = regressors

    @property
    def pyfunc_config(self):
        return {"handler_type": "sklearn_api", "regressors": self.regressors}

    def sample_input(self, predictions):
        available = [r for r in self.regressors if r in predictions.columns]
        sample = predictions[available].head(1).copy()
        for col in sample.columns:
            sample[col] = sample[col].astype(float)
        return sample

    def predict(self, model_input):
        available_regs = [r for r in self.regressors if r in model_input.columns]
        if available_regs:
            return self.model.predict(model_input[available_regs].fillna(0))
        return np.full(len(model_input), np.nan)


class StatsForecastPredictor:
    """Wraps a StatsForecast model with horizon inference."""

    def __init__(self, model):
        self.model = model

    @property
    def pyfunc_config(self):
        return {"handler_type": "statsforecast", "regressors": []}

    def sample_input(self, predictions):
        return pd.DataFrame({"QUARTER_PARSED": predictions["QUARTER_PARSED"].head(1).values})

    def predict(self, model_input):
        h = len(model_input)
        forecast = self.model.predict(h=h)
        pred_col = [c for c in forecast.columns if c not in ['unique_id', 'ds']][0]
        return forecast[pred_col].values

# COMMAND ----------

# MAGIC %md
# MAGIC # Additional Model Handlers
# MAGIC
# MAGIC These are experimental/baseline models that extend the pipeline beyond Prophet and XGBoost.
# MAGIC They are not part of the original AMD codebase. They are opt-in only -- to use them,
# MAGIC add the model type to the `model_types` widget in `02_model_training.py`.
# MAGIC
# MAGIC - **StatsForecast** - Univariate baselines (SeasonalNaive, AutoARIMA). Good for sanity checks.
# MAGIC - **NeuralForecast** - Deep learning (NHITS, RNN, AutoLSTM). Trained as a single global model
# MAGIC   on the driver via `train_neuralforecast_global()` (not through this dispatcher).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serialization Stubs
# MAGIC
# MAGIC NeuralForecast (PyTorch/Lightning internals) doesn't serialize cleanly with cloudpickle.
# MAGIC These lightweight stubs are always serializable. At predict time, the real model is
# MAGIC restored from the stub.

# COMMAND ----------

class NFCheckpoint:
    """Stores a trained NeuralForecast model as tarred bytes of its native checkpoint.
    At predict time, untar to temp dir and call NeuralForecast.load().

    For global models (trained on all partitions), partition_id filters the
    forecast to this partition's unique_id. For per-partition models (legacy),
    partition_id is None and no filtering is applied."""

    def __init__(self, model_bytes: bytes, h: int, partition_id: str = None):
        self.model_bytes = model_bytes
        self.h = h
        self.partition_id = partition_id

    @property
    def pyfunc_config(self):
        return {"handler_type": "neuralforecast", "regressors": []}

    def sample_input(self, predictions):
        return pd.DataFrame({"QUARTER_PARSED": predictions["QUARTER_PARSED"].head(1).values})

    def predict(self, model_input):
        if not hasattr(self, "_nf_model"):
            self._nf_model = _nf_load_from_bytes(self.model_bytes)
        forecast = self._nf_model.predict()
        pred_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]
        # Filter to this partition's series if global model
        pid = getattr(self, "partition_id", None)
        if pid is not None and "unique_id" in forecast.columns:
            forecast = forecast[forecast["unique_id"] == pid]
        vals = forecast[pred_col].values
        n = len(model_input)
        # Right-align predictions: NaN for historical rows, forecast values for future rows
        out = np.full(n, np.nan)
        out[-min(self.h, n):] = vals[:min(self.h, n)]
        return out


def _nf_save_to_bytes(nf) -> bytes:
    """Save a fitted NeuralForecast object to bytes via its native save() + tar."""
    tmp_dir = tempfile.mkdtemp()
    try:
        nf.save(tmp_dir)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(tmp_dir, arcname=".")
        return buf.getvalue()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _nf_load_from_bytes(model_bytes: bytes):
    """Restore a NeuralForecast object from tarred bytes."""
    from neuralforecast import NeuralForecast
    tmp_dir = tempfile.mkdtemp()
    buf = io.BytesIO(model_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(tmp_dir)
    return NeuralForecast.load(tmp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Univariate Prediction Helper

# COMMAND ----------

def _build_univariate_predictions(allData, train_end, forecast_values):
    """Build predictions DataFrame and MAPE for univariate handlers.

    Clips negative predictions and calculates MAPE via remove_unwanted_rows.

    NOTE: MAPE here uses val+test via remove_unwanted_rows, which differs from
    sklearn/Prophet handlers that use val-only APE. This inconsistency is
    intentional for now -- univariate models don't have a separate validation
    concept. TODO: Align metric calculation across all handlers.
    """
    preds = allData.iloc[train_end:].copy()
    preds["BILLING"] = preds["BB"]
    preds = preds.drop(columns=["BB"])
    n_preds = min(len(forecast_values), len(preds))
    preds = preds.iloc[:n_preds].copy()
    preds["BILLING_PREDICTED"] = forecast_values[:n_preds]
    # Clip negative predictions to 0 (demand cannot be negative)
    preds["BILLING_PREDICTED"] = preds["BILLING_PREDICTED"].clip(lower=0)

    preds_clean = remove_unwanted_rows(preds.copy())
    mape = calculate_mape(preds_clean["BILLING"], preds_clean["BILLING_PREDICTED"]).mean() if len(preds_clean) > 0 else np.nan

    return preds, mape

# COMMAND ----------

# MAGIC %md
# MAGIC ## StatsForecast Handler
# MAGIC
# MAGIC Univariate baseline models from Nixtla's statsforecast library.
# MAGIC These don't use external regressors -- good as baselines to compare against Prophet/XGBoost.
# MAGIC If Prophet/XGBoost can't beat SeasonalNaive, something is wrong with the data or config.

# COMMAND ----------

def _train_statsforecast(
    df: pd.DataFrame,
    segment: str,
    seg_config: dict,
    config: dict,
    model_type: str,
    model_conf: dict
) -> Tuple:
    """
    Handler for statsforecast models (Nixtla library).

    These are univariate models - they don't use external regressors.
    Useful as baselines to compare against regressor-based models.

    Supported models: seasonalnaive, autoarima

    Args:
        df: Input DataFrame with QUARTER_PARSED, BB columns
        segment: Market segment name
        seg_config: Segment-specific config (not used for univariate)
        config: Full config dict
        model_type: Model name (seasonalnaive, autoarima)
        model_conf: Model-specific config from YAML

    Returns:
        (model, predictions_df, metrics_dict, params_dict)
    """
    from statsforecast import StatsForecast

    # Dynamic class import (same pattern as _train_sklearn_api)
    if not model_conf or "class" not in model_conf:
        raise ValueError(f"statsforecast model '{model_type}' requires 'class' in config (e.g., 'statsforecast.models.SeasonalNaive')")

    model_class = _import_class(model_conf["class"])
    hyperparams = model_conf.get("hyperparameters", {})
    model_spec = model_class(**hyperparams)

    # Data prep
    allData, validation_horizon, prediction_horizon = prepare_training_data(df)

    min_rows = validation_horizon + prediction_horizon + 1
    if len(allData) < min_rows:
        raise ValueError(
            f"Insufficient training data: {len(allData)} rows, need at least {min_rows} "
            f"(val={validation_horizon}, pred={prediction_horizon})"
        )

    # Train/test split (same as other handlers)
    train_end = -(validation_horizon + prediction_horizon)
    train_df = allData.iloc[:train_end].copy()

    # StatsForecast expects: unique_id, ds, y columns
    sf_df = pd.DataFrame({
        "unique_id": segment,
        "ds": pd.to_datetime(train_df["QUARTER_PARSED"]),
        "y": train_df["BB"].astype(float)
    })

    # Initialize and fit model
    sf = StatsForecast(models=[model_spec], freq="QS", n_jobs=1)
    sf.fit(sf_df)

    # Predict (full horizon: validation + prediction)
    forecast_horizon = validation_horizon + prediction_horizon
    forecast = sf.predict(h=forecast_horizon)

    # Get the model name from statsforecast (e.g., "SeasonalNaive", "AutoARIMA")
    model_col = model_spec.__class__.__name__
    forecast_values = forecast[model_col].values

    # Build predictions and calculate MAPE
    preds, mape = _build_univariate_predictions(allData, train_end, forecast_values)

    params = {
        "model_type": model_type,
        "season_length": hyperparams.get("season_length", 4),
    }

    # Add standard columns for downstream consistency
    preds = ensure_demand_signals(preds)
    preds = apply_standard_renames(preds)
    preds["MODEL_USED"] = model_type.upper()
    preds["used_params"] = json.dumps(params)

    metrics = {
        "mape": mape,
        "validation_horizon": validation_horizon,
        "prediction_horizon": prediction_horizon,
        "regressors": [],  # univariate - no regressors
        "grid_search_enabled": False,  # statsforecast models don't use grid search
    }

    return StatsForecastPredictor(sf), preds, metrics, params

# COMMAND ----------

# MAGIC %md
# MAGIC # Global NeuralForecast Training
# MAGIC
# MAGIC NeuralForecast models (NHITS, RNN, AutoLSTM) train as a single global model on the
# MAGIC driver, with all partitions as separate series via `unique_id`. This avoids PyTorch
# MAGIC compilation cache collisions that occur when multiple Spark workers import torch
# MAGIC simultaneously, and lets the model learn shared patterns across partitions.

# COMMAND ----------

def train_neuralforecast_global(
    all_partitions_pdf: pd.DataFrame,
    model_type: str,
    config: dict,
) -> list:
    """Train one global NeuralForecast model on all partitions simultaneously.

    Collects all partition data into a multi-series NF DataFrame (unique_id =
    PARTITION_COLUMN), trains once, then splits predictions back per partition.
    Returns result dicts matching training_result_schema.
    """
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    os.environ["TORCHDYNAMO_SUPPRESS_ERRORS"] = "1"
    try:
        import pwd
        pwd.getpwuid(os.getuid())
    except KeyError:
        os.environ.setdefault("HOME", "/tmp")

    from neuralforecast import NeuralForecast

    model_conf = config.get("models", {}).get(model_type.lower(), {})
    if not model_conf or "class" not in model_conf:
        raise ValueError(f"neuralforecast model '{model_type}' requires 'class' in config")

    model_class = _import_class(model_conf["class"])
    hyperparams = model_conf.get("hyperparameters", {}).copy()
    input_size = hyperparams.get("input_size", 4)

    results = []
    partitions = all_partitions_pdf["PARTITION_COLUMN"].unique()

    # --- Step 1: Build multi-series NF DataFrame ---
    nf_rows = []
    partition_meta = {}

    for partition in partitions:
        pdf = all_partitions_pdf[all_partitions_pdf["PARTITION_COLUMN"] == partition].copy()
        segment = pdf["MARKET_SEGMENT"].iloc[0]

        seg_training_config = get_segment_training_config(config, segment)
        if model_type.lower() not in seg_training_config["models"]:
            results.append(_error_result(partition, model_type, f"SKIPPED: {model_type} not configured for segment {segment}"))
            continue

        try:
            allData, val_h, pred_h = prepare_training_data(pdf)
            train_end = -(val_h + pred_h)
            train_df = allData.iloc[:train_end].copy()

            # NeuralForecast needs input_size rows for the lookback window plus
            # val_size (=h) rows held out for early stopping validation.
            min_rows = input_size + (val_h + pred_h)
            if len(train_df) < min_rows:
                results.append(_error_result(partition, model_type, f"Insufficient training data: {len(train_df)} rows < {min_rows} (input_size={input_size} + h={val_h + pred_h})"))
                continue

            partition_nf = pd.DataFrame({
                "unique_id": partition,
                "ds": pd.to_datetime(train_df["QUARTER_PARSED"]),
                "y": train_df["BB"].astype(float),
            })
            nf_rows.append(partition_nf)
            partition_meta[partition] = {
                "val_h": val_h, "pred_h": pred_h,
                "allData": allData, "train_end": train_end,
                "segment": segment,
            }
        except Exception as e:
            results.append(_error_result(partition, model_type, str(e)))

    if not nf_rows:
        return results

    sf_df = pd.concat(nf_rows, ignore_index=True)
    h = max(meta["val_h"] + meta["pred_h"] for meta in partition_meta.values())

    # --- Step 2: Train one global model ---
    # Auto* models (AutoLSTM, etc.) have a restricted constructor: h, num_samples,
    # loss, valid_loss, config, cpus, gpus, verbose, backend, callbacks.
    # Trainer-level kwargs (accelerator, max_steps, etc.) go into config dict.
    is_auto = model_class.__name__.startswith("Auto")
    if is_auto:
        AUTO_PARAMS = {"num_samples", "loss", "valid_loss", "cpus", "gpus",
                       "verbose", "backend", "callbacks"}
        auto_kwargs = {k: v for k, v in hyperparams.items() if k in AUTO_PARAMS}
        config_overrides = {k: v for k, v in hyperparams.items()
                            if k not in AUTO_PARAMS and k != "h"}
        if config_overrides:
            # Optuna backend requires config to be a callable: trial -> dict.
            # Wrap static overrides in a lambda so they pass through as fixed values.
            auto_kwargs["config"] = lambda trial, _co=config_overrides: _co
        model_spec = model_class(h=h, **auto_kwargs)
    else:
        extra_kwargs = {k: v for k, v in hyperparams.items() if k not in ("input_size", "h")}
        extra_kwargs["input_size"] = input_size
        model_spec = model_class(h=h, **extra_kwargs)
    nf = NeuralForecast(models=[model_spec], freq="QS")
    print(f"    Fitting global {model_type.upper()} on {len(partition_meta)} partitions ({len(sf_df)} total rows)...")
    nf.fit(sf_df, val_size=h)

    # --- Step 3: Predict for all partitions at once ---
    forecast = nf.predict()
    pred_col = [c for c in forecast.columns if c not in ["unique_id", "ds"]][0]  # first non-id col is the model's forecast output

    # --- Step 4: Serialize model checkpoint, free PyTorch memory, build results ---
    global_model_bytes = _nf_save_to_bytes(nf)

    # Free PyTorch model, optimizer state, and training data immediately.
    del nf, model_spec, sf_df, nf_rows
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    for partition, meta in partition_meta.items():
        try:
            allData = meta["allData"]
            train_end = meta["train_end"]
            segment = meta["segment"]
            partition_h = meta["val_h"] + meta["pred_h"]

            part_forecast = forecast[forecast["unique_id"] == partition]
            forecast_values = part_forecast[pred_col].values[:partition_h]

            preds, mape = _build_univariate_predictions(allData, train_end, forecast_values)

            stub = NFCheckpoint(
                model_bytes=global_model_bytes, h=partition_h, partition_id=partition,
            )

            params = {"model_type": model_type, "input_size": input_size, "h": partition_h}

            preds = ensure_demand_signals(preds)
            preds = apply_standard_renames(preds)
            preds["MODEL_USED"] = model_type.upper()
            preds["used_params"] = json.dumps(params)

            metrics = {
                "mape": mape,
                "validation_horizon": meta["val_h"],
                "prediction_horizon": meta["pred_h"],
                "regressors": [],
                "grid_search_enabled": False,
            }

            model_config = {
                "model_type": model_type.upper(), "segment": segment,
                "partition": partition, "handler_type": "neuralforecast",
                "regressors": [],
            }

            # ForecastingModel is defined in 02_model_training.py but is in scope
            # at call time because this file is executed via %run before it.
            pyfunc_model = ForecastingModel(model=stub, model_config=model_config)
            sample_input_df = stub.sample_input(preds)
            sample_json = sample_input_df.to_json(date_format="iso")

            preds_sample_cols = [c for c in ["QUARTER_PARSED", "BILLING", "BILLING_PREDICTED",
                                             "MARKET_SEGMENT", "QUARTER"] if c in preds.columns]
            predictions_sample_json = preds[preds_sample_cols].head(20).to_json(date_format="iso")

            metrics_serializable = {
                k: (v if not isinstance(v, float) or not np.isnan(v) else None)
                for k, v in metrics.items()
            }
            params_serializable = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in params.items()
            }

            # Store the raw pyfunc_model object, NOT cloudpickle.dumps().
            # All partitions' NFCheckpoints share the same global_model_bytes
            # reference, so only one copy lives in RAM. Serialization is deferred
            # to the scoring-records write in 02_model_training.py.
            results.append({
                "PARTITION_COLUMN": partition,
                "MODEL_USED": model_type.upper(),
                "model_binary": pyfunc_model,
                "model_config": json.dumps(model_config),
                "mape": float(mape) if not np.isnan(mape) else None,
                "sample_input": sample_json,
                "predictions_sample": predictions_sample_json,
                "training_metrics": json.dumps(metrics_serializable, default=str),
                "used_params": json.dumps(params_serializable, default=str),
                "error": None,
            })
        except Exception as e:
            results.append(_error_result(partition, model_type, str(e)))

    del forecast, partition_meta
    gc.collect()

    return results
