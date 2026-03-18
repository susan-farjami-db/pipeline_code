# modeling_utils.py
#
# Utility functions for feature selection, regressor adjustment, column
# renaming, and output formatting. Called by 01_training_functions.py.

import json
import numpy as np
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

def load_segment_config(config, segment: str) -> dict:
    """
    Extract configuration for a specific market segment from YAML config.
    """
    seg = config.get("segments", {}).get(segment) or config.get("general", {})
    used_regressors = seg["USED_REGRESSORS"]
    not_used_regressors = seg.get("NOT_USED_REGRESSORS") or {}

    return {
        "regressors_weights": used_regressors,
        "used_regressors": used_regressors,
        "not_used_regressors": not_used_regressors,
        "all_regressors": list(used_regressors.keys()) + list(not_used_regressors.keys()),
    }


def get_segment_training_config(config: dict, segment: str) -> dict:
    """
    Get training config for a segment (models to train, grid search settings).

    Reads from `segments[segment]` in YAML. Each segment inherits from `general`
    anchor but can override `models` and `grid_search` settings.

    Args:
        config: Full config dict (from model_config.yaml)
        segment: Segment name (e.g., "MOBILE", "DESKTOP")

    Returns:
        {"models": ["prophet", "xgboost"], "grid_search": {"prophet": True, "xgboost": True}}
    """
    # Defaults (same as general anchor)
    defaults = {
        "models": ["prophet", "xgboost"],
        "grid_search": {"prophet": True, "xgboost": True}
    }

    # Get segment config (inherits from general via YAML anchor)
    seg = config.get("segments", {}).get(segment, {})

    return {
        "models": seg.get("models", defaults["models"]),
        "grid_search": {**defaults["grid_search"], **seg.get("grid_search", {})}
    }


def feature_selection(allData, validation_horizon, prediction_horizon, common_columns, regressors, target_column, n_features=5):
    """
    Select top features using RFE with HuberRegressor.

    Slices data before validation/prediction horizon, then uses Recursive Feature Elimination
    to select the most important regressors.
    """
    train_data = allData.iloc[:-(validation_horizon + prediction_horizon)]
    train_data = train_data[common_columns + regressors + target_column]
    train_data = train_data.dropna(subset=target_column)
    train_data[regressors] = train_data[regressors].fillna(0)

    if len(train_data) == 0:
        raise ValueError(
            f"Insufficient training data: 0 rows after reserving "
            f"validation_horizon={validation_horizon} + prediction_horizon={prediction_horizon} "
            f"from {len(allData)} total rows"
        )

    X_feature_selection = train_data[regressors]
    Y_feature_selection = train_data[target_column[0]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feature_selection)
    y_scaled = scaler.fit_transform(np.array(Y_feature_selection).reshape(-1, 1)).ravel()

    selector = RFE(estimator=HuberRegressor(), n_features_to_select=n_features)
    selector.fit(X_scaled, y_scaled)  # Only need fit, not fit_transform (result was discarded)

    return list(X_feature_selection.columns[selector.get_support()])


def adjust_regressors(data, regressors, main_regressor, Market_Segment, threshold=0.1):
    """
    Replace regressor values below threshold of main regressor (likely data quality issues).

    For GAMING segment, only replaces non-zero values below threshold.
    For other segments, replaces all values below threshold.
    """
    for reg in regressors:
        if Market_Segment == 'GAMING':
            data.loc[
                (data[reg] < threshold * data[main_regressor]) & (data[reg] != 0), reg
            ] = data[main_regressor]
        else:
            data.loc[data[reg] < threshold * data[main_regressor], reg] = data[main_regressor]
    return data


def ensure_demand_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure demand signal columns exist, filling with 0 if missing."""
    demand_signals = ['MOF_0_MONTH', 'MOF_PLUS_BUFFER_0_MONTH', 'SELLIN_0_MONTH', 'SR_0_MONTH']
    df = df.copy()
    for signal in demand_signals:
        if signal not in df.columns:
            df[signal] = 0
    return df


def split_train_val_test(df, validation_horizon, prediction_horizon, columns):
    """Split data into train, validation, and test sets based on horizons."""
    holdout = validation_horizon + prediction_horizon

    train_data = df.iloc[:-holdout][columns].copy()
    validate_data = df.iloc[-holdout:-prediction_horizon][columns].copy()
    test_data = df.iloc[-prediction_horizon:][columns].copy()

    return train_data, validate_data, test_data


def apply_standard_renames(df: pd.DataFrame, rename_map: dict = None) -> pd.DataFrame:
    """
    Apply standard column renames and clean up merge artifacts (_x suffix).

    Args:
        df: DataFrame to rename
        rename_map: Optional custom rename map. If None, uses default demand signal renames.

    Returns:
        DataFrame with renamed columns
    """
    if rename_map is None:
        import yaml
        with open("configs/variable_config.yaml", "r") as _f:
            _vc = yaml.safe_load(_f)
        rename_map = _vc.get("DEMAND_SIGNAL_RENAMES", {})

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Clean up _x suffix from merges
    for col in df.columns:
        if col.endswith('_x'):
            df = df.rename(columns={col: col.rsplit('_x', 1)[0]})

    return df


def extract_horizons(df: pd.DataFrame) -> tuple:
    """
    Extract validation and prediction horizons from DataFrame.

    Returns:
        Tuple of (validation_horizon, prediction_horizon), either can be None if column missing.
    """
    val_h = None
    pred_h = None

    if 'VERIFICATION_HORIZON' in df.columns:
        val_h = int(df['VERIFICATION_HORIZON'].median())

    if 'PREDICTION_HORIZON' in df.columns:
        pred_h = int(df['PREDICTION_HORIZON'].median())

    return val_h, pred_h


def format_output(
    df: pd.DataFrame,
    model_name: str,
    used_params: dict,
    all_regressors: list = None
) -> pd.DataFrame:
    """
    Standardize output DataFrame with consistent column names and ordering.
    Used for post-processing if needed outside the training functions.
    """
    predictions = df.copy()

    rename_map = {
        'ds': 'QUARTER_PARSED',
        'y': 'BILLING',
        'yhat': 'BILLING_PREDICTED',
        'BB': 'BILLING',
        'MOF_0_MONTH': 'MOF',
        'MOF_PLUS_BUFFER_0_MONTH': 'MOF_PLUS_BUFFER',
        'SELLIN_0_MONTH': 'SELLIN',
        'SR_0_MONTH': 'SR'
    }
    predictions = predictions.rename(columns={k: v for k, v in rename_map.items() if k in predictions.columns})

    for col in predictions.columns:
        if col.endswith('_x'):
            predictions = predictions.rename(columns={col: col.rsplit('_x', 1)[0]})

    predictions['used_params'] = json.dumps(used_params) if isinstance(used_params, dict) else str(used_params)
    predictions['MODEL_USED'] = model_name

    col_order = [
        "MARKET_SEGMENT", "QUARTER", "QUARTER_PARSED", "BILLING", "BILLING_PREDICTED",
        "VERIFICATION_HORIZON", "PREDICTION_HORIZON", "BB_VERSION", "MOF_VERSION",
        "BB_VERSION_LATEST", "MOF_VERSION_LATEST", "MOF", "MOF_PLUS_BUFFER", "SELLIN", "SR",
        "SR_MINUS3_MONTH", "SELLIN_MINUS3_MONTH", "MOF_PLUS_BUFFER_MINUS3_MONTH",
        "BILLING_MOVING_AVERAGE_2Q", "BILLING_MOVING_AVERAGE_3Q",
        "BILLING_MOVING_AVERAGE_4Q", "BILLING_MOVING_AVERAGE_5Q",
        "used_params", "MODEL_USED"
    ]
    predictions = predictions[[c for c in col_order if c in predictions.columns]]
    predictions.columns = [f"{col}_OUTPUT" for col in predictions.columns]

    return predictions


def get_feature_count(config: dict, segment: str) -> int:
    """
    Get number of features to select based on segment type.

    Extended segments (MOBILE, AIB, EMBEDDED) get 6 features, others get 5.

    Args:
        config: Full config dict (from model_config.yaml)
        segment: Segment name

    Returns:
        Number of features to select (5 or 6)
    """
    extended_segments = config.get("FEATURE_SELECTION", {}).get("EXTENDED_SEGMENTS", [])
    if segment in extended_segments:
        return config["FEATURE_SELECTION"]["N_FEATURES_EXTENDED"]
    return config["FEATURE_SELECTION"]["N_FEATURES_GENERAL"]


def filter_regressors(regressors: list, segment: str, exclude_list: list) -> list:
    """
    Remove SELLIN_MINUS3_MONTH from regressor list for specified segments.

    Args:
        regressors: List of regressor column names
        segment: Current segment name
        exclude_list: List of segments where SELLIN should be excluded

    Returns:
        Filtered list of regressors
    """
    if segment in exclude_list:
        return [r for r in regressors if r != "SELLIN_MINUS3_MONTH"]
    return regressors
