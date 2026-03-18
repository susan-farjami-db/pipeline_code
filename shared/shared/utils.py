# Import python packages
import builtins 
import pickle
# import streamlit as st
import json
import pytz
import datetime
import pandas as pd
import numpy as np
import ast
import re

from typing import List, Dict, Optional, Mapping

import statsmodels.api as sm
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso, HuberRegressor

from pyspark.sql import functions as F  # from snowflake.snowpark import functions as F
from datetime import date, timedelta
from pyspark.sql import types as T  # from snowflake.snowpark import types as T

import mlflow  # from snowflake.ml.registry import registry
from mlflow.pyfunc import PythonModel  # from snowflake.ml.model import custom_model
from mlflow.models import infer_signature, ModelSignature  # from snowflake.ml.model import model_signature
from mlflow.types.schema import Schema, ColSpec  # from snowflake.ml.model.model_signature import ModelSignature, FeatureSpec, DataType, FeatureGroupSpec

def split(value):
    return value.split('M')[0]
def process_2(rows):
    return pd.Series(rows['MOF_VERSION'].apply(split), index=rows.index)

def fix_values(column):
    return F.upper(F.regexp_replace(F.col(column), "[^a-zA-Z0-9]+", "_"))

def get_next_version(reg, model_name) -> str:
    """
    Returns the next version of a model based on the existing versions in the registry.

    Args:
        reg: The registry object that provides access to the models.
        model_name: The name of the model.

    Returns:
        str: The next version of the model in the format "V_<version_number>".

    Raises:
        ValueError: If the version list for the model is empty or if the version format is invalid.
    """
    models = reg.show_models()
    if models.empty:
        return "V_1"
    elif model_name not in models["name"].to_list():
        return "V_1"
    max_version_number = builtins.max(
        [
            int(version.split("_")[-1])
            for version in ast.literal_eval(
                models.loc[models["name"] == model_name, "versions"].values[0]
            )
        ]
    )
    return f"V_{max_version_number + 1}"

def make_verification_columns(df, affix):
    # Assign numeric values to categories
    mapping = {'VERY LOW': 1, 'LOW': 2, 'FLAT': 3, 'HIGH': 4, 'VERY HIGH': 5}
    df['INDICATOR_num'] = df['INDICATOR' + affix].map(mapping)
    df['INDICATOR_FORECASTED_num'] = df['INDICATOR_FORECASTED' + affix].map(mapping)

    # Calculate distance if both columns have values
    df['VALIDATION_DISTANCE' + affix] = df.apply(
        lambda row: abs(row['INDICATOR_num'] - row['INDICATOR_FORECASTED_num']) if pd.notnull(row['INDICATOR' + affix]) and pd.notnull(row['INDICATOR_FORECASTED' + affix]) else None,
        axis=1
    )
    
    # Define the "Same Direction" Boolean indicator
    df['DIRECTIONAL_CORRECTNESS' + affix] = df.apply(
        lambda row: (
            (row['INDICATOR_num'] > 3 and row['INDICATOR_FORECASTED_num'] > 3)  # HIGH and VERY HIGH
        ) or (
            (row['INDICATOR_num'] <= 2 and row['INDICATOR_FORECASTED_num'] <= 2)  # VERY LOW and LOW
        ) or (
            (row['INDICATOR_num'] == 3 and row['INDICATOR_FORECASTED_num'] == 3)  # FLAT and FLAT
        ) if pd.notnull(row['INDICATOR' + affix]) and pd.notnull(row['INDICATOR_FORECASTED' + affix]) else None,
        axis=1
    )
    
    df = df.drop(['INDICATOR_num', 'INDICATOR_FORECASTED_num'], axis=1)
    
    return df

def add_quarters(val, k):

    _quarter_re = re.compile(r'(?P<y>\d{4})\s*[- ]?Q\s*(?P<q>[1-4])', re.I)

    m = _quarter_re.search(str(val))
    if not m:
        return pd.NA  # or return val
    y = int(m.group('y'))
    q = int(m.group('q'))
    total = (q - 1) + k
    new_y = y + total // 4
    new_q = (total % 4) + 1
    return f"{new_y}Q{new_q}"

def _first_true_pos(cond: pd.Series) -> int:
    """Return the position of the first True. If none, return len(cond)."""
    idxs = np.flatnonzero(cond.to_numpy())
    return int(idxs[0]) if idxs.size else len(cond)

def _make_cond_from_indicator(series: pd.Series, token: str) -> pd.Series:
    """
    Build a boolean condition from an indicator column that may be:
      - string 'Q+1'/'Q+2'/...
      - boolean True/False
    """
    s = series
    # If it's effectively boolean, use it directly
    if s.dropna().isin([True, False]).all():
        return s.fillna(False).astype(bool)
    # Otherwise compare as string (trim spaces just in case)
    return s.astype("string").str.strip().eq(token).fillna(False)

def split_by_indicators(
    df: pd.DataFrame,
    part_col: str = "PARTITION_COLUMN",
    indicator_cols: Optional[Mapping[str, str]] = None,  # e.g., {"Q+1": "Q+1_INDICATOR", "Q+2": "Q+2_INDICATOR", ...}
    time_col: Optional[str] = "ds",   # optional, used only for stable ordering
    new_part_col: str = "PARTITION_HORIZON",
    include_cut_row: bool = False,    # False => 'up to' (exclusive); True => inclusive
) -> pd.DataFrame:
    """
    For each partition:
      - For each horizon token (e.g., 'Q+1', 'Q+2', ...):
          * Build a condition from the associated indicator column
          * Slice rows up to the first match (exclusive by default; inclusive if include_cut_row)
          * Tag slice with new column PARTITION_HORIZON = f'{PARTITION_COLUMN}_{horizon}'
    Concatenate all slices across partitions and horizons.
    """
    # Default to Q+1..Q+4 if indicator_cols not provided, but only include those present in df
    if indicator_cols is None:
        candidate = {f"Q+{i}": f"Q+{i}_INDICATOR" for i in range(1, 5)}
        indicator_cols = {h: c for h, c in candidate.items() if c in df.columns}

    if not indicator_cols:
        # Nothing to do; return empty frame with same cols
        return pd.DataFrame(columns=list(df.columns) + [new_part_col])

    # Sort for deterministic order per group
    sort_cols = [c for c in [part_col, time_col] if c in df.columns and c is not None]
    pdf = df.sort_values(sort_cols).copy() if sort_cols else df.copy()

    out = []
    for key, sub in pdf.groupby(part_col, dropna=False, sort=False):
        for horizon, col in indicator_cols.items():
            if col in sub.columns:
                cond = _make_cond_from_indicator(sub[col], horizon)
            else:
                # If the column is missing, treat as no matches
                cond = pd.Series(False, index=sub.index)

            pos = _first_true_pos(cond)
            end = pos + (1 if include_cut_row and pos < len(sub) else 0)

            horizon_slice = sub.iloc[:end].copy()
            if not horizon_slice.empty:
                horizon_slice[new_part_col] = horizon_slice[part_col].astype(str) + f"_{horizon}"
                out.append(horizon_slice)

    return pd.concat(out, ignore_index=True, sort=False) if out else pd.DataFrame(columns=list(df.columns) + [new_part_col])