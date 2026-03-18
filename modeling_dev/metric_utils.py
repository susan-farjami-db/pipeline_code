import re

import numpy as np
import pandas as pd

def make_verification_columns(df, affix):
    mapping = {'VERY LOW': 1, 'LOW': 2, 'FLAT': 3, 'HIGH': 4, 'VERY HIGH': 5}
    ind_num = df['INDICATOR' + affix].map(mapping)
    fore_num = df['INDICATOR_FORECASTED' + affix].map(mapping)

    both_valid = df['INDICATOR' + affix].notna() & df['INDICATOR_FORECASTED' + affix].notna()

    df['VALIDATION_DISTANCE' + affix] = np.where(
        both_valid, (ind_num - fore_num).abs().astype(float), np.nan
    )
    df['DIRECTIONAL_CORRECTNESS' + affix] = np.where(
        both_valid,
        (((ind_num > 3) & (fore_num > 3)) |
         ((ind_num <= 2) & (fore_num <= 2)) |
         ((ind_num == 3) & (fore_num == 3))).astype(float),
        np.nan
    )

    return df

def calculate_indicator(billingCol, MOFCol):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(MOFCol == 0, np.nan, (billingCol - MOFCol) / MOFCol)
    result = np.full(ratios.shape, np.nan, dtype=object)
    result[ratios < -0.2] = 'VERY LOW'
    result[(ratios >= -0.2) & (ratios <= -0.1)] = 'LOW'
    result[(ratios > -0.1) & (ratios < 0.1)] = 'FLAT'
    result[(ratios >= 0.1) & (ratios < 0.2)] = 'HIGH'
    result[ratios >= 0.2] = 'VERY HIGH'
    return result

def calculate_mape(actual, predicted):
    """Calculate MAPE for pandas Series, returning a Series of APE values."""
    mask = actual.notnull() & predicted.notnull() & (actual != 0)
    return (abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_ape(actual, predicted):
    """Calculate Absolute Percentage Error for scalar or array values."""
    if isinstance(actual, (pd.Series, np.ndarray)):
        # Vectorized for arrays
        result = np.where(
            (actual != 0) & pd.notna(actual) & pd.notna(predicted),
            np.abs((actual - predicted) / actual) * 100,
            np.nan
        )
        return result
    else:
        # Scalar
        if actual == 0 or actual is None or (isinstance(actual, float) and np.isnan(actual)):
            return np.nan
        if predicted is None or (isinstance(predicted, float) and np.isnan(predicted)):
            return np.nan
        return abs((actual - predicted) / actual) * 100

def calculate_indicators(predictions, demand_signals, validation_horizon=None, prediction_horizon=None):
    """
    Calculate directional indicators comparing actual vs predicted billing to demand signals.
    Matches the original Snowflake order of operations: compute INDICATOR, null demand/indicators,
    then compute INDICATOR_FORECASTED with the (now-nulled) demand values.
    """
    do_nulling = validation_horizon and prediction_horizon and (validation_horizon + prediction_horizon) > 0

    for demand in demand_signals:
        affix = '_'  + demand + '_REFERENCED'

        # 1. Compute actual indicator
        predictions['INDICATOR' + affix] = calculate_indicator(predictions['BILLING'].values, predictions[demand].values)

        # 2. Null demand and actual indicator for training/test rows
        if do_nulling:
            if 'VERIFICATION_HORIZON' in predictions.columns:
                is_training = predictions['VERIFICATION_HORIZON'] > validation_horizon
                is_test = predictions['VERIFICATION_HORIZON'] <= 0
                predictions.loc[is_training, demand] = None
                predictions.loc[is_training | is_test, 'INDICATOR' + affix] = None
            else:
                predictions.iloc[:-(validation_horizon + prediction_horizon), predictions.columns.get_loc(demand)] = None
                predictions.iloc[-prediction_horizon:, predictions.columns.get_loc('INDICATOR' + affix)] = None
                predictions.iloc[:-(validation_horizon + prediction_horizon), predictions.columns.get_loc('INDICATOR' + affix)] = None

        # 3. Compute forecasted indicator AFTER demand nulling (matches original Snowflake order)
        predictions['INDICATOR_FORECASTED' + affix] = calculate_indicator(predictions['BILLING_PREDICTED'].values, predictions[demand].values)

        # 4. Null forecasted indicator for training rows
        if do_nulling:
            if 'VERIFICATION_HORIZON' in predictions.columns:
                predictions.loc[is_training, 'INDICATOR_FORECASTED' + affix] = None
            else:
                predictions.iloc[:-(validation_horizon + prediction_horizon), predictions.columns.get_loc('INDICATOR_FORECASTED' + affix)] = None

        predictions = make_verification_columns(predictions, affix)

    return predictions

def remove_unwanted_rows(df, max_horizon=4):
    """
    Remove rows for same-quarter predictions and optionally filter extended horizons.

    Args:
        df: DataFrame with MOF_VERSION and QUARTER columns
        max_horizon: Maximum horizon to keep (4 = Q+1 to Q+4, 6 = Q+1 to Q+6)
    """
    # Check if MOF_VERSION has parseable quarter format (e.g., "2022Q4M1")
    # If not (e.g., masked synthetic data like "VER_XXXX"), skip filtering
    sample_mof = df['MOF_VERSION'].dropna().head(1)
    if len(sample_mof) == 0:
        print("Warning: No MOF_VERSION values found, skipping remove_unwanted_rows filtering")
        return df

    sample_val = str(sample_mof.iloc[0])
    if 'Q' not in sample_val or not any(c.isdigit() for c in sample_val.split('Q')[0]):
        print(f"Warning: MOF_VERSION format not parseable ('{sample_val[:20]}...'), skipping remove_unwanted_rows filtering")
        return df

    #Add column QUARTER_MONTH
    df['QUARTER_MONTH']= df['QUARTER']+'M3'

    def extract_quarter(value):
        quarter = value.split('M')[0]
        return quarter


    #Add column MOF_VERSION_QUARTER
    df["MOF_VERSION_QUARTER"] = df['MOF_VERSION'].apply(extract_quarter)

    #Add column REMOVE (same-quarter predictions)
    df["REMOVE"] = df['MOF_VERSION_QUARTER'] == df['QUARTER']

    # Build list of columns to filter on
    filter_cols = ['REMOVE']
    cols_to_remove = ['QUARTER_MONTH', 'REMOVE', 'MOF_VERSION_QUARTER']

    # Only filter Q+5/Q+6 if max_horizon < 5 or < 6
    if max_horizon < 5:
        df["Q+5"] = df['MOF_VERSION_QUARTER'].apply(lambda v: add_quarters(v, 5))
        df["Q+5_INDICATOR"] = df['QUARTER'] == df['Q+5']
        filter_cols.append('Q+5_INDICATOR')
        cols_to_remove.extend(['Q+5', 'Q+5_INDICATOR'])

    if max_horizon < 6:
        df["Q+6"] = df['MOF_VERSION_QUARTER'].apply(lambda v: add_quarters(v, 6))
        df["Q+6_INDICATOR"] = df['QUARTER'] == df['Q+6']
        filter_cols.append('Q+6_INDICATOR')
        cols_to_remove.extend(['Q+6', 'Q+6_INDICATOR'])

    # Remove rows where any filter column is TRUE
    df = df[~(df[filter_cols].any(axis=1))]

    # Drop helper columns
    df = df.drop([c for c in cols_to_remove if c in df.columns], axis=1)

    return df


def apply_evaluation_pipeline(df, demand_signals, max_horizon=4):
    """APE -> indicators -> null-mask -> remove_unwanted_rows.

    Shared evaluation pipeline used by notebook 03 (per-partition inside UDF)
    and notebook 04 (per-partition via groupby.apply on driver).

    Args:
        df: DataFrame for a single partition with BILLING, BILLING_PREDICTED,
            VERIFICATION_HORIZON, PREDICTION_HORIZON columns.
        demand_signals: List of demand signal names (e.g. ["MOF", "SELLIN", "SR"]).
        max_horizon: Maximum horizon to keep (4 or 6).
    """
    # 1. APE (skip null/zero BILLING)
    df['APE'] = np.where(
        df['BILLING'].notna() & (df['BILLING'] != 0),
        np.abs((df['BILLING'] - df['BILLING_PREDICTED']) / df['BILLING']) * 100,
        np.nan
    )

    # 2. Demand signal indicators
    val_h = int(df['VERIFICATION_HORIZON'].median()) if df['VERIFICATION_HORIZON'].notna().any() else None
    pred_h = int(df['PREDICTION_HORIZON'].median()) if df['PREDICTION_HORIZON'].notna().any() else None
    df = calculate_indicators(
        df, demand_signals,
        validation_horizon=val_h, prediction_horizon=pred_h
    )

    # 3. Null APE where indicators are missing
    indicator_cols = [c for c in df.columns if c.endswith('_INDICATOR')]
    if indicator_cols:
        df.loc[df[indicator_cols].isnull().any(axis=1), 'APE'] = np.nan

    # 4. Remove same-quarter and beyond-horizon rows
    df = remove_unwanted_rows(df, max_horizon=max_horizon)

    return df


# Quarter indicator helpers (ported from GENERATE_PREDICTIONS_ITERATIVE_METRICS.ipynb)

_quarter_re = re.compile(r'(?P<y>\d{4})\s*[- ]?Q\s*(?P<q>[1-4])', re.I)

def add_quarters(val, k):
    """Add k quarters to a quarter string like '2024Q1'."""
    m = _quarter_re.search(str(val))
    if not m:
        return None
    y = int(m.group('y'))
    q = int(m.group('q'))
    total = (q - 1) + k
    new_y = y + total // 4
    new_q = (total % 4) + 1
    return f"{new_y}Q{new_q}"


def _has_parseable_quarter(mof_version_series):
    """Check if MOF_VERSION contains parseable quarter information."""
    sample = mof_version_series.dropna().head(10)
    for val in sample:
        # Try to extract quarter from MOF_VERSION
        mof_quarter = str(val).split('M')[0]
        if _quarter_re.search(mof_quarter):
            return True
    return False


def _infer_mof_quarter_from_max(df, max_horizon=6):
    """
    Infer MOF_QUARTER for masked data by looking at max QUARTER per MOF_VERSION.

    Logic: Data includes up to Q+max_horizon, so MOF_QUARTER = max(QUARTER) - max_horizon quarters.
    This allows horizon calculation even when MOF_VERSION is anonymized.
    """
    # Get max QUARTER for each MOF_VERSION
    max_quarters = df.groupby('MOF_VERSION')['QUARTER'].max().to_dict()

    # Infer MOF_QUARTER as max_quarter - max_horizon
    mof_quarter_map = {}
    for mof_version, max_q in max_quarters.items():
        inferred = add_quarters(max_q, -max_horizon)
        if pd.notna(inferred):
            mof_quarter_map[mof_version] = inferred

    return mof_quarter_map


def create_quarter_indicators(pred_df, max_horizon=6):
    """
    Create Q+1 through Q+max_horizon indicator columns based on MOF_VERSION.
    Ported from GENERATE_PREDICTIONS_ITERATIVE_METRICS.ipynb

    If MOF_VERSION is masked/anonymized, infers MOF_QUARTER from max(QUARTER) per MOF_VERSION.

    Args:
        pred_df: DataFrame with MOF_VERSION and QUARTER columns
        max_horizon: Maximum horizon to create indicators for (default 6 for Q+1 to Q+6)
    """
    df = pred_df.copy()

    if 'MOF_VERSION' not in df.columns:
        return None

    # Check if MOF_VERSION has parseable quarter info
    if _has_parseable_quarter(df['MOF_VERSION']):
        # Normal path: extract MOF_QUARTER from MOF_VERSION
        df["MOF_QUARTER"] = df['MOF_VERSION'].apply(
            lambda v: str(v).split('M')[0] if pd.notna(v) else None
        )
    else:
        # Masked data: infer MOF_QUARTER from max QUARTER per MOF_VERSION
        mof_quarter_map = _infer_mof_quarter_from_max(df, max_horizon=max_horizon)
        if not mof_quarter_map:
            return None  # Can't infer anything
        df["MOF_QUARTER"] = df['MOF_VERSION'].map(mof_quarter_map)

    # Create Q+1 through Q+max_horizon columns and indicators
    q_cols = []
    for k in range(1, max_horizon + 1):
        df[f"Q+{k}"] = df["MOF_QUARTER"].apply(lambda v, kk=k: add_quarters(v, kk))
        df[f"Q+{k}_INDICATOR"] = df['QUARTER'] == df[f"Q+{k}"]
        q_cols.append(f"Q+{k}")

    # Remove rows where QUARTER == MOF_QUARTER (same quarter, no prediction needed)
    df["REMOVE_SAME_QUARTERS"] = df['QUARTER'] == df['MOF_QUARTER']
    df = df[~df['REMOVE_SAME_QUARTERS']]

    # Drop helper columns (keep indicators)
    drop_cols = ['MOF_QUARTER', 'REMOVE_SAME_QUARTERS'] + q_cols
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


def _backfill_billing_from_q2(df, q2_df, q3_df, q4_df):
    """
    Backfill Q3/Q4 billing actuals from Q2 when future actuals aren't available.
    Ported from GENERATE_PREDICTIONS_ITERATIVE_METRICS.ipynb.

    In forecasting scenarios, Q3/Q4 actuals may not exist yet, so we use Q2 billing
    as a proxy for calculating metrics.
    """
    if q2_df.empty:
        return q3_df, q4_df

    keys = ["QUARTER", "MARKET_SEGMENT"]
    q2_agg = q2_df.groupby(keys, as_index=False)["BILLING"].first()

    # Q3 backfill
    if not q3_df.empty:
        q3_df = q3_df.merge(
            q2_agg.rename(columns={"BILLING": "BILLING_FROM_Q2"}),
            on=keys, how="left"
        )
        # Only backfill where BILLING is 0 or NaN
        mask = q3_df["BILLING"].isna() | (q3_df["BILLING"] == 0)
        q3_df.loc[mask, "BILLING"] = q3_df.loc[mask, "BILLING_FROM_Q2"]
        q3_df = q3_df.drop(columns=["BILLING_FROM_Q2"], errors="ignore")

    # Q4 backfill
    if not q4_df.empty:
        q4_df = q4_df.merge(
            q2_agg.rename(columns={"BILLING": "BILLING_FROM_Q2"}),
            on=keys, how="left"
        )
        mask = q4_df["BILLING"].isna() | (q4_df["BILLING"] == 0)
        q4_df.loc[mask, "BILLING"] = q4_df.loc[mask, "BILLING_FROM_Q2"]
        q4_df = q4_df.drop(columns=["BILLING_FROM_Q2"], errors="ignore")

    return q3_df, q4_df


def calculate_horizon_metrics(pred_df, segment, demand_signals=None, backfill_billing=True, max_horizon=6):
    """
    Calculate MAPE, VALIDATION_DISTANCE, and DIRECTIONAL_CORRECTNESS for each horizon (Q+1 through Q+max_horizon).
    Ported from GENERATE_PREDICTIONS_ITERATIVE_METRICS.ipynb

    Args:
        pred_df: DataFrame with BILLING, BILLING_PREDICTED, QUARTER, MOF_VERSION columns
        segment: Segment name for labeling
        demand_signals: List of demand signal columns for indicator calculation (default: MOF only)
        backfill_billing: Whether to backfill Q3/Q4 billing from Q2 (default: True)
        max_horizon: Maximum horizon to calculate metrics for (default 6 for Q+1 to Q+6)

    Returns:
        Dict with MARKET_SEGMENT and per-horizon metrics:
        - Q+N MAPE
        - Q+N MOF VALIDATION DISTANCE (avg)
        - Q+N MOF DIRECTIONAL CORRECTNESS (avg)
        (and same for other demand signals if provided)
    """
    if demand_signals is None:
        demand_signals = ["MOF"]

    df = create_quarter_indicators(pred_df, max_horizon=max_horizon)

    metrics = {"MARKET_SEGMENT": segment}

    # If create_quarter_indicators returns None, can't calculate horizon metrics
    if df is None:
        for horizon in range(1, max_horizon + 1):
            metrics[f"Q+{horizon} MAPE"] = np.nan
            for signal in demand_signals:
                metrics[f"Q+{horizon} {signal} VALIDATION DISTANCE"] = np.nan
                metrics[f"Q+{horizon} {signal} DIRECTIONAL CORRECTNESS"] = np.nan
        return metrics

    # Ensure demand signal columns exist
    for signal in demand_signals:
        if signal not in df.columns:
            df[signal] = np.nan

    # Calculate indicators for all demand signals
    df = calculate_indicators(df, demand_signals)

    # Split by horizon - create dataframes for each Q+N
    horizon_dfs = {}
    for h in range(1, max_horizon + 1):
        indicator_col = f"Q+{h}_INDICATOR"
        if indicator_col in df.columns:
            horizon_dfs[h] = df[df[indicator_col] == True].copy()
        else:
            horizon_dfs[h] = pd.DataFrame()

    # Backfill Q3/Q4 billing from Q2 if enabled
    if backfill_billing and 2 in horizon_dfs and 3 in horizon_dfs and 4 in horizon_dfs:
        horizon_dfs[3], horizon_dfs[4] = _backfill_billing_from_q2(
            df, horizon_dfs[2], horizon_dfs[3], horizon_dfs[4]
        )

    for horizon in range(1, max_horizon + 1):
        horizon_df = horizon_dfs.get(horizon, pd.DataFrame())

        if horizon_df.empty or horizon_df['BILLING'].isna().all():
            metrics[f"Q+{horizon} MAPE"] = np.nan
            for signal in demand_signals:
                metrics[f"Q+{horizon} {signal} VALIDATION DISTANCE"] = np.nan
                metrics[f"Q+{horizon} {signal} DIRECTIONAL CORRECTNESS"] = np.nan
            continue

        # Calculate MAPE for this horizon
        mape_series = calculate_mape(horizon_df['BILLING'], horizon_df['BILLING_PREDICTED'])
        if len(mape_series) > 0 and not mape_series.isna().all():
            metrics[f"Q+{horizon} MAPE"] = mape_series.mean()
        else:
            metrics[f"Q+{horizon} MAPE"] = np.nan

        # Calculate avg VALIDATION_DISTANCE and DIRECTIONAL_CORRECTNESS per demand signal
        for signal in demand_signals:
            affix = f"_{signal}_REFERENCED"
            val_dist_col = f"VALIDATION_DISTANCE{affix}"
            dir_corr_col = f"DIRECTIONAL_CORRECTNESS{affix}"

            # Avg validation distance
            if val_dist_col in horizon_df.columns:
                val_dist = horizon_df[val_dist_col].dropna()
                metrics[f"Q+{horizon} {signal} VALIDATION DISTANCE"] = val_dist.mean() if len(val_dist) > 0 else np.nan
            else:
                metrics[f"Q+{horizon} {signal} VALIDATION DISTANCE"] = np.nan

            # Avg directional correctness (proportion True)
            if dir_corr_col in horizon_df.columns:
                dir_corr = horizon_df[dir_corr_col].dropna()
                if len(dir_corr) > 0:
                    metrics[f"Q+{horizon} {signal} DIRECTIONAL CORRECTNESS"] = dir_corr.mean()
                else:
                    metrics[f"Q+{horizon} {signal} DIRECTIONAL CORRECTNESS"] = np.nan
            else:
                metrics[f"Q+{horizon} {signal} DIRECTIONAL CORRECTNESS"] = np.nan

    return metrics


def create_backtest_windows(df, n_windows, stride_quarters=1, min_train_quarters=6, max_horizon=6):
    """Generate rolling evaluation window specs by stepping back through time.

    Example with 23 quarters, n_windows=4, stride=1, max_horizon=6:
        Window 0: eval quarters 18-23 (most recent 6)
        Window 1: eval quarters 17-22
        Window 2: eval quarters 16-21
        Window 3: eval quarters 15-20
    """
    quarters = sorted(df['QUARTER'].dropna().unique().tolist())
    total_quarters = len(quarters)

    windows = []
    for i in range(n_windows):
        offset = i * stride_quarters
        eval_end_idx = total_quarters - offset
        eval_start_idx = eval_end_idx - max_horizon

        if eval_start_idx < min_train_quarters:
            break

        eval_quarters = quarters[eval_start_idx:eval_end_idx]
        windows.append({
            'window_idx': i,
            'eval_start_quarter': eval_quarters[0],
            'eval_end_quarter': eval_quarters[-1],
            'eval_quarters': eval_quarters,
        })
    return windows
