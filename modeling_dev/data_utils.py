# data_utils.py

import json
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window


def load_latest_version(spark, table_name: str, select_columns: list = None,
                        is_synthetic_data: bool = False) -> tuple:
    """
    Load latest version from a Delta table.

    Args:
        spark: SparkSession
        table_name: Fully qualified table name (catalog.schema.table)
        select_columns: Optional list of columns to select
        is_synthetic_data: If True, apply MA imputation to fix synthetic/masked data
                          where MAs were computed from historical BB we don't have

    Returns (dataframe, version_string) tuple.
    """
    df = spark.table(table_name)

    # Rename columns (replace - with MINUS)
    for col_name in df.columns:
        if "-" in col_name:
            df = df.withColumnRenamed(col_name, col_name.replace("-", "MINUS"))

    # Get latest version
    df = df.withColumn("VERSION_NUM", F.regexp_extract("VERSION", r"V_(\d+)", 1).cast("int"))
    max_version = df.agg(F.max("VERSION_NUM")).collect()[0][0]
    data_version = f"V_{max_version}"
    df = df.filter(F.col("VERSION_NUM") == max_version).drop("VERSION_NUM")

    # Derive SCENARIO from MOF_VERSION and QUARTER if needed
    df = _ensure_scenario_column(df)

    # Forward-fill BB zeros (matches original preprocessing)
    df = _ffill_bb_zeros(df)

    # Select columns if specified
    if select_columns:
        # Make sure SCENARIO is included if we derived it
        if "SCENARIO" not in select_columns and "SCENARIO" in df.columns:
            select_columns = list(select_columns) + ["SCENARIO"]
        existing_cols = [c for c in select_columns if c in df.columns]
        df = df.select(*existing_cols)

    # Apply synthetic data imputation if requested
    if is_synthetic_data:
        df = _impute_synthetic_mas(df)

    return df, data_version


def _ensure_scenario_column(df: DataFrame) -> DataFrame:
    """
    Ensure SCENARIO column exists and is properly derived from MOF_VERSION and QUARTER.

    The SCENARIO format is: MOF_{mof_date}_QUARTER_{quarter_date}

    This handles:
    1. Missing SCENARIO column - derives from MOF_VERSION and QUARTER
    2. Malformed SCENARIO (e.g., all nulls) - re-derives from MOF_VERSION and QUARTER
    """
    has_mof = "MOF_VERSION" in df.columns
    has_quarter = "QUARTER" in df.columns

    if not has_mof or not has_quarter:
        # Can't derive SCENARIO without these columns
        return df

    # Check if SCENARIO column exists and has valid data
    if "SCENARIO" in df.columns:
        # Check if SCENARIO values look valid (should contain "MOF_" and "_QUARTER_")
        sample = df.select("SCENARIO").limit(10).collect()
        valid_count = sum(1 for row in sample if row[0] and "MOF_" in str(row[0]) and "_QUARTER_" in str(row[0]) and "XXXX-XX-XX" not in str(row[0]))
        if valid_count >= 5:
            # SCENARIO looks valid, don't override
            return df
        # SCENARIO exists but looks invalid, will re-derive below

    # Derive SCENARIO from MOF_VERSION and QUARTER
    # Format: MOF_{mof_version}_QUARTER_{quarter}
    df = df.withColumn(
        "SCENARIO",
        F.concat(
            F.lit("MOF_"), F.col("MOF_VERSION")
        )
    )

    return df


def _ffill_bb_zeros(df: DataFrame) -> DataFrame:
    """
    Forward-fill BB zeros within each partition.

    Matches original preprocessing: replace BB=0 with null, then ffill.
    This handles future quarters where actuals aren't available yet.
    """
    if "BB" not in df.columns:
        return df

    # Need MARKET_SEGMENT and QUARTER_PARSED for partitioning/ordering
    has_segment = "MARKET_SEGMENT" in df.columns
    has_scenario = "SCENARIO" in df.columns
    has_quarter = "QUARTER_PARSED" in df.columns

    if not has_quarter:
        return df

    # Build partition columns
    partition_cols = []
    if has_segment:
        partition_cols.append("MARKET_SEGMENT")
    if has_scenario:
        partition_cols.append("SCENARIO")

    # Replace 0 with null
    df = df.withColumn("BB", F.when(F.col("BB") == 0, None).otherwise(F.col("BB")))

    # Forward-fill using last() over window
    if partition_cols:
        window = Window.partitionBy(*partition_cols).orderBy("QUARTER_PARSED").rowsBetween(Window.unboundedPreceding, 0)
    else:
        window = Window.orderBy("QUARTER_PARSED").rowsBetween(Window.unboundedPreceding, 0)

    df = df.withColumn("BB", F.last("BB", ignorenulls=True).over(window))

    return df


def enable_iceberg(spark, table_name: str, default_type: str = "STRING"):
    """Enable Iceberg/UniForm compatibility on a Delta table (idempotent).
    
    Automatically casts any void/NullType columns to `default_type` first,
    since IcebergCompatV2 does not support the void data type.
    """
    try:
        # Fix void columns that would block Iceberg compatibility
        df = spark.table(table_name)
        void_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() == "void"]

        for col_name in void_cols:
            print(f"Casting void column `{col_name}` to {default_type} in {table_name}")
            spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN `{col_name}` TYPE {default_type}")

        # Step 1: Disable deletion vectors and enable column mapping
        spark.sql(f"""
            ALTER TABLE {table_name} SET TBLPROPERTIES (
                'delta.enableDeletionVectors' = 'false',
                'delta.columnMapping.mode' = 'name'
            )
        """)
        # Step 2: Purge existing deletion vectors
        spark.sql(f"REORG TABLE {table_name} APPLY (PURGE)")
        # Step 3: Enable Iceberg compatibility
        spark.sql(f"""
            ALTER TABLE {table_name} SET TBLPROPERTIES (
                'delta.enableIcebergCompatV2' = 'true',
                'delta.universalFormat.enabledFormats' = 'iceberg'
            )
        """)
    except Exception as e:
        print(f"Warning: Could not enable Iceberg on {table_name}: {e}")


def _impute_synthetic_mas(df: DataFrame) -> DataFrame:
    """
    Impute moving average columns for synthetic/masked data.

    The MAs in synthetic data were computed from historical BB values we don't have.
    This function imputes those historical values and recomputes the MAs to restore
    the correct mathematical relationship: MA_NQ = rolling_mean(BB, N).

    Must be called after data is loaded and before training.
    """
    # Convert to pandas for the imputation logic
    pdf = df.toPandas()

    # Check required columns exist
    ma_cols = ["BILLING_MOVING_AVERAGE_2Q", "BILLING_MOVING_AVERAGE_3Q",
               "BILLING_MOVING_AVERAGE_4Q", "BILLING_MOVING_AVERAGE_5Q"]
    required = ["BB", "QUARTER_PARSED", "MARKET_SEGMENT", "MOF_VERSION"] + ma_cols

    missing = [c for c in required if c not in pdf.columns]
    if missing:
        print(f"Warning: Cannot impute MAs, missing columns: {missing}")
        return df

    # Apply imputation per partition
    result_dfs = []
    for (segment, mof_version), seg_df in pdf.groupby(["MARKET_SEGMENT", "MOF_VERSION"]):
        seg_df = _impute_partition_mas(seg_df)
        result_dfs.append(seg_df)

    pdf_imputed = pd.concat(result_dfs, ignore_index=True)

    # Convert back to Spark DataFrame
    spark = df.sparkSession
    return spark.createDataFrame(pdf_imputed)


def _impute_partition_mas(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute MAs for a single partition by solving for historical BB values.

    At t=0, we have:
      MA_2Q[0] = (BB[-1] + BB[0]) / 2
      MA_3Q[0] = (BB[-2] + BB[-1] + BB[0]) / 3
      MA_4Q[0] = (BB[-3] + BB[-2] + BB[-1] + BB[0]) / 4
      MA_5Q[0] = (BB[-4] + BB[-3] + BB[-2] + BB[-1] + BB[0]) / 5

    Solving sequentially:
      BB[-1] = 2*MA_2Q[0] - BB[0]
      BB[-2] = 3*MA_3Q[0] - BB[-1] - BB[0]
      BB[-3] = 4*MA_4Q[0] - BB[-2] - BB[-1] - BB[0]
      BB[-4] = 5*MA_5Q[0] - BB[-3] - BB[-2] - BB[-1] - BB[0]
    """
    seg_df = seg_df.sort_values("QUARTER_PARSED").reset_index(drop=True)

    bb_0 = seg_df["BB"].iloc[0]
    ma_2q_0 = seg_df["BILLING_MOVING_AVERAGE_2Q"].iloc[0]
    ma_3q_0 = seg_df["BILLING_MOVING_AVERAGE_3Q"].iloc[0]
    ma_4q_0 = seg_df["BILLING_MOVING_AVERAGE_4Q"].iloc[0]
    ma_5q_0 = seg_df["BILLING_MOVING_AVERAGE_5Q"].iloc[0]

    # Check for nulls - if any are null, skip imputation for this partition
    if any(pd.isna([bb_0, ma_2q_0, ma_3q_0, ma_4q_0, ma_5q_0])):
        return seg_df

    # Solve for historical BB values
    bb_minus1 = 2 * ma_2q_0 - bb_0
    bb_minus2 = 3 * ma_3q_0 - bb_minus1 - bb_0
    bb_minus3 = 4 * ma_4q_0 - bb_minus2 - bb_minus1 - bb_0
    bb_minus4 = 5 * ma_5q_0 - bb_minus3 - bb_minus2 - bb_minus1 - bb_0

    historical_bb = [bb_minus4, bb_minus3, bb_minus2, bb_minus1]

    # Recompute MAs with historical context
    full_bb = np.concatenate([historical_bb, seg_df["BB"].values])
    n_hist = len(historical_bb)

    ma_2q_full = pd.Series(full_bb).rolling(2, min_periods=1).mean().values
    ma_3q_full = pd.Series(full_bb).rolling(3, min_periods=1).mean().values
    ma_4q_full = pd.Series(full_bb).rolling(4, min_periods=1).mean().values
    ma_5q_full = pd.Series(full_bb).rolling(5, min_periods=1).mean().values

    # Replace MA columns with imputed values
    seg_df = seg_df.copy()
    seg_df["BILLING_MOVING_AVERAGE_2Q"] = ma_2q_full[n_hist:]
    seg_df["BILLING_MOVING_AVERAGE_3Q"] = ma_3q_full[n_hist:]
    seg_df["BILLING_MOVING_AVERAGE_4Q"] = ma_4q_full[n_hist:]
    seg_df["BILLING_MOVING_AVERAGE_5Q"] = ma_5q_full[n_hist:]

    return seg_df


def write_pipeline_errors(spark, catalog: str, schema: str, errors: list, notebook: str, run_id: str):
    """
    Write pipeline errors to the pipeline_errors Delta table.

    Args:
        spark: SparkSession
        catalog: Unity Catalog name
        schema: Schema name
        errors: List of error dicts with keys: stage, partition, model_type, severity, message
        notebook: Notebook name (e.g., "02_model_training")
        run_id: MLflow parent run ID
    """
    if not errors:
        return

    from datetime import datetime
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType

    # Build records with full context
    records = []
    ts = datetime.now()
    for e in errors:
        records.append({
            "notebook": notebook,
            "run_id": run_id or "N/A",
            "stage": e.get("stage", "unknown"),
            "partition": e.get("partition", "N/A"),
            "model_type": e.get("model_type", "N/A"),
            "severity": e.get("severity", "error"),
            "message": str(e.get("message", ""))[:1000],  # Truncate long messages
            "timestamp": ts,
        })

    schema_def = StructType([
        StructField("notebook", StringType()),
        StructField("run_id", StringType()),
        StructField("stage", StringType()),
        StructField("partition", StringType()),
        StructField("model_type", StringType()),
        StructField("severity", StringType()),
        StructField("message", StringType()),
        StructField("timestamp", TimestampType()),
    ])

    errors_df = spark.createDataFrame(records, schema_def)
    table_name = f"{catalog}.{schema}.pipeline_errors"

    try:
        errors_df.write.mode("append").saveAsTable(table_name)
        enable_iceberg(spark, table_name)
        print(f"Logged {len(errors)} errors/warnings to {table_name}")
    except Exception as e:
        print(f"Warning: Could not write to pipeline_errors table: {e}")
        # Still print errors so they're visible in logs
        for record in records:
            print(f"  [{record['severity'].upper()}] {record['stage']}/{record['partition']}: {record['message']}")


# =============================================================================
# Helper functions for common operations across notebooks
# =============================================================================

def truncate_partition_name(partition: str, max_len: int = 40) -> str:
    """Truncate partition name for display with ellipsis."""
    return partition[:max_len] + "..." if len(partition) > max_len else partition


def sanitize_partition_name(partition: str) -> str:
    """Sanitize partition name for MLflow model naming (uppercase, no special chars)."""
    return partition.upper().replace("/", "_").replace(" ", "_").replace("-", "_")


def safe_json_loads(value, default=None):
    """Safely parse JSON, returning default if None/empty."""
    return json.loads(value) if value else (default if default is not None else {})


def is_valid_metric(value) -> bool:
    """Check if metric value is valid (not None/NaN)."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


def prepare_training_data(df: pd.DataFrame, sort_col: str = 'QUARTER_PARSED') -> tuple:
    """
    Common data preparation for model training: sort, clean, extract horizons.

    Args:
        df: Input DataFrame
        sort_col: Column to sort by (default: 'QUARTER_PARSED', use 'ds' for Prophet)

    Returns:
        Tuple of (prepared_df, validation_horizon, prediction_horizon)
    """
    from modeling_dev.modeling_utils import extract_horizons

    data = df.copy().sort_values(by=sort_col)

    if 'PARTITION_COLUMN' in data.columns:
        data = data.drop(columns=['PARTITION_COLUMN'])

    val_h, pred_h = extract_horizons(data)

    return data, val_h, pred_h


# =============================================================================
# Data Quality Checks (MMF-style pre-training validation)
# =============================================================================

def run_data_quality_checks(
    df: pd.DataFrame,
    target_col: str = 'BB',
    regressor_cols: list = None,
    date_col: str = 'QUARTER_PARSED',
    group_col: str = 'PARTITION_COLUMN',
    negative_threshold: float = 0.2,
    min_train_periods: int = 6,
    raise_on_failure: bool = False,
    val_horizon_col: str = 'VERIFICATION_HORIZON',
    pred_horizon_col: str = 'PREDICTION_HORIZON',
) -> tuple:
    """
    MMF-style data quality validation before training.

    Validates:
    1. Null values in target column (removes group)
    2. Null values in regressor columns (removes group)
    3. Negative target values exceeding threshold (warning only)
    4. Insufficient training periods (removes group)
    5. Insufficient training rows after train/val/test split (removes group)

    Args:
        df: Input DataFrame (pandas)
        target_col: Target column name (default: 'BB')
        regressor_cols: List of regressor columns to check for nulls
        date_col: Date column for ordering (default: 'QUARTER_PARSED')
        group_col: Partition/group column (default: 'PARTITION_COLUMN')
        negative_threshold: Warn if negative ratio exceeds this (default: 0.2)
        min_train_periods: Minimum rows required per group (default: 6)
        raise_on_failure: If True, raise ValueError when groups are removed
        val_horizon_col: Column name for validation horizon (default: 'VERIFICATION_HORIZON')
        pred_horizon_col: Column name for prediction horizon (default: 'PREDICTION_HORIZON')

    Returns:
        Tuple of (clean_df, report_dict) where report_dict contains:
        - groups_removed: list of partition names removed
        - removal_reasons: dict mapping partition -> reason
        - warnings: list of non-fatal warning messages
        - summary: dict with counts
    """
    report = {
        'groups_removed': [],
        'removal_reasons': {},
        'warnings': [],
        'summary': {
            'total_groups': 0,
            'groups_removed': 0,
            'groups_remaining': 0,
        }
    }

    if group_col not in df.columns:
        report['warnings'].append(f"Group column '{group_col}' not found, skipping quality checks")
        return df, report

    groups = df[group_col].unique()
    report['summary']['total_groups'] = len(groups)

    # Check 1: Null values in target
    if target_col in df.columns:
        null_target = df.groupby(group_col)[target_col].apply(lambda x: x.isnull().sum())
        bad_groups = null_target[null_target > 0].index.tolist()
        for g in bad_groups:
            if g not in report['groups_removed']:
                report['groups_removed'].append(g)
                report['removal_reasons'][g] = f"Null values in {target_col} ({null_target[g]} nulls)"

    # Check 2: Null values in regressors
    if regressor_cols:
        for col in regressor_cols:
            if col not in df.columns:
                continue
            null_reg = df.groupby(group_col)[col].apply(lambda x: x.isnull().sum())
            bad = null_reg[null_reg > 0].index.tolist()
            for g in bad:
                if g not in report['groups_removed']:
                    report['groups_removed'].append(g)
                    report['removal_reasons'][g] = f"Null values in regressor {col} ({null_reg[g]} nulls)"

    # Check 3: Negative target values (warning only, not removal)
    if target_col in df.columns:
        neg_ratio = df.groupby(group_col)[target_col].apply(
            lambda x: (x < 0).sum() / len(x) if len(x) > 0 else 0
        )
        for g, ratio in neg_ratio.items():
            if ratio > negative_threshold:
                report['warnings'].append(
                    f"{truncate_partition_name(g)}: {ratio:.1%} negative {target_col} values (threshold: {negative_threshold:.0%})"
                )

    # Check 4: Insufficient training periods
    period_count = df.groupby(group_col).size()
    insufficient = period_count[period_count < min_train_periods].index.tolist()
    for g in insufficient:
        if g not in report['groups_removed']:
            report['groups_removed'].append(g)
            report['removal_reasons'][g] = f"Only {period_count[g]} periods (min: {min_train_periods})"

    # Check 5: Insufficient training rows after train/val/test split
    if val_horizon_col in df.columns and pred_horizon_col in df.columns:
        for g in groups:
            if g in report['groups_removed']:
                continue
            gdf = df[df[group_col] == g]
            val_h = int(gdf[val_horizon_col].median())
            pred_h = int(gdf[pred_horizon_col].median())
            total = len(gdf)
            train_rows = total - val_h - pred_h
            if train_rows < min_train_periods:
                report['groups_removed'].append(g)
                report['removal_reasons'][g] = (
                    f"Only {train_rows} training rows after split "
                    f"(val={val_h}, pred={pred_h}, min: {min_train_periods})"
                )

    # Remove bad groups
    if report['groups_removed']:
        clean_df = df[~df[group_col].isin(report['groups_removed'])].copy()
    else:
        clean_df = df.copy()

    # Update summary
    report['summary']['groups_removed'] = len(report['groups_removed'])
    report['summary']['groups_remaining'] = report['summary']['total_groups'] - report['summary']['groups_removed']

    # Optionally raise
    if raise_on_failure and report['groups_removed']:
        raise ValueError(
            f"Data quality check failed: {len(report['groups_removed'])} groups removed. "
            f"Reasons: {report['removal_reasons']}"
        )

    return clean_df, report


def print_data_quality_report(report: dict):
    """Print a formatted data quality report."""
    summary = report['summary']
    print(f"\nData Quality Check Results:")
    print(f"  Total partitions: {summary['total_groups']}")
    print(f"  Partitions removed: {summary['groups_removed']}")
    print(f"  Partitions remaining: {summary['groups_remaining']}")

    if report['groups_removed']:
        print(f"\nRemoved partitions:")
        for partition, reason in report['removal_reasons'].items():
            print(f"  - {truncate_partition_name(partition)}: {reason}")

    if report['warnings']:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  - {warning}")


# =============================================================================
# Prediction Sanity Checks (post-inference output validation)
# =============================================================================

def run_prediction_sanity_checks(
    df: pd.DataFrame,
    negative_check: bool = True,
    extreme_ape_threshold: float = 500,
    flat_check: bool = True,
    flat_min_unique: int = 2,
    min_directional_correctness: float = 0.25,
    max_validation_distance: float = 3.0,
    group_col: str = 'PARTITION_COLUMN',
    prediction_col: str = 'BILLING_PREDICTED',
    actual_col: str = 'BILLING',
    ape_col: str = 'APE',
) -> tuple:
    """
    Post-inference sanity checks on prediction output.

    Validates per partition:
    1. Negative predictions (BILLING_PREDICTED < 0)
    2. Extreme APE (partition mean APE > threshold)
    3. Flat predictions (fewer than flat_min_unique distinct predicted values)
    4. Low directional correctness (mean DIRECTIONAL_CORRECTNESS_* < threshold)
    5. High validation distance (mean VALIDATION_DISTANCE_* > threshold)

    Returns (df_unchanged, report_dict). DataFrame is never modified.
    """
    report = {
        'flagged_partitions': [],
        'partition_details': {},
        'warnings': [],
        'errors_to_log': [],
        'summary': {
            'total_partitions': 0,
            'total_negative_rows': 0,
            'partitions_with_negatives': 0,
            'partitions_extreme_ape': 0,
            'partitions_flat': 0,
            'partitions_low_direction': 0,
            'partitions_high_distance': 0,
        }
    }

    if group_col not in df.columns:
        report['warnings'].append(f"Group column '{group_col}' not found, skipping sanity checks")
        return df, report

    partitions = df[group_col].unique()
    report['summary']['total_partitions'] = len(partitions)

    # Auto-discover indicator columns
    dir_corr_cols = [c for c in df.columns if c.startswith('DIRECTIONAL_CORRECTNESS_')]
    val_dist_cols = [c for c in df.columns if c.startswith('VALIDATION_DISTANCE_')]

    for partition in partitions:
        pdf = df[df[group_col] == partition]
        issues = []

        # Check 1: Negative predictions
        if negative_check and prediction_col in pdf.columns:
            neg_count = (pdf[prediction_col] < 0).sum()
            if neg_count > 0:
                issues.append(f"{neg_count} negative prediction(s)")
                report['summary']['total_negative_rows'] += int(neg_count)
                report['summary']['partitions_with_negatives'] += 1

        # Check 2: Extreme APE
        if extreme_ape_threshold and ape_col in pdf.columns:
            mean_ape = pdf[ape_col].mean()
            if pd.notna(mean_ape) and mean_ape > extreme_ape_threshold:
                issues.append(f"mean APE {mean_ape:.1f}% exceeds {extreme_ape_threshold}%")
                report['summary']['partitions_extreme_ape'] += 1

        # Check 3: Flat predictions
        if flat_check and prediction_col in pdf.columns:
            total_rows = len(pdf)
            n_unique = pdf[prediction_col].nunique()
            if total_rows >= flat_min_unique and n_unique < flat_min_unique:
                issues.append(f"only {n_unique} unique predicted value(s) across {total_rows} rows")
                report['summary']['partitions_flat'] += 1

        # Check 4: Low directional correctness
        if min_directional_correctness and dir_corr_cols:
            vals = pdf[dir_corr_cols].values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                mean_dc = float(np.mean(vals))
                if mean_dc < min_directional_correctness:
                    issues.append(f"directional correctness {mean_dc:.2f} below {min_directional_correctness}")
                    report['summary']['partitions_low_direction'] += 1

        # Check 5: High validation distance
        if max_validation_distance and val_dist_cols:
            vals = pdf[val_dist_cols].values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                mean_vd = float(np.mean(vals))
                if mean_vd > max_validation_distance:
                    issues.append(f"validation distance {mean_vd:.2f} exceeds {max_validation_distance}")
                    report['summary']['partitions_high_distance'] += 1

        if issues:
            report['flagged_partitions'].append(partition)
            report['partition_details'][partition] = issues
            for issue in issues:
                report['errors_to_log'].append({
                    "stage": "prediction_sanity_check",
                    "partition": partition,
                    "model_type": pdf["MODEL_USED"].iloc[0] if "MODEL_USED" in pdf.columns else "N/A",
                    "severity": "warning",
                    "message": issue,
                })

    return df, report


def prepare_for_scoring(df: pd.DataFrame, segment: str, variable_config: dict) -> pd.DataFrame:
    """Rename demand signal columns and adjust regressors for model scoring.

    Consolidates the rename + adjust_regressors pattern used identically in
    evaluate_model (03) and run_inference (04) applyInPandas functions.

    Args:
        df: Pandas DataFrame with raw column names (BB, MOF_0_MONTH, etc.)
        segment: Market segment string for adjust_regressors
        variable_config: Dict from variable_config.yaml (needs DEMAND_SIGNAL_RENAMES, REGRESSOR_COLUMNS)

    Returns:
        DataFrame with renamed columns and adjusted regressors.
    """
    from modeling_dev.modeling_utils import adjust_regressors

    renames = variable_config.get("DEMAND_SIGNAL_RENAMES", {})
    adj_regressors = variable_config.get("REGRESSOR_COLUMNS", [])
    df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns}).copy()
    df = adjust_regressors(df, adj_regressors, 'BILLING_MOVING_AVERAGE_2Q', Market_Segment=segment)
    return df


def orchestrate_dq_checks(
    spark,
    df_raw,
    config: dict,
    stage: str,
    raise_on_failure: bool = False,
) -> tuple:
    """Run data quality checks and filter bad partitions.

    Wraps run_data_quality_checks() + print_data_quality_report() with the
    standard orchestration pattern used across training/eval/inference notebooks.

    Args:
        spark: SparkSession
        df_raw: Spark DataFrame with PARTITION_COLUMN
        config: YAML config dict (reads general.USED_REGRESSORS)
        stage: Stage name for error logging (e.g. "data_quality_training")
        raise_on_failure: Whether to raise on failures

    Returns:
        (df_filtered, dq_errors_to_log, excluded_partitions) where:
        - df_filtered: Spark DataFrame with bad partitions removed (unchanged if none removed)
        - dq_errors_to_log: list of error dicts for write_pipeline_errors()
        - excluded_partitions: set of removed partition names
    """
    regressor_cols = list(config.get("general", {}).get("USED_REGRESSORS", {}).keys())

    df_raw_pdf = df_raw.toPandas()
    df_clean, dq_report = run_data_quality_checks(
        df_raw_pdf,
        target_col='BB',
        regressor_cols=regressor_cols,
        group_col='PARTITION_COLUMN',
        min_train_periods=6,
        raise_on_failure=raise_on_failure,
    )

    print_data_quality_report(dq_report)

    dq_errors_to_log = []
    excluded_partitions = set()

    if dq_report['groups_removed']:
        excluded_partitions = set(dq_report['groups_removed'])
        print(f"\nExcluding {len(excluded_partitions)} partition(s) due to data quality issues:")
        for partition in dq_report['groups_removed']:
            reason = dq_report['removal_reasons'].get(partition, "Unknown reason")
            print(f"  DROPPED: {partition} -- {reason}")
            dq_errors_to_log.append({
                "stage": stage,
                "partition": partition,
                "model_type": None,
                "severity": "warning",
                "message": reason,
            })
        print(f"Proceeding with {dq_report['summary']['groups_remaining']} partition(s)\n")
        df_raw = spark.createDataFrame(df_clean)
    else:
        print("\nAll partitions passed data quality checks.")

    return df_raw, dq_errors_to_log, excluded_partitions


def print_sanity_check_report(report: dict):
    """Print a formatted prediction sanity check report."""
    summary = report['summary']
    flagged = report['flagged_partitions']

    print(f"\nPrediction Sanity Check Results:")
    print(f"  Total partitions checked: {summary['total_partitions']}")
    print(f"  Partitions flagged: {len(flagged)}")

    if summary['total_negative_rows']:
        print(f"  Negative prediction rows: {summary['total_negative_rows']} across {summary['partitions_with_negatives']} partition(s)")
    if summary['partitions_extreme_ape']:
        print(f"  Extreme APE partitions: {summary['partitions_extreme_ape']}")
    if summary['partitions_flat']:
        print(f"  Flat prediction partitions: {summary['partitions_flat']}")
    if summary['partitions_low_direction']:
        print(f"  Low directional correctness: {summary['partitions_low_direction']}")
    if summary['partitions_high_distance']:
        print(f"  High validation distance: {summary['partitions_high_distance']}")

    if flagged:
        print(f"\nFlagged partitions:")
        for partition in flagged:
            issues = report['partition_details'].get(partition, [])
            print(f"  - {truncate_partition_name(partition)}:")
            for issue in issues:
                print(f"      {issue}")

    if not flagged:
        print("  All partitions passed sanity checks.")


# =============================================================================
# Delta Table Schema Utilities
# =============================================================================

def get_next_metrics_version(spark, table_name: str) -> str:
    """Auto-increment metrics version for a Delta table.

    Extracts max version number from "V_N" format, increments it.
    Returns "V_1" on first run (table doesn't exist).
    Also handles schema evolution (adds eval_source column if missing).
    """
    try:
        latest_version_num = (
            spark.table(table_name)
            .select(F.max(F.regexp_extract("METRICS_VERSION", r"V_(\d+)", 1).cast("int")))
            .first()[0]
        )
        new_version = f"V_{(latest_version_num or 0) + 1}"

        # Schema evolution: add eval_source column if table predates it
        existing_cols = {c.name for c in spark.table(table_name).schema}
        if "eval_source" not in existing_cols:
            spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS (eval_source STRING)")
    except Exception:
        new_version = "V_1"

    return new_version


def align_schema_for_delta(spark_df, table_name: str, spark) -> "DataFrame":
    """Cast DataFrame columns to match an existing Delta table schema.

    Handles three issues that arise when writing pandas-derived DataFrames to Delta:
    1. NullType columns (all-None) -- Delta can't write VOID type, cast to StringType
    2. Numeric type mismatches -- pandas None/int mix becomes float64/DoubleType,
       which must match existing IntegerType columns exactly
    3. Stale columns -- columns in the existing table that are not in the new DataFrame
       and have incompatible types from prior schema versions are dropped

    If the table doesn't exist yet (first write), only NullType casting is applied.
    """
    from pyspark.sql.types import NullType

    # Cast NullType columns to StringType
    for field in spark_df.schema:
        if isinstance(field.dataType, NullType):
            spark_df = spark_df.withColumn(field.name, F.col(field.name).cast("string"))

    # Match existing table column types and drop stale conflicting columns
    try:
        existing_schema = {f.name: f.dataType for f in spark.table(table_name).schema}
        df_schema = {f.name: f.dataType for f in spark_df.schema}

        for col_name in spark_df.columns:
            if col_name in existing_schema and not isinstance(existing_schema[col_name], NullType):
                spark_df = spark_df.withColumn(col_name, F.col(col_name).cast(existing_schema[col_name]))

        # Drop columns from the existing table that are not in the new DataFrame.
        # These are stale columns from prior runs that can cause mergeSchema failures
        # (DELTA_FAILED_TO_MERGE_FIELDS) when types have drifted.
        stale_cols = set(existing_schema.keys()) - set(df_schema.keys())
        for col_name in stale_cols:
            try:
                spark.sql(f"ALTER TABLE {table_name} DROP COLUMN `{col_name}`")
                print(f"Dropped stale column '{col_name}' from {table_name}")
            except Exception as e:
                print(f"Could not drop column '{col_name}' from {table_name}: {e}")
    except Exception:
        pass  # Table doesn't exist yet, skip type matching

    return spark_df
