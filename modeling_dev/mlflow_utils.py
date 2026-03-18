# mlflow_utils.py
# Shared MLflow utilities for pipeline notebooks 02-05.

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from typing import Callable, List, Tuple, Optional

import cloudpickle
import numpy as np
import mlflow
from mlflow import MlflowClient
import pandas as pd
from tqdm.auto import tqdm

from modeling_dev.data_utils import (
    sanitize_partition_name, truncate_partition_name,
    safe_json_loads, is_valid_metric,
)

# Concurrency defaults (Databricks rate limits: 120 req/s tracking, 40 req/s registry)
_DEFAULT_MAX_WORKERS = 8
_REGISTRY_MAX_WORKERS = 4


def _resolve_model_binary(model_binary):
    """Deserialize model_binary if it's bytes; return as-is if already an object."""
    if isinstance(model_binary, (bytes, bytearray)):
        return cloudpickle.loads(model_binary)
    return model_binary


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def configure_mlflow_retries():
    """Set MLflow HTTP retry env vars for resilience under parallel load."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "5")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")


def setup_mlflow_client() -> MlflowClient:
    """Set Unity Catalog registry URI, suppress progress bars, configure retries, return client."""
    mlflow.set_registry_uri("databricks-uc")
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    configure_mlflow_retries()
    mlflow.enable_system_metrics_logging()
    return MlflowClient()


def _run_parallel(
    func: Callable,
    items: list,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    desc: str = "items",
) -> list:
    """Execute func(item) in parallel using ThreadPoolExecutor.

    Returns list of (item, result_or_None, error_or_None) tuples in
    completion order. Failures in one item do not crash others.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        with tqdm(total=len(items), desc=desc, unit="model") as pbar:
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append((item, result, None))
                except Exception as e:
                    results.append((item, None, e))
                    print(f"  Error processing {desc}: {e}")
                pbar.update(1)
    return results


def get_or_create_experiment(experiment_name: str) -> str:
    """Get or create an MLflow experiment, set it globally, return experiment_id.

    Uses set_experiment(experiment_id=...) so nested runs work in Databricks
    job context.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


# ---------------------------------------------------------------------------
# Run search helpers
# ---------------------------------------------------------------------------

def find_latest_training_run(
    experiment_name: str,
    scoring_table: str,
    parent_run_id: str = None,
) -> Tuple[str, str]:
    """Find the latest parent training run by scoring_table tag.

    If parent_run_id is provided (e.g. passed via dbutils.jobs.taskValues),
    skips the MLflow search and returns it directly. This prevents race
    conditions when two pipeline runs execute concurrently in the same experiment.

    Returns (parent_run_id, experiment_id).
    Raises ValueError if no training runs found.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    if parent_run_id:
        return parent_run_id, experiment.experiment_id

    parent_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.scoring_table = '{scoring_table}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if parent_df.empty:
        raise ValueError("No training runs found in experiment")

    return parent_df.iloc[0]["run_id"], experiment.experiment_id


def get_nested_runs(experiment_id: str, parent_run_id: str) -> pd.DataFrame:
    """Get all nested runs (one per model) from a parent training run."""
    return mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )


def resolve_champion_and_mape_from_runs(
    nested_df: pd.DataFrame,
) -> dict:
    """Build champion status and MAPE lookup from the nested runs DataFrame.

    Reads tags.is_champion and metrics.eval_horizon_mape directly from the
    DataFrame returned by search_runs(), avoiding per-run API calls.

    Returns dict keyed by run_id:
        {"run_id": {"is_champion": bool, "mape": float|None}}
    """
    lookup = {}
    for _, row in nested_df.iterrows():
        run_id = row["run_id"]
        is_champ_raw = row.get("tags.is_champion", "")
        is_champion = str(is_champ_raw).strip().lower() == "true"

        mape = row.get("metrics.eval_horizon_mape")
        if not is_valid_metric(mape):
            mape = row.get("metrics.eval_mape")
        if not is_valid_metric(mape):
            mape = None

        lookup[run_id] = {"is_champion": is_champion, "mape": mape}
    return lookup


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def build_registered_model_name(
    catalog: str,
    schema: str,
    prefix: str,
    partition_clean: str,
    model_type: str,
) -> str:
    """Build fully-qualified Unity Catalog model name.

    Convention: catalog.schema.prefix_PARTITIONNAME_modeltype
    """
    return f"{catalog}.{schema}.{prefix}_{partition_clean}_{model_type.lower()}"


def sanitize_metric_name(name: str, max_len: int = 50) -> str:
    """Sanitize a feature/regressor name for use as an MLflow metric key."""
    return name.replace("/", "_").replace(" ", "_").replace("-", "_").lower()[:max_len]


# ---------------------------------------------------------------------------
# Lineage helper
# ---------------------------------------------------------------------------

def log_delta_lineage(table_context_pairs: List[Tuple[str, str]]) -> None:
    """Log Delta tables as MLflow datasets for lineage tracking.

    Must be called inside an active mlflow.start_run() context.

    Args:
        table_context_pairs: List of (table_name, context_string) tuples,
            e.g. [("catalog.schema.data", "training_data")].
    """
    try:
        import mlflow.data
        for table_name, context in table_context_pairs:
            dataset = mlflow.data.load_delta(table_name=table_name)
            mlflow.log_input(dataset, context=context)
    except Exception as e:
        print(f"Warning: Could not log Delta tables to MLflow: {e}")


# ---------------------------------------------------------------------------
# Model logging
# ---------------------------------------------------------------------------

def log_model_to_mlflow(
    row: dict,
    model_name_prefix: str,
    catalog: str,
    schema: str,
    data_table: str,
    data_version: str,
    parent_run_id: str,
    experiment_id: str,
    horizon_range_str: str,
    experiment_name: str,
    log_artifacts: bool = False,
    register_to_registry: bool = False,
) -> Optional[dict]:
    """Create a nested MLflow run with params, metrics, and optionally register the model."""
    partition = row["PARTITION_COLUMN"]
    model_type = row["MODEL_USED"]

    if row["error"]:
        print(f"  [{partition}] {model_type} skipped (error: {row['error']})")
        return None

    # Parse enriched data from training functions
    metrics = safe_json_loads(row["training_metrics"])
    params = safe_json_loads(row["used_params"])
    model_config = safe_json_loads(row["model_config"])
    mape = row["mape"]

    partition_clean = sanitize_partition_name(partition)
    run_name = f"{partition_clean}_{model_type.lower()}"

    # Create nested run. Uses explicit parentRunId tag instead of nested=True
    # so this works correctly from both the main thread and worker threads
    # (thread-local run stacks don't share the parent run across threads).
    with mlflow.start_run(
        run_name=run_name,
        experiment_id=experiment_id,
        tags={"mlflow.parentRunId": parent_run_id},
    ) as nested_run:
        # --- Params (batched into a single API call) ---
        selected = metrics.get("regressors", [])
        dropped = metrics.get("dropped_features", [])

        all_params = {
            "partition": partition,
            "segment": model_config.get("segment", ""),
            "model_type": model_type,
            "data_table": data_table,
            "data_version": data_version,
            "validation_horizon": metrics.get("validation_horizon"),
            "prediction_horizon": metrics.get("prediction_horizon"),
            "horizon_range": horizon_range_str,
            "grid_search_enabled": metrics.get("grid_search_enabled", False),
            "selected_features": ", ".join(selected) if selected else "",
            "dropped_features": ", ".join(dropped) if dropped else "",
            "n_features": len(selected),
        }

        # Model-specific hyperparameters
        for key, value in params.items():
            if key in ["regressors", "model_class"]:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                all_params[key] = value

        if metrics.get("cap") is not None:
            all_params["cap"] = metrics.get("cap")

        mlflow.log_params(all_params)

        # --- Metrics (batched into a single API call) ---
        all_metrics = {}
        if is_valid_metric(mape):
            all_metrics["mape"] = mape
        if metrics.get("grid_search_val_mae") is not None:
            all_metrics["grid_search_val_mae"] = metrics["grid_search_val_mae"]
        if metrics.get("best_iteration") is not None:
            all_metrics["best_iteration"] = metrics["best_iteration"]
        if metrics.get("grid_search_combos"):
            all_metrics["grid_search_combos"] = metrics["grid_search_combos"]
        if metrics.get("grid_search_failures"):
            all_metrics["grid_search_failures"] = metrics["grid_search_failures"]

        if all_metrics:
            mlflow.log_metrics(all_metrics)

        # --- Grid search results artifact (top 10 combos) ---
        cv_summary = metrics.get("cv_results_summary")
        if cv_summary:
            mlflow.log_dict(cv_summary, artifact_file="grid_search/cv_results.json")

        # --- Predictions sample artifact ---
        predictions_sample = row.get("predictions_sample")
        if predictions_sample:
            mlflow.log_table(pd.read_json(StringIO(predictions_sample)), artifact_file="predictions/sample.json")

        # --- Deserialize model once for feature importance + artifact logging ---
        pyfunc_model = None
        if row.get("model_binary"):
            pyfunc_model = _resolve_model_binary(row["model_binary"])

        # --- Feature importance artifact (sklearn models with feature_importances_) ---
        if pyfunc_model is not None:
            try:
                raw_model = pyfunc_model.model
                if hasattr(raw_model, "feature_importances_"):
                    regressors = metrics.get("regressors", [])
                    importance_df = pd.DataFrame({
                        "feature": regressors,
                        "importance": raw_model.feature_importances_
                    }).sort_values("importance", ascending=False)
                    mlflow.log_table(importance_df, artifact_file="feature_importance/importance.json")
            except Exception:
                pass

        # --- Model artifact (optional, slow) ---
        registered_model_name = None
        model_version = None
        if log_artifacts and pyfunc_model is not None:
            input_example = pd.read_json(StringIO(row["sample_input"]))
            artifact_name = f"{partition_clean}_{model_type.lower()}"

            logged_model_tags = {
                "segment": pyfunc_model.model_config.get("segment", ""),
                "model_type": model_type.lower(),
                "partition": partition,
            }
            mlflow.pyfunc.log_model(
                name=artifact_name,
                python_model=pyfunc_model,
                input_example=input_example,
                pip_requirements=["prophet", "xgboost", "scikit-learn", "pandas", "numpy", "cloudpickle", "statsforecast", "catboost", "lightgbm", "neuralforecast"],
                tags=logged_model_tags,
            )

            if register_to_registry:
                registered_model_name = build_registered_model_name(catalog, schema, model_name_prefix, partition_clean, model_type)
                model_uri = f"runs:/{nested_run.info.run_id}/{artifact_name}"
                registered_version = mlflow.register_model(model_uri, registered_model_name)
                model_version = registered_version.version
                mlflow.set_tag("registry_version", str(model_version))

                client = mlflow.MlflowClient()
                segment = model_config.get("segment", "")
                client.set_registered_model_tag(name=registered_model_name, key="segment", value=segment)
                client.set_registered_model_tag(name=registered_model_name, key="model_type", value=model_type.lower())
                client.set_registered_model_tag(name=registered_model_name, key="catalog", value=catalog)
                client.set_registered_model_tag(name=registered_model_name, key="schema", value=schema)
                client.set_registered_model_tag(name=registered_model_name, key="data_table", value=data_table)
                client.set_registered_model_tag(name=registered_model_name, key="experiment_name", value=experiment_name)
                client.set_registered_model_tag(name=registered_model_name, key="grid_search_enabled", value=str(metrics.get("grid_search_enabled", False)))

                client.set_model_version_tag(name=registered_model_name, version=model_version, key="mape", value=str(mape) if is_valid_metric(mape) else "N/A")
                client.set_model_version_tag(name=registered_model_name, version=model_version, key="partition", value=str(partition))

    partition_short = truncate_partition_name(partition)
    mape_str = f"{mape:.2f}%" if is_valid_metric(mape) else "N/A"
    artifact_str = " [+artifact]" if log_artifacts else ""
    registry_str = f" -> {registered_model_name.split('.')[-1]}" if registered_model_name else ""
    print(f"  [{partition_short}] {model_type} MAPE: {mape_str}{artifact_str}{registry_str}")

    return {
        "partition": partition,
        "segment": model_config.get("segment", ""),
        "MODEL_USED": model_type,
        "mape": mape,
        "model_version": model_version,
        "registered_model_name": registered_model_name,
        "nested_run_id": nested_run.info.run_id,
    }


def log_models_to_mlflow_parallel(
    trained_models: list,
    model_name_prefix: str,
    catalog: str,
    schema: str,
    data_table: str,
    data_version: str,
    parent_run_id: str,
    experiment_id: str,
    horizon_range_str: str,
    experiment_name: str,
    log_artifacts: bool = False,
    register_to_registry: bool = False,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> Tuple[list, dict]:
    """Log all trained models to MLflow nested runs in parallel.

    Each thread creates its own nested run under parent_run_id via the
    explicit mlflow.parentRunId tag (no dependency on thread-local run stack).

    Returns (results_list, nested_run_ids_dict) matching the serial interface.
    """
    if register_to_registry:
        max_workers = min(max_workers, _REGISTRY_MAX_WORKERS)

    def _log_one(item):
        idx, row = item
        mlflow.set_experiment(experiment_id=experiment_id)
        return log_model_to_mlflow(
            row,
            model_name_prefix,
            catalog,
            schema,
            data_table,
            data_version,
            parent_run_id=parent_run_id,
            experiment_id=experiment_id,
            horizon_range_str=horizon_range_str,
            experiment_name=experiment_name,
            log_artifacts=log_artifacts,
            register_to_registry=register_to_registry,
        )

    items = list(enumerate(trained_models))
    parallel_results = _run_parallel(_log_one, items, max_workers=max_workers, desc="model logging")

    results = []
    nested_run_ids = {}
    for (idx, _row), result, error in parallel_results:
        if result:
            results.append(result)
            nested_run_ids[idx] = result["nested_run_id"]
        elif error:
            print(f"  Model {idx} logging failed: {error}")

    return results, nested_run_ids


# ---------------------------------------------------------------------------
# High-level model loading
# ---------------------------------------------------------------------------

def _resolve_nested_runs(experiment_name, scoring_table, parent_run_id=None):
    """Find training run, get nested runs, return valid (partition, model_type, row) tuples."""
    parent_run_id, experiment_id = find_latest_training_run(experiment_name, scoring_table, parent_run_id)
    print(f"Latest training run: {parent_run_id}")

    nested_df = get_nested_runs(experiment_id, parent_run_id)
    print(f"Found {len(nested_df)} models in this run")

    valid_items = []
    for _, run_row in nested_df.iterrows():
        partition = run_row.get("params.partition")
        model_type = (run_row.get("params.model_type") or "").lower()
        if not partition or not model_type:
            continue
        valid_items.append((partition, model_type, run_row))

    return parent_run_id, experiment_id, nested_df, valid_items


def load_models_from_latest_run_parallel(
    client: MlflowClient,
    experiment_name: str,
    scoring_table: str,
    catalog: str,
    schema: str,
    model_name_prefix: str,
    parent_run_id: str = None,
) -> Tuple[list, str]:
    """Build metadata-only entries for all models in the latest training run.

    No artifact downloads -- callers load artifacts serially on the driver.
    parent_run_id: if provided, skips find_latest_training_run query (see that
    function's docstring for why this matters with concurrent pipeline runs).
    """
    parent_run_id, experiment_id, nested_df, valid_items = _resolve_nested_runs(
        experiment_name, scoring_table, parent_run_id
    )

    models = []
    for partition, model_type, run_row in valid_items:
        nrid = run_row["run_id"]
        partition_clean = sanitize_partition_name(partition)
        registered_name = build_registered_model_name(
            catalog, schema, model_name_prefix, partition_clean, model_type
        )
        v = run_row.get("tags.registry_version")
        models.append({
            "partition": partition,
            "model_type": model_type,
            "pyfunc_model": None,
            "registered_name": registered_name,
            "version": v,
            "nested_run_id": nrid,
            "deferred": True,
            "artifact_uri": f"runs:/{nrid}/{partition_clean}_{model_type}",
        })
        v_str = f"(v{v})" if v else "(unregistered)"
        print(f"  Queued {truncate_partition_name(partition)}... {model_type.upper()} {v_str}")

    return models, parent_run_id


def load_champion_models_parallel(
    client: MlflowClient,
    experiment_name: str,
    scoring_table: str,
    catalog: str,
    schema: str,
    model_name_prefix: str,
    parent_run_id: str = None,
) -> list:
    """Load champion models for partitions in the latest training run (serial).

    Uses is_champion tag from nested runs (set by eval notebook 03) to identify
    champions, then loads model artifacts from the run serially on the driver.

    Falls back to @champion alias lookup if no is_champion tags found (backward
    compatibility with runs evaluated before this fix).
    parent_run_id: if provided, skips find_latest_training_run query (see that
    function's docstring for why this matters with concurrent pipeline runs).
    """
    parent_run_id, experiment_id, nested_df, valid_items = _resolve_nested_runs(
        experiment_name, scoring_table, parent_run_id
    )

    # Identify champions via is_champion tag (set by eval notebook 03)
    champion_lookup = resolve_champion_and_mape_from_runs(nested_df)

    work_items = []
    for partition, model_type, run_row in valid_items:
        champ_info = champion_lookup.get(run_row["run_id"], {})
        if champ_info.get("is_champion"):
            work_items.append(run_row)

    # Fallback: if no is_champion tags found, use @champion alias lookup
    use_alias_fallback = False
    if not work_items:
        print("No is_champion tags found; falling back to @champion alias lookup...")
        use_alias_fallback = True
        for partition, model_type, run_row in valid_items:
            work_items.append(run_row)
    else:
        print(f"Found {len(work_items)} champion model(s) by is_champion tag")

    def _load_champion(run_row):
        nrid = run_row["run_id"]
        partition = run_row.get("params.partition")
        model_type = (run_row.get("params.model_type") or "").lower()
        partition_clean = sanitize_partition_name(partition)
        model_name = build_registered_model_name(
            catalog, schema, model_name_prefix, partition_clean, model_type
        )

        if use_alias_fallback:
            # Legacy: check for @champion alias in registry
            try:
                version_info = client.get_model_version_by_alias(model_name, "champion")
            except Exception:
                return None
            loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
            version = version_info.version
        else:
            # Primary: load from run artifact (consistent with NB03)
            artifact_name = f"{partition_clean}_{model_type}"
            loaded_model = mlflow.pyfunc.load_model(f"runs:/{nrid}/{artifact_name}")
            version = nrid

        pyfunc_model = loaded_model.unwrap_python_model()

        return {
            "partition": partition,
            "model_type": model_type,
            "pyfunc_model": pyfunc_model,
            "model_name": model_name,
            "version": version,
            "nested_run_id": nrid,
        }

    models = []
    seen_partitions = {}
    for run_row in work_items:
        try:
            result = _load_champion(run_row)
        except Exception as e:
            partition = run_row.get("params.partition", "unknown")
            print(f"  Skipping {partition}: {e}")
            continue
        if result is None:
            continue
        partition = result["partition"]
        # Deduplicate by partition (relevant for alias fallback where
        # stale aliases can produce multiple champions per partition)
        if partition in seen_partitions:
            prev = seen_partitions[partition]
            print(f"  WARNING: Duplicate champion for {truncate_partition_name(partition)}, "
                  f"keeping {prev['model_type'].upper()}, skipping {result['model_type'].upper()}")
            continue
        seen_partitions[partition] = result
        models.append(result)
        print(f"  Loaded {truncate_partition_name(result['partition'])} champion "
              f"{result['model_type'].upper()} (v{result['version']})")

    return models


# ---------------------------------------------------------------------------
# Delta-path model loading
# ---------------------------------------------------------------------------

def load_models_from_delta(
    spark,
    scoring_table: str,
    champions_only: bool = False,
    run_id: str = None,
    data_table: str = None,
    validate_data_table: str = None,
    catalog: str = None,
    schema: str = None,
    model_name_prefix: str = None,
) -> tuple:
    """Load model binaries from scoring Delta table.

    Handles three loading patterns used across the pipeline:
    - Eval (03): all models from latest run, with data table validation
    - Inference (04): champion models, most recent per partition across runs
    - Interpretability (05): optional run_id and scope filtering

    Args:
        spark: SparkSession
        scoring_table: Fully qualified scoring table name
        champions_only: If True, only load models with is_champion=True
        run_id: Specific run_id to filter by (None = auto-resolve)
        data_table: Filter rows by DATA_TABLE column value
        validate_data_table: Validate that training data table matches this value
        catalog, schema, model_name_prefix: For constructing model names

    Returns:
        (model_data_list, resolved_run_id) where model_data_list contains dicts
        with: PARTITION_COLUMN, model_type, model, model_config, is_champion,
        mape, nested_run_id, model_name, version
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    scoring_df = spark.table(scoring_table)
    resolved_run_id = run_id

    # Filter by data_table if specified (inference filters to matching data table)
    if data_table:
        scoring_df = scoring_df.filter(F.col("DATA_TABLE") == data_table)

    # Resolve run_id
    if run_id:
        scoring_df = scoring_df.filter(F.col("run_id") == run_id)
        print(f"Filtering to run_id: {run_id}")
    elif not champions_only:
        # Eval pattern: get latest run_id
        latest_row = (
            scoring_df
            .orderBy(F.desc("LOAD_TS"))
            .select("run_id", "DATA_TABLE", "DATA_VERSION")
            .first()
        )
        if not latest_row:
            return [], None

        resolved_run_id = latest_row["run_id"]
        print(f"Using run_id: {resolved_run_id}")

        # Validate data table if requested (eval checks training vs eval table)
        if validate_data_table:
            training_data_table = latest_row["DATA_TABLE"]
            if training_data_table and training_data_table != validate_data_table:
                raise ValueError(
                    f"Data table mismatch: training={training_data_table}, "
                    f"eval={validate_data_table}"
                )
            training_data_version = latest_row["DATA_VERSION"]
            if training_data_version:
                print(f"Training used data_version: {training_data_version}")

        scoring_df = scoring_df.filter(F.col("run_id") == resolved_run_id)

    # Filter to champions
    if champions_only:
        scoring_df = scoring_df.filter(F.col("is_champion") == True)
        print("Filtering to champion models")

        # If no specific run_id, get most recent champion per partition
        if not run_id:
            window = Window.partitionBy("PARTITION_COLUMN").orderBy(F.desc("LOAD_TS"))
            scoring_df = (
                scoring_df
                .withColumn("_rank", F.row_number().over(window))
                .filter(F.col("_rank") == 1)
                .drop("_rank")
            )

    scoring_pdf = scoring_df.toPandas()

    if scoring_pdf.empty:
        return [], resolved_run_id

    # Log run info for multi-run champion loading
    if champions_only and not run_id:
        run_ids = scoring_pdf["run_id"].unique()
        print(f"Found {len(scoring_pdf)} champion(s) from {len(run_ids)} run(s)")

        # Resolve to the most recent training run so downstream notebooks
        # (04_inference, 05_interpretability) can resume the parent MLflow run
        if resolved_run_id is None and "LOAD_TS" in scoring_pdf.columns:
            latest_idx = scoring_pdf["LOAD_TS"].idxmax()
            resolved_run_id = scoring_pdf.loc[latest_idx, "run_id"]
            print(f"Resolved training run: {resolved_run_id}")

    model_data = []
    for _, row in scoring_pdf.iterrows():
        partition = row["PARTITION_COLUMN"]
        model_type = row["MODEL_USED"].lower()

        if row.get("model_binary") is None:
            print(f"  Skipping {partition}/{model_type}: model_binary is None")
            continue

        try:
            model = cloudpickle.loads(row["model_binary"])
            model_config = safe_json_loads(row.get("model_config"))
        except Exception as e:
            print(f"  Error loading {partition}/{model_type}: {e}")
            continue

        model_name = None
        if catalog and schema and model_name_prefix:
            partition_clean = sanitize_partition_name(partition)
            model_name = build_registered_model_name(catalog, schema, model_name_prefix, partition_clean, model_type)

        model_data.append({
            "PARTITION_COLUMN": partition,
            "model_type": model_type,
            "model": model,
            "model_config": model_config,
            "is_champion": bool(row.get("is_champion", False)),
            "mape": row.get("mape"),
            "nested_run_id": row.get("nested_run_id"),
            "model_name": model_name,
            "version": row.get("run_id", "delta"),
        })

    print(f"Loaded {len(model_data)} model(s) from Delta")
    return model_data, resolved_run_id


# ---------------------------------------------------------------------------
# Parallel evaluation helpers
# ---------------------------------------------------------------------------

def set_champion_aliases_parallel(
    client: MlflowClient,
    champion_rows: list,
    max_workers: int = _REGISTRY_MAX_WORKERS,
) -> None:
    """Set @champion aliases in parallel.

    Args:
        client: MlflowClient instance (stateless HTTP, thread-safe)
        champion_rows: List of dicts with model_name, version,
            PARTITION_COLUMN, MODEL_USED
    """
    def _set_one(row):
        if not row.get('version'):
            return  # Not registered — nothing to alias
        client.set_registered_model_alias(
            name=row['model_name'], alias="champion",
            version=row['version']
        )

    parallel_results = _run_parallel(
        _set_one, champion_rows,
        max_workers=max_workers, desc="champion alias"
    )

    for row, result, error in parallel_results:
        partition_short = truncate_partition_name(row['PARTITION_COLUMN'])
        if error:
            print(f"  [{partition_short}] Error setting alias: {error}")
        else:
            print(f"  [{partition_short}] Set @champion on {row['MODEL_USED']} v{row['version']}")


def clear_stale_champion_aliases(
    client: MlflowClient,
    non_champion_model_names: list,
    max_workers: int = _REGISTRY_MAX_WORKERS,
) -> None:
    """Remove @champion alias from non-champion registered models.

    Prevents stale aliases from previous eval runs from interfering
    with champion loading in the inference notebook. Called before
    set_champion_aliases_parallel to ensure only the current champion
    per partition has the @champion alias.
    """
    if not non_champion_model_names:
        return

    def _clear(name):
        try:
            client.delete_registered_model_alias(name, "champion")
        except Exception:
            pass

    parallel_results = _run_parallel(
        _clear, non_champion_model_names,
        max_workers=max_workers, desc="clear stale aliases",
    )
    cleared = sum(1 for _, _, error in parallel_results if error is None)
    print(f"Cleared @champion alias from {cleared} non-champion model(s)")


def update_nested_runs_parallel(
    valid_metrics_df: pd.DataFrame,
    client: MlflowClient,
    experiment_id: str,
    horizon_start: int,
    horizon_end: int,
    log_charts: bool = True,
    chart_func: Callable = None,
    backtest_windows: int = 1,
    max_workers: int = _DEFAULT_MAX_WORKERS,
) -> int:
    """Write eval metrics back to nested MLflow runs in parallel.

    Each thread opens an existing run by run_id (safe: each targets a
    distinct run) and logs metrics + tags. When backtest_windows > 1,
    also logs backtest aggregation metrics (BT_CV, BT_STD_HORIZON_MAPE,
    BT_SELECTION_SCORE, BT_N_WINDOWS).
    """
    import matplotlib.pyplot as plt

    counter = {"logged": 0}
    counter_lock = threading.Lock()

    def _update_one(row_dict):
        nrid = row_dict.get("nested_run_id")
        if not nrid or (isinstance(nrid, float) and np.isnan(nrid)):
            return

        mlflow.set_experiment(experiment_id=experiment_id)

        with mlflow.start_run(run_id=nrid):
            # Eval metrics (batched)
            eval_metrics = {}
            if is_valid_metric(row_dict.get("MAPE")):
                eval_metrics["eval_mape"] = row_dict["MAPE"]
            if is_valid_metric(row_dict.get("HORIZON_MAPE")):
                eval_metrics["eval_horizon_mape"] = row_dict["HORIZON_MAPE"]
            for q in range(1, 7):
                val = row_dict.get(f"Q{q}_MAPE")
                if is_valid_metric(val):
                    eval_metrics[f"eval_q{q}_mape"] = val
            if row_dict.get("prediction_count") is not None:
                eval_metrics["eval_prediction_count"] = row_dict["prediction_count"]

            # Backtest aggregation metrics
            if backtest_windows > 1:
                for bt_key, metric_name in [
                    ("BT_CV", "eval_bt_cv"),
                    ("BT_STD_HORIZON_MAPE", "eval_bt_std_horizon_mape"),
                    ("BT_SELECTION_SCORE", "eval_bt_selection_score"),
                    ("BT_N_WINDOWS", "eval_bt_n_windows"),
                ]:
                    val = row_dict.get(bt_key)
                    if is_valid_metric(val):
                        eval_metrics[metric_name] = val

            if eval_metrics:
                mlflow.log_metrics(eval_metrics)

            mlflow.set_tag("is_champion", str(row_dict.get("is_champion", False)))

            # Model version tags (stateless client calls, thread-safe)
            model_name = row_dict.get("model_name")
            version = row_dict.get("version")
            if model_name and version:
                client.set_model_version_tag(model_name, str(version), "is_champion", str(row_dict.get("is_champion", False)))
                if is_valid_metric(row_dict.get("HORIZON_MAPE")):
                    client.set_model_version_tag(model_name, str(version), "horizon_mape", f"{row_dict['HORIZON_MAPE']:.2f}")
                for q in range(horizon_start, horizon_end + 1):
                    val = row_dict.get(f"Q{q}_MAPE")
                    if is_valid_metric(val):
                        client.set_model_version_tag(model_name, str(version), f"q{q}_mape", f"{val:.2f}")

            # Per-model horizon MAPE chart
            if log_charts and chart_func:
                q_mapes = [row_dict.get(f"Q{h}_MAPE") for h in range(horizon_start, horizon_end + 1)]
                if any(is_valid_metric(m) for m in q_mapes):
                    try:
                        fig = chart_func(
                            q_mapes,
                            row_dict["PARTITION_COLUMN"],
                            row_dict["MODEL_USED"],
                            (horizon_start, horizon_end)
                        )
                        mlflow.log_figure(fig, "charts/horizon_ape.png")
                        plt.close(fig)
                    except Exception:
                        pass

        with counter_lock:
            counter["logged"] += 1

    rows = [row.to_dict() for _, row in valid_metrics_df.iterrows()]
    _run_parallel(_update_one, rows, max_workers=max_workers, desc="nested run update")

    return counter["logged"]
