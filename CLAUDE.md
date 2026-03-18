# AMD Capacity Forecasting Pipeline

## Development Environment

These are Databricks notebooks stored as `.py` files with `# COMMAND ----------` cell separators.
They run on Databricks clusters, not locally. `spark`, `dbutils`, and `display()` are provided by
the Databricks runtime. There is no local test harness.

`%run ./01_training_functions` is a Databricks magic command that executes another notebook in the
same process (like a Python import but at notebook scope). `dbutils.widgets` provides parameterized
notebook inputs. Neither works outside Databricks.

To test utility module changes, you can import and call functions from `modeling_dev/` in a scratch
notebook or Python REPL with appropriate mock data.

## Notebook Format and Code Style

Cells are separated by `# COMMAND ----------`. Markdown cells start with `# MAGIC %md`. Magic
commands like `%pip`, `%run`, `%sql` are prefixed with `# MAGIC ` when stored as `.py` files.
Do not remove these markers -- Databricks uses them to reconstruct the notebook UI.

Every notebook starts with `%pip install -r requirements.txt`, then `dbutils.library.restartPython()`,
then imports. Follow this pattern when adding new notebooks.

**Code style rules:**
- Keep code simple and barebones. This code is a template for the team to work off of.
- Docstrings on functions. Use extensive inline comments where logic is unclear.
- Put explanatory text in markdown cells (`# MAGIC %md`), not code comments.
- Do NOT use emojis anywhere.
- Use f-strings for output that should display in the notebook.

## Data Context

Target variable: `BB` (Billing + Backlog = unconstrained demand). Forecast horizon: 1-6 quarters.

Features: billing moving averages (`BILLING_MOVING_AVERAGE_2Q` through `5Q`), current-period demand
signals (`MOF`, `SELLIN`, `SR`, `MOF_PLUS_BUFFER`), and 3-month / 6-month lagged versions of the
same signals. Full column lists are in `variable_config.yaml`.

Data is synthetic (AMD insider trading restrictions). Structure matches production schema.

## Research Conventions

When asked about Databricks or MLflow best practices, ALWAYS look up current docs first:
- Databricks docs (`databricks.com` / `*.databricks.com`)
- MLflow docs
- Real code examples online

Do not write MLflow- or Databricks-specific API code from memory unless extremely confident.
These APIs change frequently and a wrong method signature wastes hours of debugging on the cluster.

## Pipeline Flow

```
02_model_training  -->  03_model_evaluation  -->  04_inference
                                             -->  05_interpretability (parallel with 04)
                                                  04 --> 06_model_monitoring
```

All notebooks share a single MLflow parent run. Each notebook resumes that run (by `run_id`) to add
its own metrics, tags, and artifacts. The parent run is identified by its `scoring_table` tag.

## File Map

### Notebooks

| File | Role |
|---|---|
| `01_training_functions.py` | **Business logic**: all model training handlers. Imported via `%run` into 02. Not run directly. |
| `02_model_training.py` | **Orchestration**: parallel training via `applyInPandas`, MLflow logging, Delta writes. |
| `03_model_evaluation.py` | **Orchestration**: loads trained models, computes per-horizon MAPE, picks champions, sets `@champion` aliases. |
| `04_inference.py` | **Orchestration**: loads champions, generates predictions, runs sanity checks, writes to Delta. |
| `05_interpretability.py` | **Analysis**: SHAP, feature importance, Prophet components. |
| `06_model_monitoring.py` | **Analysis**: drift detection, backtest-vs-actuals, staleness checks. |

### Utility Modules (`modeling_dev/`) -- Business Logic

| File | What it owns |
|---|---|
| `metric_utils.py` | MAPE/APE calculation, horizon metrics, demand signal indicators, `remove_unwanted_rows`, backtest windows. |
| `modeling_utils.py` | Feature selection (`feature_selection()`), regressor adjustment (`adjust_regressors()`), train/val/test splitting, segment config loading. |
| `data_utils.py` | Data loading (`load_latest_version`), data quality checks, prediction sanity checks, synthetic data imputation, pipeline error logging. |
| `mlflow_utils.py` | MLflow experiment setup, parallel model logging/loading, champion alias management, nested run updates. |
| `chart_utils.py` | Matplotlib chart helpers for horizon APE and champion summary visualizations. |

### Config Files (`configs/`)

| File | What it controls |
|---|---|
| `model_config.yaml` | Per-segment model lists, hyperparameter grids, quality gates, monitoring thresholds, model class definitions. This is the main config file. |
| `variable_config.yaml` | Column lists (`SELECT_COLUMNS`, `REGRESSOR_COLUMNS`, `DISPLAY_COLUMNS_BASE`) and rename mappings shared across all notebooks. |

### Deployment

| File | Purpose |
|---|---|
| `databricks.yml` | Databricks Asset Bundle config. Defines the job DAG, parameters, and targets (dev, dev_brand, qa, prod). |
| `requirements.txt` | Python dependencies installed at the top of each notebook. |

## Where to Make Common Changes

- **MAPE/APE calculation**: `modeling_dev/metric_utils.py` -- `calculate_mape()`, `calculate_ape()`, `calculate_horizon_metrics()`
- **Feature selection or regressor handling**: `modeling_dev/modeling_utils.py` -- `feature_selection()`, `adjust_regressors()`, `filter_regressors()`
- **Add a new sklearn-compatible model**: `configs/model_config.yaml` only (add entry under `models:` with `type: sklearn_api`). No code change needed.
- **Add a model with a different interface**: Write a new `_train_xxx()` handler in `01_training_functions.py`, add a dispatcher branch in `train_model()`, add a Predictor wrapper class
- **Add a new regressor**: (1) `variable_config.yaml` -- add to `REGRESSOR_COLUMNS` and `SELECT_COLUMNS`, (2) `model_config.yaml` -- add to `general.USED_REGRESSORS`, (3) verify column exists in source data table
- **Change quality gate thresholds**: `model_config.yaml` under `QUALITY_GATES`
- **Change which models train per segment**: `model_config.yaml` under `segments.{SEGMENT}.models`
- **Change grid search parameters**: `model_config.yaml` under `models.{model}.grid`
- **Change demand signal indicators**: `model_config.yaml` under `DEMAND_SIGNALS`, logic in `metric_utils.py::calculate_indicators()`
- **Change train/val/test split**: `modeling_dev/modeling_utils.py::split_train_val_test()` and `prepare_training_data()` in `data_utils.py`

## Config Precedence (highest wins)

1. **Widget parameter** (set in notebook UI or job config via `databricks.yml`)
2. **Segment-specific override** in `model_config.yaml` (`segments.MOBILE.grid_search.prophet: false`)
3. **Model-specific default** in `model_config.yaml` (`models.prophet.grid_search: true`)
4. **General anchor** in `model_config.yaml` (`general.grid_search.prophet: true`)

The `segments:` section uses YAML anchors (`<<: *general`) to inherit from the `general:` block.
Overrides in a segment block replace (not merge with) the inherited value.

## Fragile Areas (edit carefully)

- **`eval_result_schema` / `training_result_schema` / `prediction_schema`**: Column names must exactly match what the `applyInPandas` functions return. Adding/removing/renaming a column requires updating both the schema definition AND the function that produces the DataFrame.
- **Closure-captured variables** (`model_lookup`, `training_config`): These dicts are referenced inside `applyInPandas` functions and serialized to Spark workers via cloudpickle. Don't put unpicklable objects in them (database connections, Spark sessions, open file handles).
- **`# COMMAND ----------` separators**: Databricks uses these to parse cells. Don't merge cells by removing separators or split cells by adding them in the wrong place.
- **`dbutils.library.restartPython()` placement**: Must come immediately after `%pip install`. If imports appear before the restart, they'll use stale package versions.

## Key Concepts

### Partitioning
Each model trains on one partition. By default, `PARTITION_COLUMN = MARKET_SEGMENT + "_" + SCENARIO`.
For brand-level forecasting, `PRODUCT_SUB_FAMILY` is added as a third column, creating many more partitions.

### Model Storage (Dual Path)
- **MLflow path**: Models logged as PyFunc artifacts in nested runs, registered to Unity Catalog Model Registry. Used for segment-level (fewer models).
- **Delta path**: Model binaries stored as `cloudpickle` bytes in the `scoring_output` Delta table. Used for brand-level (many models where MLflow registry doesn't scale).

Notebooks 03/04/05 accept an `eval_source` / `model_source` widget (`mlflow` or `delta`) to choose which path.

### Champion Selection
Notebook 03 picks the best model per partition based on HORIZON_MAPE (optionally weighted by backtest stability). The chosen model gets `is_champion=True` in the scoring table and a `@champion` alias in the MLflow registry.

### Parallel Execution
Training and inference use `applyInPandas` (Spark workers). MLflow logging, model loading, and champion alias setting use `ThreadPoolExecutor` (driver threads). The `applyInPandas` functions receive data as pandas DataFrames and must return DataFrames matching the declared schema.

### Error Handling
Errors don't crash the pipeline. They're collected as dicts (`stage`, `partition`, `model_type`, `severity`, `message`), written to the `pipeline_errors` Delta table, and surfaced in notebook output.

## Delta Tables

| Table | Written by | Purpose |
|---|---|---|
| `scoring_output` | 02 (training) | Model binaries, training metrics, configs. One row per model per run. |
| `predictions_metrics` | 03 (eval) | Per-model per-horizon MAPE, champion flags, backtest scores. |
| `{data_table}_predictions` | 04 (inference) | Predictions with actuals, APE, demand signals. |
| `{data_table}_monitoring` | 06 (monitoring) | Backtest, drift, staleness, coverage results. |
| `pipeline_errors` | 02, 03, 04, 06 | Errors and alerts from all pipeline stages. |

## Widget Parameters

All notebooks use `dbutils.widgets` for configuration. Set interactively in the notebook UI or
passed as job parameters via `databricks.yml`. Key shared parameters:

- `catalog` / `schema`: Unity Catalog location
- `data_table`: Source horizon data table name
- `scoring_table`: Delta table for model binaries
- `partition_columns`: Comma-separated columns to partition by
- `config_path`: Path to `model_config.yaml`
- `experiment_name`: MLflow experiment path

## Validating Changes Locally

There is no unit test suite and notebooks can't run outside Databricks. What you can do:

- `python3 -m py_compile {file}.py` on any modified `.py` file to catch syntax errors
- `grep` for removed identifiers to confirm no remaining references
- Read callers of any function you change to verify signature compatibility
- For `modeling_dev/` modules: check that all notebooks importing from the module still have correct import lists
