# Databricks notebook source
# MAGIC %md
# MAGIC # GENERATE_PREDICTIONS_ALL_QUARTER (Databricks Version)
# MAGIC
# MAGIC Direct port of the original Snowflake notebook to Databricks.
# MAGIC Structure and logic preserved exactly - only platform-specific code changed.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Imports

# COMMAND ----------

import sys
import os

notebook_path = os.getcwd()
if notebook_path not in sys.path:
    sys.path.insert(0, notebook_path)

# COMMAND ----------

import builtins
import pickle
import json
import pytz
import datetime
import pandas as pd
import numpy as np
import ast
import itertools
import cloudpickle

import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso, HuberRegressor

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import date, timedelta

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

from prophet import Prophet
import xgboost as xgb

# COMMAND ----------

# MAGIC %md
# MAGIC ## Custom Utils

# COMMAND ----------

from modeling_dev.data_utils import load_latest_version
from modeling_dev.modeling_utils import load_segment_config

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Functions

# COMMAND ----------

# MAGIC %run ./01_training_functions

# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets

# COMMAND ----------

dbutils.widgets.text("catalog", "dash_workshop", "Catalog")
dbutils.widgets.text("schema", "default", "Schema")
dbutils.widgets.text("model_name_prefix", "forecasting", "Model Name Prefix")
dbutils.widgets.text("experiment_name", "/Shared/amd-forecasting", "Experiment Name")
dbutils.widgets.text("data_table", "synthetic_horizon_data_aib", "Data Table")
dbutils.widgets.text("config_path", "configs/model_config.yaml", "Config Path")
dbutils.widgets.text("scoring_table", "scoring_output", "Scoring Table")
dbutils.widgets.text("predictions_table", "predictions_df", "Predictions Table")
dbutils.widgets.text("params_table", "params_df", "Params Table")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME_PREFIX = dbutils.widgets.get("model_name_prefix")
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name")
DATA_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('data_table')}"
CONFIG_PATH = dbutils.widgets.get("config_path")
SCORING_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('scoring_table')}"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('predictions_table')}"
PARAMS_TABLE = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('params_table')}"

print(f"Data table: {DATA_TABLE}")
print(f"Scoring table: {SCORING_TABLE}")
print(f"Predictions table: {PREDICTIONS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC Run One Model

# COMMAND ----------

def get_next_version(client, model_name) -> str:
    """
    Returns the next version of a model based on the existing versions in the registry.
    """
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return "V_1"
        max_version_number = builtins.max([int(v.version) for v in versions])
        return f"V_{max_version_number + 1}"
    except Exception:
        return "V_1"

# COMMAND ----------

client = MlflowClient()
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

def set_market_segments_model_params():
    config_table = {}
    general_config = {}
    config_table['SERVER/WORKSTATION'] = {}
    config_table['MOBILE'] = {}
    config_table['DESKTOP'] = {}
    config_table['GPUDATACENTER'] = {}
    config_table['AIB'] = {}
    config_table['WORK'] = {}
    config_table['IDES'] = {}
    config_table['MISC'] = {}
    config_table['DCGPUCGVDI'] = {}
    config_table['EDG'] = {}
    config_table['EMBEDDED'] = {}
    config_table['DTOEMGPU'] = {}
    config_table['GFXCOMPUTE'] = {}
    config_table['MOBGPU'] = {}
    config_table['GAMING'] = {}


    general_config['PREFERRED_MODEL'] = 'PROPHET'
    general_config['USED_REGRESSORS'] = {'SELLIN_MINUS3_MONTH': 1, 'MOF_PLUS_BUFFER_MINUS3_MONTH': 0.1, 'SR_MINUS3_MONTH': 1, 'BILLING_MOVING_AVERAGE_2Q': 1, 'BILLING_MOVING_AVERAGE_3Q': 1, 'BILLING_MOVING_AVERAGE_4Q': 1, 'BILLING_MOVING_AVERAGE_5Q': 1}
    general_config['NOT_USED_REGRESSORS'] = {}
    general_config['TRAIN_PARAMS'] = {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'}


    config_table['MOBILE']['PREFERRED_MODEL'] = 'PROPHET'
    config_table['MOBILE']['USED_REGRESSORS'] = {'SR_MINUS3_MONTH': 1, 'BILLING_MOVING_AVERAGE_2Q': 1, 'BILLING_MOVING_AVERAGE_3Q': 1, 'BILLING_MOVING_AVERAGE_4Q': 1, 'BILLING_MOVING_AVERAGE_5Q': 1, 'MOF_PLUS_BUFFER_MINUS3_MONTH': 0.1, 'SELLIN_MINUS3_MONTH': 1}
    config_table['MOBILE']['NOT_USED_REGRESSORS'] = {}
    config_table['MOBILE']['TRAIN_PARAMS'] = {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'}


    config_table['SERVER/WORKSTATION']['PREFERRED_MODEL'] = 'PROPHET'
    config_table['SERVER/WORKSTATION']['USED_REGRESSORS'] = {'SELLIN_MINUS3_MONTH': 1, 'MOF_PLUS_BUFFER_MINUS3_MONTH': 0.1, 'SR_MINUS3_MONTH': 1, 'BILLING_MOVING_AVERAGE_2Q': 1, 'BILLING_MOVING_AVERAGE_3Q': 1, 'BILLING_MOVING_AVERAGE_4Q': 1, 'BILLING_MOVING_AVERAGE_5Q': 1}
    config_table['SERVER/WORKSTATION']['NOT_USED_REGRESSORS'] = {}
    config_table['SERVER/WORKSTATION']['TRAIN_PARAMS'] = {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'}


    config_table['DESKTOP']['PREFERRED_MODEL'] = 'PROPHET'
    config_table['DESKTOP']['USED_REGRESSORS'] = {'SELLIN_MINUS3_MONTH': 1, 'MOF_PLUS_BUFFER_MINUS3_MONTH': 0.1, 'SR_MINUS3_MONTH': 1, 'BILLING_MOVING_AVERAGE_2Q': 1, 'BILLING_MOVING_AVERAGE_3Q': 1, 'BILLING_MOVING_AVERAGE_4Q': 1, 'BILLING_MOVING_AVERAGE_5Q': 1}
    config_table['DESKTOP']['NOT_USED_REGRESSORS'] = {}
    config_table['DESKTOP']['TRAIN_PARAMS'] = {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01, 'seasonality_mode': 'additive'}

    config_table['IDES'] = general_config
    config_table['MISC'] = general_config
    config_table['DCGPUCGVDI'] = general_config
    config_table['EDG'] = general_config
    config_table['EMBEDDED'] = general_config
    config_table['DTOEMGPU'] = general_config
    config_table['GFXCOMPUTE'] = general_config
    config_table['MOBGPU'] = general_config
    config_table['WORK'] = general_config
    config_table['AIB'] = general_config
    config_table['GPUDATACENTER'] = general_config
    config_table['GAMING'] = general_config

    # Add common test/synthetic segments
    config_table['ABC'] = general_config
    config_table['SEGMENT_A'] = general_config
    config_table['SEGMENT_B'] = general_config
    config_table['SEGMENT_E'] = general_config
    config_table['SEGMENT_F'] = general_config
    config_table['SEGMENT_X'] = general_config

    return config_table

#Save it as a Dataframe
market_segment_config_table = set_market_segments_model_params()
market_segment_config_table = pd.DataFrame(market_segment_config_table)
market_segment_config_table = market_segment_config_table.T
market_segment_config_table = market_segment_config_table.reset_index()
market_segment_config_table.rename(columns={"index": "MARKET_SEGMENT"}, inplace=True)


def clean_dict(d):
    return json.dumps(d, indent=None, separators=(",", ":"), ensure_ascii=False)

# Apply this to all dict-containing columns
for col in market_segment_config_table.columns:
    if market_segment_config_table[col].apply(lambda x: isinstance(x, dict)).any():
        market_segment_config_table[col] = market_segment_config_table[col].apply(clean_dict)

#Save to Delta
market_segment_config_spark = spark.createDataFrame(market_segment_config_table)
market_segment_config_spark.write.mode("overwrite").saveAsTable(PARAMS_TABLE)

#look up from Delta
param_lookup = spark.table(PARAMS_TABLE).toPandas().set_index("MARKET_SEGMENT")

# Save as a pickle file for inclusion in the model artifact
param_lookup.to_pickle("/tmp/param_lookup.pkl")

display(param_lookup)

# COMMAND ----------

# MAGIC %md
# MAGIC # Utilities for calculation

# COMMAND ----------

def categorize_percentage_difference(ratio):
    if ratio < -0.2:
        return 'VERY LOW'
    elif -0.2 <= ratio <= -0.1:
        return 'LOW'
    elif -0.1 <= ratio < 0.1:
        return 'FLAT'
    elif 0.1 <= ratio < 0.2:
        return 'HIGH'
    elif 0.2 <= ratio:
        return 'VERY HIGH'

def calculate_indicator(billingCol, MOFCol):
    ratios = (billingCol - MOFCol)/MOFCol
    vectorized_function = np.vectorize(categorize_percentage_difference)
    indicators = vectorized_function(ratios)
    return indicators

def calculate_mape(actual, predicted):
    # Guard against division by zero (actual=0 produces Infinity)
    mask = actual.notnull() & predicted.notnull() & (actual != 0)
    return (abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

def make_verification_columns(df, affix):
    mapping = {'VERY LOW': 1, 'LOW': 2, 'FLAT': 3, 'HIGH': 4, 'VERY HIGH': 5}
    df['INDICATOR_num'] = df['INDICATOR' + affix].map(mapping)
    df['INDICATOR_FORECASTED_num'] = df['INDICATOR_FORECASTED' + affix].map(mapping)

    df['VALIDATION_DISTANCE' + affix] = df.apply(
        lambda row: float(abs(row['INDICATOR_num'] - row['INDICATOR_FORECASTED_num'])) if pd.notnull(row['INDICATOR' + affix]) and pd.notnull(row['INDICATOR_FORECASTED' + affix]) else np.nan,
        axis=1
    )

    df['DIRECTIONAL_CORRECTNESS' + affix] = df.apply(
        lambda row: float((
            (row['INDICATOR_num'] > 3 and row['INDICATOR_FORECASTED_num'] > 3)
        ) or (
            (row['INDICATOR_num'] <= 2 and row['INDICATOR_FORECASTED_num'] <= 2)
        ) or (
            (row['INDICATOR_num'] == 3 and row['INDICATOR_FORECASTED_num'] == 3)
        )) if pd.notnull(row['INDICATOR' + affix]) and pd.notnull(row['INDICATOR_FORECASTED' + affix]) else np.nan,
        axis=1
    )

    df = df.drop(['INDICATOR_num', 'INDICATOR_FORECASTED_num'], axis=1)
    return df

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

def remove_unwanted_rows(df):
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

    df['QUARTER_MONTH']= df['QUARTER']+'M3'

    def extract_quarter(value):
        if value is None or pd.isna(value):
            return None
        return str(value).split('M')[0]

    df["MOF_VERSION_QUARTER"] = df['MOF_VERSION'].apply(extract_quarter)
    df["REMOVE"] = (df['MOF_VERSION_QUARTER'] == df['QUARTER']) & df['MOF_VERSION_QUARTER'].notna()

    def next_Q5(value):
        if value is None or pd.isna(value):
            return None
        year_quarter = str(value).split('M')[0]
        year = int(year_quarter.split('Q')[0])
        quarter = int(str(value).split('Q')[1].split('M')[0])

        if quarter == 4:
            new_year = year + 2
            new_quarter = 1
        else:
            new_year = year + 1
            new_quarter = quarter + 1
        return str(new_year) + 'Q' + str(new_quarter)

    df["Q+5"] = df['MOF_VERSION_QUARTER'].apply(next_Q5)

    def next_Q6(value):
        if value is None or pd.isna(value):
            return None
        year_quarter = str(value).split('M')[0]
        year = int(year_quarter.split('Q')[0])
        quarter = int(str(value).split('Q')[1].split('M')[0])

        if quarter in [3, 4]:
            new_year = year + 2
            if quarter == 3:
                new_quarter = 1
            else:
                new_quarter = 2
        else:
            new_year = year + 1
            new_quarter = quarter + 2
        return str(new_year) + 'Q' + str(new_quarter)

    df["Q+6"] = df['MOF_VERSION_QUARTER'].apply(next_Q6)
    # Only compare when values are not null
    df["Q+5_INDICATOR"] = (df['QUARTER'] == df['Q+5']) & df['Q+5'].notna()
    df["Q+6_INDICATOR"] = (df['QUARTER'] == df['Q+6']) & df['Q+6'].notna()

    df = df[~(df[['REMOVE','Q+5_INDICATOR','Q+6_INDICATOR']].fillna(False).any(axis=1))]

    cols_to_remove = ['QUARTER_MONTH','MOF_VERSION_QUARTER','REMOVE','Q+5', 'Q+6','Q+5_INDICATOR','Q+6_INDICATOR']
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')

    return df


# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

SELECT_COLUMNS = [
    "QUARTER", "QUARTER_PARSED", "MARKET_SEGMENT", "SCENARIO",
    "VERIFICATION_HORIZON", "PREDICTION_HORIZON", "BB",
    "BILLING_MOVING_AVERAGE_2Q", "BILLING_MOVING_AVERAGE_3Q",
    "BILLING_MOVING_AVERAGE_4Q", "BILLING_MOVING_AVERAGE_5Q",
    "BB_VERSION", "MOF_VERSION", "BB_VERSION_LATEST", "MOF_VERSION_LATEST",
    "MOF_0_MONTH", "MOF_PLUS_BUFFER_0_MONTH", "SELLIN_0_MONTH", "SR_0_MONTH",
    "SELLIN_MINUS3_MONTH", "MOF_MINUS3_MONTH", "SR_MINUS3_MONTH", "MOF_PLUS_BUFFER_MINUS3_MONTH"
]

df_raw, DATA_VERSION = load_latest_version(spark, DATA_TABLE, SELECT_COLUMNS)
print(f"Using data version: {DATA_VERSION}")

# Add PARTITION_COLUMN
partition_columns = ["MARKET_SEGMENT", "SCENARIO"]
df_raw = df_raw.withColumn("PARTITION_COLUMN", F.concat_ws("_", *[F.col(c) for c in partition_columns]))

segments = [row["MARKET_SEGMENT"] for row in df_raw.select("MARKET_SEGMENT").distinct().collect()]
print(f"Found {len(segments)} segments: {segments}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Testing before Model Training and Registration
# MAGIC
# MAGIC Test training on a single segment before full parallel execution.

# COMMAND ----------

# Filter to single segment for testing
df_test = df_raw.filter(F.col("MARKET_SEGMENT") == segments[0])
df = df_test.toPandas()

print(f"Testing with segment: {segments[0]}, rows: {len(df)}")

# COMMAND ----------

import yaml
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Test train_prophet on single segment
segment = df['MARKET_SEGMENT'].iloc[0]
seg_config = load_segment_config(config, segment)

print(f"Testing Prophet training for {segment}...")
model, preds, metrics, params = train_prophet(df.copy(), segment, seg_config, config)
print(f"Prophet MAPE: {metrics.get('mape', 'N/A')}")

print(f"\nTesting XGBoost training for {segment}...")
model_xgb, preds_xgb, metrics_xgb, params_xgb = train_xgboost(df.copy(), segment, seg_config, config)
print(f"XGBoost MAPE: {metrics_xgb.get('mape', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Training and Registration
# MAGIC
# MAGIC The monster class - direct port from Snowflake custom_model to MLflow PyFunc.

# COMMAND ----------

class FORECASTINGMODEL_XGBOOST_PROPHET(PythonModel):
    """
    Direct port of AMD's FORECASTINGMODEL_XGBOOST_PROPHET class.
    Uses MLflow PyFunc instead of Snowflake custom_model.
    """

    def load_context(self, context):
        """Load parameter table from artifact (equivalent to __init__ in Snowflake)"""
        with open(context.artifacts["param_lookup"], "rb") as f:
            self.market_segment_config_table = pickle.load(f)

    def predict(self, context, model_input):
        """
        Main prediction method - exact port of the Snowflake predict() method.
        """
        df = model_input

        from prophet import Prophet
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import HuberRegressor

        def feature_selection(allData, market_segment, validation_horizon, prediction_horizon, common_columns, regressors, target_column):
            train_data = allData.iloc[:-(validation_horizon + prediction_horizon)]
            train_data = train_data[common_columns + regressors + target_column]
            train_data = train_data.dropna(subset=target_column)
            train_data[regressors] = train_data[regressors].fillna(0)

            X_feature_selection = train_data[regressors]
            Y_feature_selection = train_data[target_column[0]]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_feature_selection)
            y_scaled = scaler.fit_transform(np.array(Y_feature_selection).reshape(-1, 1)).ravel()

            if market_segment in ['MOBILE', 'AIB', 'EMBEDDED']:
                n_features = 6
            else:
                n_features = 5
            selector = RFE(estimator=HuberRegressor(), n_features_to_select=n_features)
            selector.fit_transform(X_scaled, y_scaled)

            return list(X_feature_selection.columns[selector.get_support()])

        def adjust_regressors(data, regressors, main_regressor, Market_Segment, threshold=0.1):
            for reg in regressors:
                if Market_Segment in ['GAMING']:
                    data.loc[(data[reg] < threshold * data[main_regressor]) & (data[reg] != 0), reg] = data[main_regressor]
                else:
                    data.loc[(data[reg] < threshold * data[main_regressor]), reg] = data[main_regressor]
            return data

        try:
            regressors_weight_use = False
            df = df.reset_index(drop=True)
            MARKET_SEGMENT = df['MARKET_SEGMENT'].iloc[0]
            scenario = df['SCENARIO'].iloc[0] if 'SCENARIO' in df.columns else 'DEFAULT'

            # Check if segment exists in config, use GAMING (general) as fallback
            if MARKET_SEGMENT not in self.market_segment_config_table.index:
                print(f"Warning: {MARKET_SEGMENT} not in config, using GAMING defaults")
                MARKET_SEGMENT_CONFIG = 'GAMING'
            else:
                MARKET_SEGMENT_CONFIG = MARKET_SEGMENT

            PREFERRED_MODEL = self.market_segment_config_table['PREFERRED_MODEL'][MARKET_SEGMENT_CONFIG]

            if PREFERRED_MODEL == 'PROPHET':
                regressors_weights = json.loads(self.market_segment_config_table['USED_REGRESSORS'][MARKET_SEGMENT_CONFIG])
                used_regressors = json.loads(self.market_segment_config_table['USED_REGRESSORS'][MARKET_SEGMENT_CONFIG])
                not_used_regressors = json.loads(self.market_segment_config_table['NOT_USED_REGRESSORS'][MARKET_SEGMENT_CONFIG])
                all_regressors = list(used_regressors.keys()) + list(not_used_regressors.keys())

                train_params = json.loads(self.market_segment_config_table['TRAIN_PARAMS'][MARKET_SEGMENT_CONFIG])

                allData = df.rename(columns={'QUARTER_PARSED': 'ds', 'BB': 'y'})
                allData = allData.sort_values(by='ds')
                allData = allData.drop(columns=['PARTITION_COLUMN'], errors='ignore')
                allData = allData.drop_duplicates(subset=['ds'], keep='first')

                validation_horizon = int(allData['VERIFICATION_HORIZON'].median())
                prediction_horizon = int(allData['PREDICTION_HORIZON'].median())

                common_columns = ['MARKET_SEGMENT', 'QUARTER']
                target_column = ['y']  # Use actual target, not date column

                allData = adjust_regressors(allData, all_regressors, 'BILLING_MOVING_AVERAGE_2Q', Market_Segment=MARKET_SEGMENT)

                if MARKET_SEGMENT in ['MOBILE', 'AIB', 'EMBEDDED']:
                    remove_cols = ["SELLIN_MINUS3_MONTH"]
                else:
                    remove_cols = []

                all_regressors_edited = [col for col in all_regressors if col not in remove_cols]

                top_regressors = feature_selection(allData, MARKET_SEGMENT, validation_horizon, prediction_horizon, common_columns, all_regressors_edited, target_column)
                remaining_regressors = [reg for reg in all_regressors if reg not in top_regressors]
                used_regressors = {key: regressors_weights[key] for key in top_regressors if key in regressors_weights}
                not_used_regressors = {key: regressors_weights[key] for key in remaining_regressors if key in regressors_weights}

                prophet_data = allData.iloc[:-(validation_horizon + prediction_horizon)].copy()
                prophet_data = prophet_data.dropna(subset=['y'])

                for reg in used_regressors.keys():
                    prophet_data.loc[prophet_data[reg] == 0, reg] = prophet_data['BILLING_MOVING_AVERAGE_2Q']

                cap = prophet_data['y'].max() * 10
                prophet_data['floor'] = 0
                prophet_data['cap'] = cap

                model = Prophet(
                    changepoint_prior_scale=train_params['changepoint_prior_scale'],
                    seasonality_prior_scale=train_params['seasonality_prior_scale'],
                    seasonality_mode=train_params['seasonality_mode'],
                    growth='logistic',
                    uncertainty_samples=0
                )
                used_params = {**train_params, **used_regressors}

                if len(used_regressors.keys()) > 0:
                    for regressor in used_regressors.keys():
                        if regressors_weight_use:
                            model.add_regressor(regressor, prior_scale=used_regressors[regressor])
                        else:
                            model.add_regressor(regressor)

                model.add_seasonality(name='quarterly', period=91.3125, fourier_order=5)
                model.fit(prophet_data[['ds', 'y', 'floor', 'cap'] + list(used_regressors.keys())])

                future = model.make_future_dataframe(periods=validation_horizon + prediction_horizon, freq='Q')
                future = pd.merge(future, allData[['ds'] + list(used_regressors.keys())], on='ds', how='left')
                future[list(used_regressors.keys())] = future[list(used_regressors.keys())].ffill().bfill().fillna(0)
                future['floor'] = 0
                future['cap'] = cap

                forecast = model.predict(future)

                # Normalize dates to avoid timezone/precision mismatches in merge
                allData['ds'] = pd.to_datetime(allData['ds']).dt.normalize()
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.normalize()

                demand_signals = ['MOF_0_MONTH', 'MOF_PLUS_BUFFER_0_MONTH', 'SELLIN_0_MONTH', 'SR_0_MONTH']
                for demand_signal in demand_signals:
                    if demand_signal not in allData.columns:
                        allData[demand_signal] = 0

                fullResult = pd.merge(allData, forecast, on='ds', how='right')

                displayCols = ['MARKET_SEGMENT', 'QUARTER', 'ds', 'y', 'yhat', 'VERIFICATION_HORIZON', 'PREDICTION_HORIZON',
                              'BB_VERSION', 'MOF_VERSION', 'BB_VERSION_LATEST', 'MOF_VERSION_LATEST',
                              'MOF_0_MONTH', 'MOF_PLUS_BUFFER_0_MONTH', 'SELLIN_0_MONTH', 'SR_0_MONTH'] + \
                             list(not_used_regressors.keys()) + [reg + '_x' for reg in list(used_regressors.keys()) if reg + '_x' in fullResult.columns]

                # Only select columns that exist
                displayCols = [c for c in displayCols if c in fullResult.columns]
                predictions = fullResult[displayCols].copy()

                predictions = predictions.rename(columns={
                    'ds': 'QUARTER_PARSED', 'y': 'BILLING', 'yhat': 'BILLING_PREDICTED',
                    'MOF_0_MONTH': 'MOF', 'MOF_PLUS_BUFFER_0_MONTH': 'MOF_PLUS_BUFFER',
                    'SELLIN_0_MONTH': 'SELLIN', 'SR_0_MONTH': 'SR'
                })

                for reg in [reg + '_x' for reg in list(used_regressors.keys())]:
                    if reg in predictions.columns:
                        predictions = predictions.rename(columns={reg: reg.split('_x')[0]})

                predictions['used_params'] = str(used_params)
                predictions['MODEL_USED'] = PREFERRED_MODEL
                predictions['SCENARIO'] = scenario

            elif PREFERRED_MODEL == 'XGBOOST':
                used_regressors = json.loads(self.market_segment_config_table['USED_REGRESSORS'][MARKET_SEGMENT_CONFIG])
                not_used_regressors = json.loads(self.market_segment_config_table['NOT_USED_REGRESSORS'][MARKET_SEGMENT_CONFIG])
                all_regressors = list(used_regressors.keys()) + list(not_used_regressors.keys())

                allData = df.copy()
                allData = allData.sort_values(by='QUARTER_PARSED')
                allData = allData.drop(columns=['PARTITION_COLUMN'], errors='ignore')
                allData = allData.drop_duplicates(subset=['QUARTER_PARSED'], keep='first')

                common_columns = ['MARKET_SEGMENT', 'QUARTER', 'QUARTER_PARSED']
                target_column = ['BB']

                validation_horizon = int(allData['VERIFICATION_HORIZON'].median())
                prediction_horizon = int(allData['PREDICTION_HORIZON'].median())

                allData = adjust_regressors(allData, all_regressors, 'BILLING_MOVING_AVERAGE_2Q', Market_Segment=MARKET_SEGMENT)

                if MARKET_SEGMENT in ['AIB', 'EMBEDDED']:
                    remove_cols = ["SELLIN_MINUS3_MONTH"]
                else:
                    remove_cols = []

                all_regressors_edited = [col for col in all_regressors if col not in remove_cols]
                regressors = feature_selection(allData, MARKET_SEGMENT, validation_horizon, prediction_horizon, common_columns, all_regressors_edited, target_column)

                train_data = allData.iloc[:-(validation_horizon + prediction_horizon)].copy()
                train_data = train_data[common_columns + regressors + target_column]
                train_data = train_data.dropna(subset=target_column)
                validate_data = allData.iloc[-(validation_horizon + prediction_horizon):-prediction_horizon].copy()
                validate_data = validate_data[common_columns + regressors + target_column]
                validate_data = validate_data.dropna(subset=target_column)
                test_data = allData.iloc[-prediction_horizon:].copy()
                test_data = test_data[common_columns + regressors + target_column]

                if validate_data.empty and len(train_data) > 1:
                    pseudo_val = train_data.tail(1).copy()
                    train_data = train_data.iloc[:-1].copy()
                    validate_data = pd.concat([validate_data, pseudo_val], ignore_index=True)

                X = train_data[regressors]
                y = train_data[target_column[0]]
                Xv = validate_data[regressors]
                yv = validate_data[target_column[0]]

                has_val = len(Xv) > 0

                grid = {
                    "n_estimators": [20, 40, 60],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [6],
                    "min_child_weight": [1, 3, 5],
                    "reg_lambda": [0],
                    "reg_alpha": [0],
                    "subsample": [0.5],
                    "colsample_bytree": [0.8],
                    "tree_method": ["hist"],
                    "objective": ["reg:squarederror"],
                    "random_state": [42],
                    "eval_metric": ["rmse"],
                }

                def param_combos(g):
                    keys = list(g.keys())
                    for values in itertools.product(*[g[k] for k in keys]):
                        yield dict(zip(keys, values))

                best_params = None
                best_model = None
                best_val_mae = np.inf

                for params in param_combos(grid):
                    params_with_rounds = params.copy()
                    if has_val:
                        params_with_rounds["early_stopping_rounds"] = 5

                    xgb_model = xgb.XGBRegressor(**params_with_rounds)

                    if has_val:
                        xgb_model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
                        yv_pred = xgb_model.predict(Xv)
                        val_mae = mean_absolute_error(yv, yv_pred)

                        if val_mae < best_val_mae:
                            best_val_mae = val_mae
                            best_params = params_with_rounds
                            best_model = xgb_model
                    else:
                        xgb_model.fit(X, y, verbose=False)
                        if best_model is None:
                            best_model = xgb_model
                            best_params = params_with_rounds
                            best_val_mae = float("nan")

                X_test = test_data[regressors]
                train_data['BILLING_PREDICTED'] = np.nan
                validate_data['BILLING_PREDICTED'] = best_model.predict(Xv)
                test_data['BILLING_PREDICTED'] = best_model.predict(X_test)

                forecast = pd.concat([train_data, validate_data, test_data])

                demand_signals = ['MOF_0_MONTH', 'MOF_PLUS_BUFFER_0_MONTH', 'SELLIN_0_MONTH', 'SR_0_MONTH']
                for demand_signal in demand_signals:
                    if demand_signal not in allData.columns:
                        allData[demand_signal] = 0

                fullResult = pd.merge(allData, forecast, on=['MARKET_SEGMENT', 'QUARTER', 'QUARTER_PARSED', 'BB'] + regressors, how='right')

                displayCols = ['MARKET_SEGMENT', 'QUARTER', 'QUARTER_PARSED', 'BB', 'BILLING_PREDICTED',
                              'VERIFICATION_HORIZON', 'PREDICTION_HORIZON', 'BB_VERSION', 'MOF_VERSION',
                              'BB_VERSION_LATEST', 'MOF_VERSION_LATEST',
                              'MOF_0_MONTH', 'MOF_PLUS_BUFFER_0_MONTH', 'SELLIN_0_MONTH', 'SR_0_MONTH'] + all_regressors
                displayCols = [c for c in displayCols if c in fullResult.columns]

                predictions = fullResult[displayCols].copy()
                predictions = predictions.rename(columns={
                    'BB': 'BILLING', 'MOF_0_MONTH': 'MOF', 'MOF_PLUS_BUFFER_0_MONTH': 'MOF_PLUS_BUFFER',
                    'SELLIN_0_MONTH': 'SELLIN', 'SR_0_MONTH': 'SR'
                })

                used_params = best_model.get_xgb_params()
                used_params['regressors'] = regressors
                predictions['used_params'] = json.dumps(used_params)
                predictions['MODEL_USED'] = PREFERRED_MODEL
                predictions['SCENARIO'] = scenario

            # Add _OUTPUT suffix (AMD convention)
            predictions.columns = [col + "_OUTPUT" for col in predictions.columns]
            return predictions

        except Exception as e:
            import traceback
            segment_name = df['MARKET_SEGMENT'].iloc[0] if len(df) > 0 else 'unknown'
            print(f"Error in predict for {segment_name}: {e}")
            traceback.print_exc()
            return pd.DataFrame()


# Test the model locally
my_forecasting_model = FORECASTINGMODEL_XGBOOST_PROPHET()
my_forecasting_model.market_segment_config_table = param_lookup

# COMMAND ----------

display(df_test)

# COMMAND ----------

# Check segment alignment
test_segment = df_test.select("MARKET_SEGMENT").first()[0]
print(f"Test data segment: '{test_segment}'")
print(f"Available segments in param_lookup: {list(param_lookup.index)}")
print(f"Segment in param_lookup? {test_segment in param_lookup.index}")

# COMMAND ----------

# Test predict locally - with error surfacing
df_input = df_test.toPandas()
try:
    ans = my_forecasting_model.predict(context=None, model_input=df_input)
    if ans is None or ans.empty:
        print("predict() returned empty DataFrame - check segment match above")
    else:
        display(ans)
except Exception as e:
    print(f"Error during predict: {e}")
    import traceback
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC # Register Model to MLflow

# COMMAND ----------

from mlflow.models.signature import infer_signature
signature = infer_signature(df_test.toPandas(), ans)
print(signature)

# COMMAND ----------

model_name = f"{CATALOG}.{SCHEMA}.FORECASTINGMODEL_XGBOOST_PROPHET"

with mlflow.start_run() as run:
    model_version = get_next_version(client, model_name)

    mlflow.pyfunc.log_model(
        name="model",
        python_model=FORECASTINGMODEL_XGBOOST_PROPHET(),
        artifacts={"param_lookup": "/tmp/param_lookup.pkl"},
        pip_requirements=["prophet", "pandas", "scikit-learn", "xgboost", "numpy"],
        signature=signature,
        registered_model_name=model_name,
    )

    run_id = run.info.run_id
    print(f"Logged model with run_id: {run_id}")
    print(f"Model version: {model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Inference and Prediction

# COMMAND ----------

# Output schema for applyInPandas
output_schema = StructType([
    StructField("MARKET_SEGMENT_OUTPUT", StringType(), True),
    StructField("QUARTER_OUTPUT", StringType(), True),
    StructField("QUARTER_PARSED_OUTPUT", TimestampType(), True),
    StructField("BILLING_OUTPUT", DoubleType(), True),
    StructField("BILLING_PREDICTED_OUTPUT", DoubleType(), True),
    StructField("VERIFICATION_HORIZON_OUTPUT", DoubleType(), True),
    StructField("PREDICTION_HORIZON_OUTPUT", DoubleType(), True),
    StructField("BB_VERSION_OUTPUT", StringType(), True),
    StructField("MOF_VERSION_OUTPUT", StringType(), True),
    StructField("BB_VERSION_LATEST_OUTPUT", StringType(), True),
    StructField("MOF_VERSION_LATEST_OUTPUT", StringType(), True),
    StructField("MOF_OUTPUT", DoubleType(), True),
    StructField("MOF_PLUS_BUFFER_OUTPUT", DoubleType(), True),
    StructField("SELLIN_OUTPUT", DoubleType(), True),
    StructField("SR_OUTPUT", DoubleType(), True),
    StructField("SR_MINUS3_MONTH_OUTPUT", DoubleType(), True),
    StructField("SELLIN_MINUS3_MONTH_OUTPUT", DoubleType(), True),
    StructField("MOF_PLUS_BUFFER_MINUS3_MONTH_OUTPUT", DoubleType(), True),
    StructField("BILLING_MOVING_AVERAGE_2Q_OUTPUT", DoubleType(), True),
    StructField("BILLING_MOVING_AVERAGE_3Q_OUTPUT", DoubleType(), True),
    StructField("BILLING_MOVING_AVERAGE_4Q_OUTPUT", DoubleType(), True),
    StructField("BILLING_MOVING_AVERAGE_5Q_OUTPUT", DoubleType(), True),
    StructField("used_params_OUTPUT", StringType(), True),
    StructField("MODEL_USED_OUTPUT", StringType(), True),
    StructField("SCENARIO_OUTPUT", StringType(), True),
])

# Capture param_lookup in closure for workers
param_lookup_for_workers = param_lookup.copy()

def run_partitioned_inference(pdf: pd.DataFrame) -> pd.DataFrame:
    """Run inference for a partition using the model's predict logic."""
    model = FORECASTINGMODEL_XGBOOST_PROPHET()
    model.market_segment_config_table = param_lookup_for_workers

    try:
        result = model.predict(context=None, model_input=pdf)
        if result is None or result.empty:
            return pd.DataFrame()
        return result
    except Exception as e:
        print(f"Error in partition: {e}")
        return pd.DataFrame()

# Run partitioned inference
print("Running parallel inference...")
tbl = df_raw.groupBy("PARTITION_COLUMN").applyInPandas(run_partitioned_inference, schema=output_schema).toPandas()
print(f"Generated {len(tbl)} predictions")

# COMMAND ----------

#Select the output columns only
output_cols = [col for col in tbl.columns if col.endswith('_OUTPUT')]
results = tbl[output_cols].copy()
rename_map = {col: col.replace('_OUTPUT', '') for col in output_cols}
results.rename(columns=rename_map, inplace=True)

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Validation Metrics

# COMMAND ----------

results['APE'] = calculate_mape(results['BILLING'], results['BILLING_PREDICTED'])
results['PARTITION_COLUMN'] = results['MARKET_SEGMENT'].astype(str) + "_" + results['SCENARIO'].astype(str)

# COMMAND ----------

def wrapped_indicator(group_df):
    val_med = group_df['VERIFICATION_HORIZON'].median()
    pred_med = group_df['PREDICTION_HORIZON'].median()
    if pd.isna(val_med) or pd.isna(pred_med):
        return group_df
    val_horizon = int(val_med)
    pred_horizon = int(pred_med)

    return calculate_indicators(
        group_df,
        ['MOF', 'MOF_PLUS_BUFFER', 'SELLIN', 'SR'],
        validation_horizon=val_horizon,
        prediction_horizon=pred_horizon
    )

results_in = results.groupby(["PARTITION_COLUMN", "MODEL_USED"], group_keys=False).apply(wrapped_indicator)

results_in['APE'] = results_in.apply(lambda row: row['APE'] if row.drop('APE').notnull().all() else None, axis=1)

results_in = remove_unwanted_rows(results_in)

display(results_in)

# COMMAND ----------

#Add model version column
results_in["MODEL_VERSION"] = model_version

#Add version for predictions
try:
    df_versions = spark.table(PREDICTIONS_TABLE).toPandas()
    latest_version = df_versions["PREDICTIONS_VERSION"].str.extract(r'V_(\d+)')[0].dropna().astype(int).max()
    new_version = f"V_{int(latest_version) + 1}"
except:
    new_version = "V_0"

results_in["PREDICTIONS_VERSION"] = new_version

# Set LOAD_TS to current CST time
cst = pytz.timezone("US/Central")
results_in["LOAD_TS"] = datetime.datetime.now(cst).replace(tzinfo=None)

# Training data version
results_in["TRAINING_DATA_VERSION"] = DATA_VERSION

display(results_in.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # Write Predictions

# COMMAND ----------

save_predictions = spark.createDataFrame(results_in)
save_predictions.write.mode("append").option("mergeSchema", "true").saveAsTable(PREDICTIONS_TABLE)

print(f"Predictions written to {PREDICTIONS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary

# COMMAND ----------

print(f"Run ID: {run_id}")
print(f"Model version: {model_version}")
print(f"Data version: {DATA_VERSION}")
print(f"Predictions version: {new_version}")
print(f"Total predictions: {len(results_in)}")

summary = results_in.groupby(["MARKET_SEGMENT", "MODEL_USED"]).agg({
    "APE": ["mean", "count"]
}).reset_index()
summary.columns = ["MARKET_SEGMENT", "MODEL_USED", "avg_ape", "prediction_count"]
display(summary)
