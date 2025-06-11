import traceback

import pandas as pd
import logging
import re
import os
from typing import Tuple, Any, List, Optional

from pyspark.sql import SparkSession

from db.queries import get_predictive_data, get_actual_data, ForecastConfig
from db.error_logger import insert_profiling_error
from docstring.utilities import profiled_function
from profiler.errors.decorators import databricks_safe
from profiler.errors.utils import get_error_metadata
from profiler.profiler_switch import profiling_switch
from utils.exit_handler import safe_exit

# Setup logger
logging.basicConfig(level=logging.INFO)


@profiled_function(category="database_call", enabled=profiling_switch.enabled)
def load_and_prepare_data(ufm_config: ForecastConfig, rows=5000, actual=False, save=False, spark=None) -> pd.DataFrame:
    """
        Function: Loads and prepares predictive input dataset either from a CSV file or by fetching it via a function.


        Parameters:
        ----------
        ufm : ForecastConfig
            Configuration that contains information about the model
            It contains the following variables:
                ufmd : int, default=39
                    UFMID to use when fetching dataset using `get_predictive_data`.
                method : str, default="ARIMA"
                    Method name to be used in the filename when saving the dataset.

        Returns:
        -------
        pd.DataFrame
            A prepared DataFrame ready for use in predictive models.
        """
    try:
        environment = os.getenv("ENV", "DEV")
        path = f"dataset/{environment}"
        os.makedirs(path, exist_ok=True)

        csv = f"{path}/{'ActualData' if actual else 'PredictiveInputData'}{ufm_config.forecast_method_name}.csv"

        # 🔍 Step 1: Load from DB if CSV does not exist
        if not os.path.isfile(csv):
            logging.info(f"📥 Fetching data for UFMID={ufm_config.user_forecast_method_id}")
            try:
                df = get_actual_data(rows=rows) if actual else get_predictive_data(ufm_config.user_forecast_method_id)
            except Exception as e:
                meta = get_error_metadata("ConnectionRefused", {"resource": "database"})
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="ConnectionRefused",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                safe_exit(meta["code"], meta["message"])

            if df.empty:
                meta = get_error_metadata("EmptyQueryResult", {"forecast_method_id": ufm_config.forecast_method_id})
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="EmptyQueryResult",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                safe_exit(meta["code"], meta["message"])

            if save:
                df.to_csv(csv, index=False)
                logging.info(f"💾 Data saved to {csv}")
        else:
            logging.info(f"📂 Loading dataset from {csv}")
            try:
                df = pd.read_csv(csv)
                logging.info("✅ Raw dataset loaded.")
            except Exception as e:
                meta = get_error_metadata("PandasLoadError", {"exception": str(e)})
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="PandasLoadError",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                safe_exit(meta["code"], meta["message"])

        # 🔍 Step 2: Schema validation
        required_columns = ["CustomerID", "PodID"]
        for col in required_columns:
            if col not in df.columns:
                meta = get_error_metadata("SchemaMismatch", {"column": col})
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback="",  # or traceback.format_exc()
                    error_type="SchemaMismatch",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                safe_exit(meta["code"], meta["message"])

        # 🔍 Step 3: Type conversion
        try:
            df["CustomerID"] = df["CustomerID"].astype(str)
            df["PodID"] = df["PodID"].astype(str)
        except Exception as e:
            meta = get_error_metadata("PandasLoadError", {"exception": str(e)})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="PandasLoadError",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])

        df = df.sort_values(by=["PodID", "ReportingMonth"])
        logging.info("🔢 Sorted by PodID and ReportingMonth.")

        # 🔍 Step 4: Final cleaning
        df = clean_dataframe(df)
        logging.info("🧹 Data cleaned.")

        return df

    except Exception as fallback:
        meta = get_error_metadata("UnknownError", {"exception": str(fallback)})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="UnknownError",
            severity=meta["severity"],
            component=meta["component"]
        )
        safe_exit(meta["code"], meta["message"])


def create_month_and_year_columns(df: pd.DataFrame, column: str = 'ReportingMonth') -> pd.DataFrame:
    df.reset_index(inplace=True)
    df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'])
    df['Month'] = df['ReportingMonth'].dt.month
    df['Year'] = df['ReportingMonth'].dt.year
    return df

import numpy as np

def create_month_and_year_columns_(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhances the dataframe with temporal features:
    - Year
    - Month
    - Sine/Cosine month encodings for seasonality
    - TimeIndex (optional): number of months since start

    Assumes 'ReportingMonth' is a datetime column.
    """
    df.reset_index(inplace=True)
    df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'])
    df["Year"] = df["ReportingMonth"].dt.year
    df["Month"] = df["ReportingMonth"].dt.month

    # Add cyclical encoding for seasonality
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    # Optional: continuous time index (months since start)
    df["TimeIndex"] = (
        (df["ReportingMonth"].dt.year - df["ReportingMonth"].dt.year.min()) * 12
        + df["ReportingMonth"].dt.month
    )

    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Remove CSV index column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop constant columns like UserForecastMethodID
    if 'UserForecastMethodID' in df.columns and df['UserForecastMethodID'].nunique() == 1:
        df.drop(columns=['UserForecastMethodID'], inplace=True)

    # Convert ReportingMonth to datetime and set as index
    df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'], format='%Y-%m-%d')
    df['ReportingMonth'] = df['ReportingMonth'].dt.to_period('M').dt.to_timestamp()

    df.set_index('ReportingMonth', inplace=True)

    logging.info("✅ Raw dataset cleaned.")
    return df


def get_unique_list_of_customer_and_pod(df: pd.DataFrame) -> Tuple[list, list]:
    customers = df['CustomerID'].unique().tolist()
    pod_ids = df['PodID'].unique().tolist()

    if customers:
        logging.info(f"🧮 Forecasting for {len(customers)}")
    else:
        logging.warning("⚠️ No Customer IDs found for forecasting.")

    return customers, pod_ids

def get_forecast_range(ufm_config: ForecastConfig) -> pd.DatetimeIndex:
    try:
        forecast_dates = pd.date_range(start=ufm_config.start_date, end=ufm_config.end_date, freq='MS')
        logging.info(f"📅 Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
        return forecast_dates
    except Exception as e:
        logging.error(f"Failed to generate forecast range: {e}")
        return pd.DatetimeIndex([])

def get_melted_df(df: pd.DataFrame, customer_col: str= 'CustomerID',id_vars=None, value_vars=None, consumption_type: str = "ConsumptionType", value_name: str = 'kWh'):
    if id_vars is None:
        id_vars = ['CustomerID', 'ReportingMonth']
    if value_vars is None:
        value_vars = [
            'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
            'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
            'Block4Consumption', 'NonTOUConsumption'
        ]
    top_customers = df[customer_col].value_counts().head(4).index.tolist()
    sample_df = df.reset_index()
    sample_df = sample_df[sample_df[customer_col].isin(top_customers)]
    # Melt the dataframe for easier time series plotting
    melted_df = pd.melt(sample_df,id_vars=id_vars,value_vars=value_vars,var_name=consumption_type,value_name=value_name)
    return melted_df

def extract_xgboost_params(param_str: str) -> Tuple[float, float, float,float ,float]:
    try:
        matches = re.findall(r'\(.*?\)', param_str)
        params = tuple(map(float, matches[0].strip('()').split(',')))
        return params
    except Exception as e:
        logging.error(f"❌ Failed to parse model parameters: {e}")
        return (0, 0, 0, 0, 0)


import re
import logging
from typing import Tuple, Optional, Union

def extract_random_forest_params(
    param_str: str
) -> Tuple[
    Optional[int],       # n_estimators
    Optional[int],       # max_depth
    Optional[int],       # min_samples_split
    Optional[int],       # min_samples_leaf
    Union[int, float, str, None],  # max_features
    bool                 # bootstrap
]:
    """
    Parse a string like "(50,10,2,1,0.75,true)" into a 6-tuple of RF params:
      (n_estimators, max_depth, min_samples_split,
       min_samples_leaf, max_features, bootstrap)

    max_features : {"sqrt", "log2", None}, int or float, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    If the string contains exactly one value—e.g. "(100)"—we return the
    full set of defaults instead. On any other parse error, we also fall
    back to defaults.
    """
    # Define the “canonical” defaults
    DEFAULT_PARAMS: Tuple[
        Optional[int], Optional[int], Optional[int],
        Optional[int], Union[int, float, str, None], bool
    ] = (
        100,     # n_estimators
        10,    # max_depth
        2,       # min_samples_split
        1,       # min_samples_leaf
        5,  # max_features
        True     # bootstrap
    )

    try:
        # 1) Extract inside of first parentheses
        match = re.search(r'\((.*?)\)', param_str)
        if not match:
            raise ValueError("No parentheses found")

        parts = [p.strip() for p in match.group(1).split(',')]

        # 2) Edge case: single value → return full defaults
        if len(parts) == 1:
            logging.info(
                f"ℹ️ Only one parameter '{parts[0]}' provided; "
                "falling back to all default RF params"
            )
            return DEFAULT_PARAMS

        # 3) Must have exactly six
        if len(parts) != 6:
            raise ValueError(f"Expected 6 values, got {len(parts)}")

        # 4) Convert first four to ints
        n_estimators    = int(parts[0])
        max_depth       = int(parts[1])
        min_samples_split = int(parts[2])
        min_samples_leaf  = int(parts[3])

        # 5) Parse max_features
        mf = parts[4].lower()
        if mf in ("sqrt", "log2"):
            max_features: Union[int, float, str, None] = mf
        elif mf in ("none", ""):
            max_features = None
        else:
            # try int, then float
            try:
                max_features = int(parts[4])
            except ValueError:
                max_features = float(parts[4])

        # 6) Last: boolean
        bool_str = parts[5].lower()
        bootstrap = bool_str in ("true", "tru", "1", "yes")

        return (
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            bootstrap
        )

    except Exception as e:
        logging.error(f"❌ Failed to parse RF params '{param_str}': {e}")
        return DEFAULT_PARAMS


def extract_sarimax_params(param_str: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    try:
        matches = re.findall(r'\(.*?\)', param_str)
        order = tuple(map(int, matches[0].strip('()').split(',')))
        seasonal_order = tuple(map(int, matches[1].strip('()').split(','))) if len(matches) > 1 else None
        logging.info(f"📌 Parsed ARIMA Order: {order}, Seasonal Order: {seasonal_order}")
        return order, seasonal_order
    except Exception as e:
        logging.error(f"❌ Failed to parse model parameters: {e}")
        return (0, 0, 0), (0, 0, 0, 0)

def get_single_time_series_for_single_customer(df, column: str='PeakConsumption') -> tuple[Any, Any, Any]:
    # Prepare dataset for a single customer to perform time series diagnostics
    df = df.reset_index()
    single_customer = df['CustomerID'].value_counts().idxmax()
    cust_df = df[df['CustomerID'] == single_customer].sort_values('ReportingMonth')
    # Select a single time series (e.g., PeakConsumption)
    ts = cust_df.set_index('ReportingMonth')[column]
    return ts, single_customer, cust_df

