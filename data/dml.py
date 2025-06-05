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


def extract_random_forest_params(param_str: str) -> Tuple[int, int, int, int, int, bool]:
    try:
        # Extract the content inside the first pair of parentheses.
        match = re.search(r'\((.*?)\)', param_str)
        if not match:
            raise ValueError("Input string does not contain parentheses in the expected format.")

        # Split the extracted string by commas.
        parts = [part.strip() for part in match.group(1).split(',')]
        if len(parts) != 6:
            raise ValueError("Expected exactly 6 parameters.")

        # Convert the first five parts to integers.
        numeric_params = [int(part) for part in parts[:5]]

        # Process the last part: Accept any variant of "true"/"tru" as True.
        bool_str = parts[5].lower()
        bool_value = bool_str in ("true", "tru")

        return tuple(numeric_params + [bool_value])

    except Exception as e:
        logging.error(f"❌ Failed to parse model parameters: {e}")
        return (0, 0, 0, 0, 0, False)


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



def time_series_train_test_split(
    series: pd.Series,
    test_ratio: float = 0.2
) -> Tuple[pd.Series, pd.Series, int]:
    """
    Splits a time series into train and test sets, using a fraction of the dataset
    for the test set. The function ensures the test set is at least 1 record
    if the series is very small.

    Parameters
    ----------
    series : pd.Series
        The time series dataset.
    test_ratio : float, optional
        The fraction of the dataset to allocate to the test set. Default is 0.2 (20%).

    Returns
    -------
    train : pd.Series
        The training subset of the series.
    test : pd.Series
        The test (hold-out) subset of the series.
    """
    n = len(series)
    test_size = max(int(n * test_ratio), 1)
    test_size = min(test_size, n - 1)  # ensure at least 1 train point

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    return train, test, test_size


def train_test_split(X, Y, test_size=0.2):
    """
    Splits time series dataset into training and test sets while preserving the temporal order.

    Parameters:
    X (array-like): Feature dataset.
    Y (array-like): Target dataset.
    test_size (float): Proportion of the dataset to reserve for testing. Default is 0.2 (20%).

    Returns:
    tuple: A tuple containing:
           - (X_train, Y_train): Training dataset.
           - (X_test, Y_test): Testing dataset.
    """
    n_samples = len(X)
    # Determine the index at which to split the dataset.
    split_idx = int(n_samples * (1 - test_size))

    # Create the train/test split.
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    return (X_train, Y_train), (X_test, Y_test)


def split_time_series_three_way(
    series: pd.Series,
    final_test_ratio: float = 0.2,
    validation_ratio: float = 0.25
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Splits the series into:
      - final_test (the last `final_test_ratio` portion)
      - validation (the last `validation_ratio` portion of the remaining dataset)
      - sub_train (the rest)
    in chronological order.
    """
    n = len(series)
    final_test_size = max(1, int(n * final_test_ratio))
    final_test_size = min(final_test_size, n - 1)

    # Final test is the tail
    final_test = series.iloc[-final_test_size:]
    train_val = series.iloc[:-final_test_size]

    val_size = max(1, int(len(train_val) * validation_ratio))
    val_size = min(val_size, len(train_val) - 1)

    validation = train_val.iloc[-val_size:]
    sub_train = train_val.iloc[:-val_size]

    return sub_train, validation, final_test

def split_time_series_three_ways(index: pd.Index,
                                val_ratio: float = 0.25,
                                test_ratio: float = 0.2) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """
    Splits the index into sub_train, validation, final_test.
    E.g. if you have 100 points, test_ratio=0.2 => 20 test points,
    then val_ratio=0.25 => 20 of the remaining 80 => 20 val, 60 sub_train.
    """
    n = len(index)
    test_size = int(n * test_ratio)
    test_size = min(max(test_size, 0), n)
    remain_after_test = n - test_size

    val_size = int(remain_after_test * val_ratio)
    val_size = min(max(val_size, 0), remain_after_test)

    sub_train_idx = index[:remain_after_test - val_size]
    val_idx = index[remain_after_test - val_size: remain_after_test]
    final_test_idx = index[remain_after_test:]
    return sub_train_idx, val_idx, final_test_idx


def  prepare_lag_features(df, lag_columns = None, lags=3, base_features=None):
    """
    Create lag features for the specified columns, convert them to numeric,
    and construct a final list of feature columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        lag_columns (list of str): List of column names for which to create lag features.
        lags (int): Number of lag periods to generate.
        base_features (list of str, optional): List of base feature column names.
            Defaults to ["Month", "Year"].

    Returns:
        tuple: A tuple containing:
            - The modified DataFrame with lag features.
            - A list of feature column names including base features and lag features.
    """
    # Use default base_features if none provided
    if lag_columns is None:
        lag_columns = ['StandardConsumption']
    # Create lag features using the provided function (assumed to be defined elsewhere)
    df = create_lag_features(df, lag_columns, lags)

    # Generate list of lag feature column names
    lag_feature_cols = [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, lags + 1)]

    # Ensure lag feature columns are numeric and fill missing values with 0
    for col in lag_feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Construct final list of feature columns


    return df, lag_feature_cols

def prepare_features_and_target(
    df: pd.DataFrame,
    feature_columns: List[str],
    consumption_type: str
) -> (pd.DataFrame, pd.Series):
    """
    Extracts X and y from df for the given consumption_type.
    Example: X = df[feature_columns], y = df[consumption_type].
    You might handle missing or frequency alignment here too.
    """
    sub_df = df.dropna(subset=[consumption_type])
    if sub_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    X = sub_df[feature_columns]
    y = sub_df[consumption_type]
    return X, y


def convert_column(df, col: str = 'PodID', to_type: type(str) = str):
    df = df.astype({col: to_type})
    return df


def create_lag_features(df, lag_columns, lags):
    for col in lag_columns:
            for lag in range(1,lags+1):
                df[f"{col}_lag{lag}"]=df[col].shift(lag)
    return df