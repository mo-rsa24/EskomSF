import pandas as pd
import logging
import re
from typing import Tuple, Optional

# Setup logger
logging.basicConfig(level=logging.INFO)


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    logging.info("✅ Raw dataset loaded.")

    # Type conversion
    df['CustomerID'] = df['CustomerID'].astype(str)


    # Sorting
    df = df.sort_values(by=["PodID", "ReportingMonth"])
    df = clean_dataframe(df)

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Remove CSV index column
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Drop constant columns like UserForecastMethodID
    if 'UserForecastMethodID' in df.columns and df['UserForecastMethodID'].nunique() == 1:
        df.drop(columns=['UserForecastMethodID'], inplace=True)

    # Convert ReportingMonth to datetime and set as index
    df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'], format='%Y-%m-%d') \
        .dt.to_period('M').dt.to_timestamp()
    df.set_index('ReportingMonth', inplace=True)

    logging.info("✅ Raw dataset cleaned.")
    return df


def get_unique_list_of_customer_and_pod(df: pd.DataFrame) -> Tuple[list, list]:
    customers = df['CustomerID'].unique().tolist()
    pod_ids = df['PodID'].unique().tolist()

    if customers:
        logging.info(f"🧮 Forecasting for {len(customers)} customers: {customers}")
    else:
        logging.warning("⚠️ No Customer IDs found for forecasting.")

    return customers, pod_ids

def get_forecast_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    try:
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        logging.info(f"📅 Forecast period: {forecast_dates[0]} to {forecast_dates[-1]}")
        return forecast_dates
    except Exception as e:
        logging.error(f"Failed to generate forecast range: {e}")
        return pd.DatetimeIndex([])

def extract_sarimax_params(param_str: str) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
    try:
        matches = re.findall(r'\(.*?\)', param_str)
        order = tuple(map(int, matches[0].strip('()').split(',')))
        seasonal_order = tuple(map(int, matches[1].strip('()').split(','))) if len(matches) > 1 else (0,0,0)
        logging.info(f"📌 Parsed ARIMA Order: {order}, Seasonal Order: {seasonal_order}")
        return order, seasonal_order
    except Exception as e:
        logging.error(f"❌ Failed to parse model parameters: {e}")
        return (0, 0, 0), (0, 0, 0, 0)