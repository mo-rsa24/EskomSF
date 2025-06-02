import logging
from typing import Tuple

import numpy as np
import pandas as pd

from data import grouped_engineer_features
from evaluation.performance import ModelPodPerformance, build_forecast_df, finalize_model_performance_df


def _get_customer_data(df: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    filtered = df[df['CustomerID'] == customer_id].sort_values('PodID')
    logging.info(f"📊 Found {len(filtered)} rows for CustomerID={customer_id}")
    return filtered

def _convert_to_model_performance_row(pod_perf, customer_id, pod_id, config):
    df_long = pod_perf.performance_data_frame
    metrics = {
        row.consumption_type: {"RMSE": row.RMSE, "R2": row.R2}
        for row in df_long.itertuples(index=False)
    }
    return ModelPodPerformance(
        ModelName=config.forecast_method_name,
        CustomerID=customer_id,
        PodID=pod_id,
        DataBrickID=getattr(config, "databrick_task_id", None),
        UserForecastMethodID=config.user_forecast_method_id,
        metrics=metrics
    )

def _convert_forecast_map_to_df(pod_perf, customer_id, pod_id, config):
    df_long = pod_perf.performance_data_frame
    forecast_map = df_long.set_index("consumption_type")["forecast"].to_dict()
    return build_forecast_df(
        forecast_map, customer_id, pod_id, list(forecast_map.keys()), config.user_forecast_method_id
    )

def _convert_forecast_map_to_spark_df(pod_perf, config):
    df_long = pod_perf.performance_data_frame
    return finalize_model_performance_df(
            df_long,
            model_name=config.forecast_method_name,
            databrick_id=getattr(config, 'databrick_task_id', None),
            user_forecast_method_id=config.user_forecast_method_id
    )


def _apply_log(series: pd.Series, log: bool) -> pd.Series:
    """
    Optionally apply a log transformation to the series.
    Adds a small constant for numerical stability.
    """
    if log:
        return np.log(series + 1e-6)
    return series

from sa_holiday_loader import get_sa_holidays

def extract_exogenous_features(
    df: pd.DataFrame,
    series: pd.Series,
    consumption_type: str,
    pod_id: str,
    sa_holidays: set = None,
    use_feature_engineering: bool = True,
    lag_months: list = [12, 24, 36, 48, 60],
) -> pd.DataFrame:
    """
    Extract exogenous features aligned with the target series for SARIMAX/ML models.

    - If `use_feature_engineering` is True, generate lag/calendar features via grouped_engineer_features()
    - Otherwise return empty exog_df with correct index
    - Always slices output to align with the index of prepare_time_series_data()

    Returns:
        pd.DataFrame: Aligned exog_df
    """

    # Clean the target series to determine correct index (MS frequency)
    pod_df = df[df["PodID"] == pod_id].sort_index()

    if not use_feature_engineering:
        return pd.DataFrame(index=series.index)

    if sa_holidays is None:
        sa_holidays = get_sa_holidays(2023, 2026)

    # Run grouped feature generation (only for this target & pod)
    engineered_df = grouped_engineer_features(
        df=pod_df,
        target_col=consumption_type,
        sa_holidays=sa_holidays,
        lag_hours=lag_months,
        drop_na=True
    )

    # Align and clean exog
    exog_df = (
        engineered_df
        .loc[series.index]
        .drop(columns=["CustomerID", "PodID", consumption_type], errors="ignore")
    )

    return exog_df



def _aggregate_forecast_outputs(consumer_perf, arima_rows, all_forecasts):
    all_perf_df = consumer_perf.get_pod_performance_data()
    final_perf_df = consumer_perf.convert_pod_id_performance_data(all_perf_df)
    arima_performance_df = pd.DataFrame([m.to_row() for m in arima_rows])
    forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
    logging.info("✅ Forecast aggregation complete.")
    return all_perf_df, final_perf_df, arima_performance_df, forecast_combined_df





def _collect_metrics(pod_id, customer_id, consumption_type, forecast, metrics=None, baseline_metrics=None):
    row = {'pod_id': pod_id, 'customer_id': customer_id, 'consumption_type': consumption_type, 'forecast': forecast}
    for metric in ['RMSE', 'MAE', 'R2']:
        row[metric] = metrics.get(metric) if metrics else 0.0
        row[f'{metric}_baseline'] = baseline_metrics.get(metric) if baseline_metrics else 0.0
    return row

def _plot_forecast(series, forecast, validation, final_test):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Actual')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.axvline(validation.index[-1], color='orange', linestyle=':', label='Validation Split')
    plt.axvline(final_test.index[-1], color='red', linestyle=':', label='Test Split')
    plt.legend()
    plt.show()

def is_series_valid(series: pd.Series, min_length=30) -> Tuple[bool, str]:
    if series.nunique() <= 1:
        return False, "Constant series"
    if (series == 0).all():
        return False, "All zeros"
    if len(series) < min_length:
        return False, "Too few observations"
    return True, "Series valid"