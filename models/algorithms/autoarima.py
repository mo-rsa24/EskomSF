from pmdarima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from data import grouped_engineer_features
from data.dml import *
import joblib
import pandas as pd
from typing import Tuple, Union, Optional, NamedTuple

from data.lag_safety import validate_lag_vs_horizon, log_lag_strategy
from db.queries import ForecastConfig
from db.error_logger import insert_profiling_error
from evaluation.performance import *
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _aggregate_forecast_outputs, _convert_to_model_performance_row, \
    _convert_forecast_map_to_df, _get_customer_data, _collect_metrics, _plot_forecast, _apply_log, \
    extract_exogenous_features
from models.algorithms.utilities import prepare_time_series_data, evaluate_predictions
from models.base import ForecastModel
from sa_holiday_loader import get_sa_holidays

# Setup logger with basic configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Optionally, define a named tuple for forecast results
class ForecastResult(NamedTuple):
    forecast: pd.Series
    metrics: dict
    baseline_metrics: dict


def fit_time_series_model(series, order, seasonal_order, log=False, exog=None):
    if seasonal_order:
        return train_sarima_model(series, order, seasonal_order, log, exog)
    return train_arima_model(series, order, log, exog)

def train_auto_arima(series, exog, seasonal=True, m=12):
    model = auto_arima(series, exogenous=exog, seasonal=seasonal, m=m,
                       stepwise=True, trace=True, error_action='ignore', suppress_warnings=True)
    return model

def train_arima_model(series: pd.Series, order: Tuple[int, int, int], log: bool = False, endog: Optional[pd.DataFrame] = None, exog: Optional[pd.DataFrame] = None):
    logger.info("🔧 Training ARIMA with exog")
    series = _apply_log(series, log)
    model = ARIMA(series, order=order, exog=exog)
    return model.fit(method_kwargs={"maxiter": 200})


def train_sarima_model(series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int], log: bool = False, endog: Optional[pd.DataFrame] = None, exog: Optional[pd.DataFrame] = None):
    logger.info(f"🔧 Training seasonal SARIMA with exog: {exog.columns.tolist() if exog is not None else 'None'}")
    series = _apply_log(series, log)
    model = SARIMAX(endog=series,exog=exog, order=order, seasonal_order=seasonal_order)
    return model.fit(method_kwargs={"maxiter": 200})




def predict_time_series_model(model, steps, return_ci=False, log = False):
    forecast = model.get_forecast(steps=steps)
    mean_forecast = forecast.predicted_mean
    if log:
        mean_forecast = np.exp(mean_forecast)
    if return_ci:
        conf_int = forecast.conf_int()
        return mean_forecast, conf_int
    return mean_forecast

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)


def evaluate_model_on_test(model_path: str, n_periods: int) -> pd.Series:
    model = load_model(model_path)
    forecast = predict_time_series_model(model,n_periods)
    return forecast

def _load_or_tune_params(model_dir, consumption_type, series, default_order, default_seasonal, log, mode):
    params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")

    if mode == 'validation':
        if os.path.exists(params_path):
            params = joblib.load(params_path)
            best_order, best_seasonal = params['order'], params['seasonal_order']
            logger.info(f"🔁 Loaded existing ARIMA params for {consumption_type}: {best_order}, {best_seasonal}")
        else:
            logger.info(f"🔍 Tuning ARIMA params for {consumption_type} during validation mode")
            param_grid = [default_order, (1, 1, 1), (2, 1, 2)]
            seasonal_grid = [default_seasonal, None, (1, 0, 1, 12)]
            best_order, best_seasonal = tune_arima_with_cv(series, param_grid, seasonal_grid, log)
            joblib.dump({'order': best_order, 'seasonal_order': best_seasonal}, params_path)
            logger.info(f"💾 Saved tuned ARIMA params for {consumption_type}: {best_order}, {best_seasonal}")
    else:
        if os.path.exists(params_path):
            params = joblib.load(params_path)
            best_order, best_seasonal = params['order'], params['seasonal_order']
            logger.info(f"🔁 Loaded existing ARIMA params for {consumption_type}: {best_order}, {best_seasonal}")
        else:
            best_order, best_seasonal = default_order, default_seasonal
            logger.info(f"⚙️ Using default ARIMA params for {consumption_type}: {best_order}, {best_seasonal}")

    return best_order, best_seasonal

def tune_arima_with_cv(series: pd.Series, param_grid: List[Tuple[int, int, int]], seasonal_grid: List[Optional[Tuple[int, int, int, int]]], log: bool = False, n_splits: int = 3) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
    """
    Perform cross-validation to select the best ARIMA or SARIMA parameters.
    """
    best_score = float('inf')
    best_params = (None, None)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for order in param_grid:
        for seasonal_order in seasonal_grid:
            rmse_scores = []
            for train_idx, val_idx in tscv.split(series):
                train_series = series.iloc[train_idx]
                val_series = series.iloc[val_idx]
                try:
                    model = fit_time_series_model(train_series, order, seasonal_order, log)
                    preds = predict_time_series_model(model, order, len(val_series), seasonal_order, log)
                    metrics, _ = evaluate_predictions(val_series, preds)
                    rmse_scores.append(metrics['RMSE'])
                except Exception as e:
                    logger.warning(f"Failed combo {order}, {seasonal_order}: {e}")
                    continue
            if rmse_scores:
                avg_rmse = sum(rmse_scores) / len(rmse_scores)
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = (order, seasonal_order)

    logger.info(f"✅ Best ARIMA config: {best_params} with RMSE: {best_score:.2f}")
    return best_params

@profiled_function(category="model_training",enabled=profiling_switch.enabled)
@databricks_safe(error_key="ModelFitFailure")  # fallback
def forecast_arima_for_single_customer(model: ForecastModel):
    """
    Function: Forecast ARIMA/SARIMA models for a single customer using information embedded in the model instance.

    This function extracts all necessary parameters from the ForecastModel instance, including:
      - The processed DataFrame from the ForecastDataset.
      - The customer identifier.
      - Forecast configuration and selected consumption columns.

    It then:
      1. Extracts hyperparameters (order and seasonal_order) using the configured model_parameters string.
      2. Filters the data for the specific customer.
      3. Iterates over unique pod IDs for the customer.
      4. For each pod, trains a time series model and performs forecasting.
      5. Aggregates the performance metrics into a consolidated DataFrame.
    Args:
        model (ForecastModel): An instance of ForecastModel (or subclass) containing all forecasting parameters.
        log (bool, optional): Whether to apply log transformation. Defaults to False.

    Returns:
        pd.DataFrame: Aggregated forecasting results in long format.
    """
    try:
        forecast_method_id = getattr(model.dataset.ufm_config, "forecast_method_id", None)
        # Step 1: Validate dataset
        if model.dataset is None or model.dataset.processed_df is None:
            meta = get_error_metadata("ModelConfigMissing", {"field": "dataset.df"})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="ModelConfigMissing",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])

        if model.dataset.processed_df.empty:
            meta = get_error_metadata("EmptySeries", {"forecast_method_id": forecast_method_id})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="EmptySeries",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])

        df = model.dataset.processed_df
        unique_customers, unique_pod_ids = model.dataset.extract_unique_customers_and_pods()
        ufm_config = model.dataset.ufm_config
        consumption_types = getattr(model.dataset, 'variable_ids', None) or model.config.consumption_types

        order, seasonal_order = get_model_hyperparameters(ufm_config.forecast_method_name, ufm_config.model_parameters)
        logger.info(f"💡 Extracted hyperparameters: order={order}, seasonal_order={seasonal_order}")

        all_forecasts = []
        arima_model_performances_dataframes: List[pd.DataFrame] = []
        for customer_id in unique_customers:
            customer_data = _get_customer_data(df, customer_id)
            if customer_data.empty:
                logger.warning(f"🚫 No data found for customer {customer_id}, skipping.")
                continue

            consumer_perf = CustomerPerformanceData(customer_id=customer_id, columns=consumption_types)
            arima_rows: List[ModelPodPerformance] = []

            unique_pod_ids = customer_data['PodID'].unique().tolist()
            for pod_id in unique_pod_ids:
                pod_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
                logger.info(f"🚀 Forecasting Customer {customer_id}, Pod {pod_id}")
                pod_perf = forecast_for_podel_id(
                    pod_df, order, customer_id, pod_id, consumption_types, ufm_config,
                    forecast_model=model, seasonal_order=seasonal_order)
                consumer_perf.pod_by_id_performance.append(pod_perf)
                # --- Performance Dataclass (Modularized) ---
                mpp = _convert_to_model_performance_row(pod_perf, customer_id, pod_id, ufm_config)
                arima_rows.append(mpp)

                # --- Forecast DataFrame (Pandas) ---
                forecast_df = _convert_forecast_map_to_df(pod_perf, customer_id, pod_id, ufm_config)
                all_forecasts.append(forecast_df)
                logger.info(f"✅ Processed pod {pod_id} for customer {customer_id}.")

            arima_performance_df = pd.DataFrame([m.to_row() for m in arima_rows])
            arima_model_performances_dataframes.append(arima_performance_df)

            logger.info("✅ Forecast aggregation complete for {customer_id} complete.")

        # Combine across all customers
        arima_performance = pd.concat(arima_model_performances_dataframes).reset_index().drop(columns=['index'])
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
        return arima_performance, forecast_combined_df
    except Exception as e:
        meta = get_error_metadata("ModelFitFailure", {"exception": str(e)})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",  # or traceback.format_exc()
            error_type="ModelFitFailure",
            severity=meta["severity"],
            component=meta["component"]
        )
        raise

def forecast_for_podel_id(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    customer_id: str,
    pod_id: str,
    consumption_types: List[str],
    ufm_config,
    forecast_model: ForecastModel,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> PodIDPerformanceData:
    """
    Refactored forecasting function handling in-sample, out-of-sample, and future forecasts.
    """
    data = []
    forecast_horizon = get_forecast_range(ufm_config)
    sa_holidays = get_sa_holidays(2023, 2026)
    steps = len(forecast_horizon)

    # validate_lag_vs_horizon(forecast_model.config.lag_hours, steps, fail_on_violation=False)
    # log_lag_strategy(use_lag_features=forecast_model.config.use_feature_engineering, train_mode=forecast_model.config.train_mode)

    model_dir = f"model_configuration/{ufm_config.forecast_method_name}/customer_{customer_id}/pod_{pod_id}"
    os.makedirs(model_dir, exist_ok=True)

    for consumption_type in consumption_types:
        pod_df = df[df["PodID"] == pod_id].sort_index()

        # --- Prepare series (y)
        series = prepare_time_series_data(df, consumption_type)
        if series.isnull().all() or series.nunique() <= 1:
            meta = get_error_metadata("InvalidSeries", {"pod_id": pod_id, "consumption_type":consumption_type})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="InvalidSeries",
                severity=meta["severity"],
                component=meta["component"]
            )
            logger.warning(f"⚠️ Invalid series for {consumption_type} @ Pod {pod_id}. Skipping.")
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            continue

        if len(series) < 10:
            meta = get_error_metadata("SplitConfigurationError", {
                "series_length": len(series),
                "consumption_type": consumption_type
            })
            insert_profiling_error(log_id=None, **meta, traceback="")
            forecast = pd.Series([0] * steps, index=forecast_horizon)
            data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
            logger.warning(meta["message"])
            continue

            # --- Prepare exog (X)
        if forecast_model.config.use_feature_engineering:
            engineered_df = grouped_engineer_features(
                df=pod_df,
                target_col=consumption_type,
                sa_holidays=sa_holidays,
                lag_hours=forecast_model.config.lag_hours,
                add_calendar=True,
                use_extended_calendar_features=forecast_model.config.use_extended_calendar_features,
                drop_na=True
            )

            exog_df = engineered_df.loc[series.index] \
                .drop(columns=["CustomerID", "PodID", consumption_type], errors="ignore")
        else:
            exog_df = pd.DataFrame(index=series.index)
        aligned_exog = exog_df.loc[series.index]
        sub_train, validation, final_test = split_time_series_three_way(series)
        exog_train = aligned_exog.loc[sub_train.index.union(validation.index)]
        exog_test = aligned_exog.loc[final_test.index]
        exog_future = aligned_exog.iloc[-steps:]  # Future steps


        best_order, best_seasonal = _load_or_tune_params(model_dir, consumption_type, sub_train, order, seasonal_order, forecast_model.config.log, forecast_model.config.mode)
        try:

            model = fit_time_series_model(pd.concat([sub_train, validation]), best_order, best_seasonal,
                                          forecast_model.config.log, exog=exog_train)

            in_sample_preds = model.predict(start=sub_train.index[0], end=validation.index[-1],
                                            exog=exog_train if forecast_model.config.use_feature_engineering else None)
            if forecast_model.config.log: in_sample_preds = np.exp(in_sample_preds)

            test_forecast = model.get_forecast(steps=len(final_test),
                                               exog=exog_test if forecast_model.config.use_feature_engineering else None).predicted_mean
            if forecast_model.config.log: test_forecast = np.exp(test_forecast)

            try:
                if series.index[-1] < forecast_horizon[0]:
                    shifted_start = series.index[-1] + pd.offsets.MonthBegin(1)
                    fallback_index = pd.date_range(start=shifted_start, periods=steps, freq='MS')

                    logger.warning(
                        f"⚠️ [WARNING] Forecast misalignment detected: requested start={forecast_horizon[0].date()}, "
                        f"but available data ends={series.index[-1].date()}\n"
                        f"Proceeding to generate forecast of {steps} steps starting from {shifted_start.date()}\n"
                        f"Affected: Customer={customer_id}, Pod={pod_id}, ConsumptionType={consumption_type}"
                    )

                    future_forecast = model.get_forecast(
                        steps=steps,
                        exog=exog_future if forecast_model.config.use_feature_engineering else None
                    ).predicted_mean
                    if forecast_model.config.log:
                        future_forecast = np.exp(future_forecast)
                    future_forecast.index = fallback_index

                else:
                    future_forecast = model.get_forecast(
                        steps=steps,
                        exog=exog_future if forecast_model.config.use_feature_engineering else None
                    ).predicted_mean
                    if forecast_model.config.log:
                        future_forecast = np.exp(future_forecast)
                    future_forecast.index = forecast_horizon
                full_forecast = pd.concat([in_sample_preds, test_forecast, future_forecast])

                if forecast_model.config.visualize:
                    _plot_forecast(series, full_forecast, validation, final_test)

                metrics, baseline_metrics = evaluate_predictions(final_test, test_forecast)
                data.append(
                    _collect_metrics(pod_id, customer_id, consumption_type, future_forecast, metrics, baseline_metrics))

            except IndexError as ie:
                meta = get_error_metadata("ForecastIndexError", {
                    "pod_id": pod_id,
                    "steps": steps
                })
                insert_profiling_error(log_id=None, **meta, traceback=traceback.format_exc())
                forecast = pd.Series([0] * steps, index=forecast_horizon)
                data.append(_collect_metrics(pod_id, customer_id, consumption_type, forecast))
                logger.warning(meta["message"])
                continue
        except Exception as e:
            meta = get_error_metadata("ModelFitFailure", {"exception": str(e)})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="ModelFitFailure",
                severity=meta["severity"],
                component=meta["component"]
            )
            logger.warning(f"❌ Logged model failure for {consumption_type} | Pod {pod_id}: {meta['message']}")

    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=pd.DataFrame(data)
    )