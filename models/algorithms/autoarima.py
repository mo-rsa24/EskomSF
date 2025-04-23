from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from data.dml import *
import joblib
import pandas as pd
from typing import Tuple, Union, Optional, NamedTuple
from db.queries import ForecastConfig
from evaluation.performance import *
from hyperparameters import get_model_hyperparameters
from models.base import ForecastModel


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

def _apply_log(series: pd.Series, log: bool) -> pd.Series:
    """
    Optionally apply a log transformation to the series.
    Adds a small constant for numerical stability.
    """
    if log:
        return np.log(series + 1e-6)
    return series


def train_time_series_model(
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        log: bool = False
) -> Union[ARIMAResults, SARIMAXResults]:
    """
    Train an ARIMA or SARIMA model on the provided series.

    Args:
        series (pd.Series): The training time series.
        order (Tuple[int, int, int]): ARIMA order (p, d, q).
        seasonal_order (Optional[Tuple[int, int, int, int]]): SARIMA seasonal order (P, D, Q, s).
        log (bool, optional): Whether to apply log transformation.

    Returns:
        Union[ARIMAResults, SARIMAXResults]: The fitted model.
    """
    series = _apply_log(series, log)

    if seasonal_order is None:
        logger.info("🔧 Training non-seasonal ARIMA model")
        model = ARIMA(series, order=order)
        fitted_model = model.fit(method_kwargs={"maxiter": 200})
    else:
        logger.info(f"🔧 Training seasonal ARIMA (SARIMA) model with seasonal_order={seasonal_order}")
        model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit(method_kwargs={"maxiter": 200})

    return fitted_model


def predict_time_series_model(
        model: Union[ARIMAResults, SARIMAXResults],
        order: Tuple[int, int, int],
        steps: int,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        log: bool = False
) -> pd.Series:
    """
    Generate forecasts using the fitted time series model.

    Parameters:
        model (Union[ARIMAResults, SARIMAXResults]): The fitted model.
        order (Tuple[int, int, int]): The ARIMA order parameters.
        steps (int): Number of steps to forecast.
        seasonal_order (Optional[Tuple[int, int, int, int]]): Seasonal order parameters if applicable.
        log (bool, optional): Whether to reverse a log transform (exponentiate results). Defaults to False.

    Returns:
        pd.Series: Forecasted values.
    """
    # Note: The seasonal_order parameter in predict may not be required for SARIMAX;
    # adjust if the API differs. Here, we assume a unified interface.
    if seasonal_order is None:
        results = model.forecast(steps=steps, params=order)
    else:
        results = model.forecast(steps=steps, params=order, seasonal_order=seasonal_order)

    if log:
        results = np.exp(results)
    return results


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def mean_forecast(series: pd.Series, n_periods: int) -> pd.Series:
    """Mean forecast using average of training series."""
    return np.full(n_periods, series.mean())

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Evaluate forecast using RMSE and R2 metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def generate_forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    metrics = evaluate_forecast(y_true, y_pred)
    baseline = mean_forecast(pd.Series(y_true), len(y_true))
    baseline_metrics = evaluate_forecast(y_true, baseline)
    return metrics, baseline_metrics

def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series
) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Return (metrics, baseline_metrics) given actual and forecast dataset.
    """
    metrics, baseline_metrics = generate_forecast_metrics(y_true, y_pred)
    return metrics, baseline_metrics

def load_model(path):
    return joblib.load(path)


def evaluate_model_on_test(model_path: str, order: Tuple[int, int, int],  n_periods: int, seasonal_order: Tuple[int, int, int, int] = None, log: bool = False) -> pd.Series:
    model = load_model(model_path)
    forecast = predict_time_series_model(model, order, n_periods, seasonal_order, log=log)
    return forecast


def forecast_arima_for_single_customer(model: ForecastModel, log: bool = False) -> pd.DataFrame:
    """
    Forecast ARIMA/SARIMA models for a single customer using information embedded in the model instance.

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
    # Extract necessary fields from the model
    df: pd.DataFrame = model.dataset.processed_df
    # customer_ids, variable_ids = model.dataset.parse_identifiers()
    # customer_id: str = customer_ids[0]
    customer_id: str = "7397925811"
    ufm_config = model.dataset.ufm_config
    selected_columns: Optional[List[str]] = model.config.selected_columns
    consumption_types: Optional[List[str]] = model.config.consumption_types
    mode: str = model.config.mode
    log: bool = model.config.log
    arima = None
    # Extract hyperparameters from the config using our helper
    order, seasonal_order = get_model_hyperparameters(ufm_config.forecast_method_name, ufm_config.model_parameters)
    logger.info(f"💡 Extracted hyperparameters: order={order}, seasonal_order={seasonal_order}")

    # Filter the DataFrame for the target customer (assumes a 'CustomerID' column)
    customer_data = df[df['CustomerID'] == customer_id]
    customer_data = customer_data.sort_values('PodID')
    if customer_data.empty:
        logger.error(f"🚫 No data for customer {customer_id}. Aborting forecast.")
        raise ValueError(f"Customer {customer_id} not present in the dataset.")

    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)
    arima_rows: List[ModelPodPerformance] = []
    # Process each pod for the customer
    unique_pod_ids: List[str] = list(customer_data["PodID"].unique())
    logger.info(f"📊 Processing {len(unique_pod_ids)} pod(s) for customer {customer_id}.")
    forecast_map: Dict[str, pd.Series] = {}
    all_forecasts = []
    # Iterate over each unique pod and perform forecasting (pseudo-code shown here)
    for pod_id in unique_pod_ids:
        pod_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        steps = len(model.dataset.forecast_dates)
        performance_data = forecast_for_podel_id(
            pod_df, order, customer_id, pod_id, consumption_types, ufm_config,
            mode=mode, seasonal_order=seasonal_order, log=log, steps=steps
        )
        consumer_performance_data.pod_by_id_performance.append(performance_data)

        # now convert the long-format DataFrame to a wide row:
        df_long = performance_data.performance_data_frame
        wide = finalize_model_performance_df(
            df_long,
            model_name=ufm_config.forecast_method_name,
            databrick_id=getattr(ufm_config, 'databrick_id', None),
            user_forecast_method_id=ufm_config.user_forecast_method_id
        )
        # df_long has columns ['consumption_type','RMSE','R2', ...]
        metrics: Dict[str, Dict[str, float]] = {
            row.consumption_type: {"RMSE": row.RMSE, "R2": row.R2}
            for row in df_long.itertuples(index=False)
        }

        # --- 2) instantiate the dataclass ---
        mpp = ModelPodPerformance(
            ModelName=ufm_config.forecast_method_name,
            CustomerID=customer_id,
            PodID=pod_id,
            DataBrickID=getattr(ufm_config, "databrick_id", None),
            UserForecastMethodID=ufm_config.user_forecast_method_id,
            metrics=metrics
        )
        arima_rows.append(mpp)

        # Consumption Forecast
        forecast_map = df_long.set_index("consumption_type")["forecast"].to_dict()
        forecast_df = build_forecast_df(
            forecast_map,
            customer_id,
            pod_id,
            list(metrics.keys()),  # exactly the types we have
            ufm_config.user_forecast_method_id
        )
        all_forecasts.append(forecast_df)

        logger.info(f"✅ Processed pod {pod_id} for customer {customer_id}.")

    all_performance_df: pd.DataFrame = consumer_performance_data.get_pod_performance_data()
    arima_performance_df = pd.DataFrame([m.to_row() for m in arima_rows])
    final_performance_df = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)
    forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)

    logger.info("✅ Forecast aggregation complete.")
    return final_performance_df


def forecast_arima_model(
    model: Union[ARIMAResults, SARIMAXResults],
    order: Tuple[int,int,int],
    steps: int,
    seasonal_order: Optional[Tuple[int,int,int,int]]=None,
    log: bool=False
) -> pd.Series:
    """
    Generate forecasts from a previously fitted ARIMA/SARIMA model.
    """
    forecast = predict_time_series_model(model, order, steps, seasonal_order, log=log)
    return forecast

def fit_arima_model(
    train_series: pd.Series,
    order: Tuple[int,int,int],
    seasonal_order: Optional[Tuple[int,int,int,int]]=None,
    log: bool=False,
    save_path: Optional[str]=None
) -> Union[ARIMAResults, SARIMAXResults]:
    """
    Trains an ARIMA/SARIMA model on the given series.
    Optionally saves the fitted model to disk.
    """
    model = train_time_series_model(train_series, order, seasonal_order, log=log)
    if save_path:
        joblib.dump(model, save_path)
    return model


def prepare_time_series_data(
    df: pd.DataFrame,
    consumption_type: str
) -> pd.Series:
    """
    Convert a specified consumption column into a time-indexed Series.
    Ensures monthly frequency, fills missing, etc.
    """
    series = pd.to_numeric(df[consumption_type], errors='coerce')
    series.index = pd.to_datetime(series.index)  # convert index
    series = series.asfreq('MS').fillna(0)       # monthly start freq, fill missing
    return series

def tune_arima_with_cv(
    series: pd.Series,
    param_grid: List[Tuple[int, int, int]],
    seasonal_grid: List[Optional[Tuple[int, int, int, int]]],
    log: bool = False,
    n_splits: int = 3
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:

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
                    model = train_time_series_model(train_series, order, seasonal_order, log=log)
                    preds = predict_time_series_model(model, order, len(val_series), seasonal_order, log=log)
                    metrics, _ = evaluate_predictions(val_series, preds)
                    rmse_scores.append(metrics['RMSE'])
                except Exception as e:
                    logging.warning(f"Failed combo {order}, {seasonal_order}: {e}")
                    continue

            if rmse_scores:
                avg_rmse = sum(rmse_scores) / len(rmse_scores)
                if avg_rmse < best_score:
                    best_score = avg_rmse
                    best_params = (order, seasonal_order)

    logging.info(f"✅ Best ARIMA config: {best_params} with RMSE: {best_score:.2f}")
    return best_params


def is_series_valid(series: pd.Series, min_length=30) -> Tuple[bool, str]:
    if series.nunique() <= 1:
        return False, "Constant series"
    if (series == 0).all():
        return False, "All zeros"
    if len(series) < min_length:
        return False, "Too few observations"
    return True, "Series valid"


def forecast_for_podel_id(
        df: pd.DataFrame,
        order: Tuple[int, int, int],
        customer_id: str,
        pod_id: str,
        consumption_types: List[str],
        ufm_config: ForecastConfig,
        mode: str = 'train',  # can be 'train', 'validation', 'test'
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        log: bool = False
) -> PodIDPerformanceData:
    """
    Forecast approach with three modes: 'validation', 'train', 'test'.

    1) 'validation': train on sub_train, forecast validation
         - Split into sub_train, validation, final_test
         - Train on sub_train -> evaluate on validation
         - Possibly do hyperparam tuning or cross-validation here
    2) 'train': after picking the best hyperparams,
         - Retrain on sub_train + validation (or just sub_train, your choice)
         - Forecast on that same range or keep it for final usage
         - Optionally do an internal check on the tail of that combined set.
    3) 'test': final unseen forecast on final_test
         - Use or load the final model (trained on sub_train+validation)
           to forecast final_test, purely for final evaluation
    """
    logging.info("Predicting ARIMA for pod_id=%s in mode=%s", pod_id, mode)
    metric_keys = {'RMSE', 'MAE', 'R2'}
    data = []

    forecast_horizon =  get_forecast_range(ufm_config)
    steps = len(forecast_horizon)
    # Directory structure for saving/loading
    model_dir = f"model_configuration/{ufm_config.forecast_method_name}/customer_{customer_id}/pod_{pod_id}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for consumption_type in consumption_types:
        if df[consumption_type].isnull().all():
            continue

        # Initialize placeholders
        forecast = None
        metrics = {'RMSE': None, 'MAE': None, 'R2': None}
        baseline_metrics = {'RMSE': None, 'MAE': None, 'R2': None}

        # 1) Prepare the time series for this consumption type
        series = prepare_time_series_data(df, consumption_type)

        valid, reason = is_series_valid(series)
        if not valid:
            logging.warning(f"Skipping {consumption_type} for pod {pod_id}: {reason}")
            forecast = pd.Series([0] * steps)  # placeholder forecast
            metrics = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}
            baseline_metrics = {'RMSE': np.nan, 'MAE': np.nan, 'R2': np.nan}

            data.append({
                'pod_id': pod_id,
                'customer_id': customer_id,
                'consumption_type': consumption_type,
                'forecast': forecast,
                **{k: metrics[k] for k in metric_keys},
                **{f'{k}_baseline': baseline_metrics[k] for k in metric_keys}
            })
            continue  # Skip to the next consumption type

        # 2) Split into sub_train, validation, final_test
        sub_train, validation, final_test = split_time_series_three_way(series)

        model_path = os.path.join(model_dir, f"{consumption_type}.pkl")

        if mode == 'validation':
            # -------------------------------
            # A) Train on sub_train only
            # B) Forecast exactly the length of validation
            # C) Evaluate on validation
            # -------------------------------

            # Candidate hyperparameters
            param_grid = [(1, 1, 1), (2, 1, 2), (0, 1, 1)]
            seasonal_grid = [None, (1, 0, 1, 12), (0, 1, 1, 12)]  # Example, can be extended


            # Tune ARIMA with cross-validation
            best_order, best_seasonal = tune_arima_with_cv(
                sub_train,
                param_grid=param_grid,
                seasonal_grid=seasonal_grid,
                log=log
            )

            model = fit_arima_model(sub_train, best_order, best_seasonal, log=log)

            if len(validation) > 0:
                val_forecast = forecast_arima_model(model, best_order, steps=len(validation), seasonal_order=best_seasonal, log=log)
                metrics, baseline_metrics = evaluate_predictions(validation, val_forecast)
                forecast = val_forecast
            # Potentially store model or intermediate results if you want
            params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")
            joblib.dump({'order': best_order, 'seasonal_order': best_seasonal}, params_path)

        elif mode == 'train':
            train_plus_val = pd.concat([sub_train, validation])
            # Try loading best hyperparams from validation, if available
            params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")

            if os.path.exists(params_path):
                logging.info(f"🔁 Loading best ARIMA params for {consumption_type} from validation.")
                best_params = joblib.load(params_path)
                best_order = best_params.get('order', order)
                best_seasonal = best_params.get('seasonal_order', seasonal_order)
            else:
                logging.warning(f"⚠️ No tuned params found for {consumption_type}. Using default.")
                best_order = order
                best_seasonal = seasonal_order

            model = fit_arima_model(train_plus_val, best_order, best_seasonal, log=log, save_path=model_path)

            if len(validation) > 0:
                tail_forecast = forecast_arima_model(model, best_order, steps=len(validation), seasonal_order=best_seasonal, log=log)
                forecast = tail_forecast
                metrics, baseline_metrics = evaluate_predictions(validation, tail_forecast)
        elif mode == 'test':
            train_plus_val = pd.concat([sub_train, validation])
            # Try loading best hyperparams from validation
            params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")

            if os.path.exists(params_path):
                logging.info(f"🔁 Loading best ARIMA params for {consumption_type} from validation.")
                best_params = joblib.load(params_path)
                best_order = best_params.get('order', order)
                best_seasonal = best_params.get('seasonal_order', seasonal_order)
            else:
                logging.warning(f"⚠️ No tuned params found for {consumption_type}. Using default.")
                best_order = order
                best_seasonal = seasonal_order

            # Fit model on entire train+val set
            model = fit_arima_model(train_plus_val, best_order, best_seasonal, log=log)
            # Forecast on final test
            if len(final_test) > 0:
                test_forecast = forecast_arima_model(model, best_order, steps=len(final_test),
                                                     seasonal_order=best_seasonal, log=log)
                metrics, baseline_metrics = evaluate_predictions(final_test, test_forecast)
            else:
                logging.warning("⚠️ No final test dataset to evaluate.")

            model = fit_arima_model(series, best_order, best_seasonal, log=log)
            forecast = forecast_arima_model(model, best_order, steps=steps,
                                                     seasonal_order=best_seasonal, log=log)
        else:
            raise ValueError("mode must be 'validation', 'train', or 'test'")

        # Collect results
        # Depending on your pipeline, you might store different forecasts in different modes
        row = {
            'pod_id': pod_id,
            'customer_id': customer_id,
            'consumption_type': consumption_type,
            'forecast': forecast
        }
        for key in metric_keys:
            row[key] = metrics[key]
            row[f'{key}_baseline'] = baseline_metrics[key]

        data.append(row)

    performance_data_frame = pd.DataFrame(data)
    return PodIDPerformanceData(
        pod_id,
        ufm_config.forecast_method_name,
        customer_id,
        ufm_config.user_forecast_method_id,
        performance_data_frame
    )