import os

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Optional, Dict
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import pmdarima as pm


PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2', 0.3],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'oob_score': [True],
    'random_state': [42]
}

def regressor_grid_for_pipeline(grid, prefix="regressor__"):
    """
    Update the keys of a hyperparameter grid by adding a prefix.
    This makes the grid compatible with a scikit-learn Pipeline.
    """
    updated_grid = {}
    for key, value in grid.items():
        # Add prefix if not already present
        if not key.startswith(prefix):
            updated_grid[f"{prefix}{key}"] = value
        else:
            updated_grid[key] = value
    return updated_grid

def load_hyperparameter_grid_rf(config = None):
    """
    Load the hyperparameter grid for Random Forest from the YAML configuration
    using the utilities module.

    The config.yaml file should contain a key 'random_forest' with the following structure:

    random_forest:
      regression:
        n_estimators: [100, 200, 300]
        max_depth: [null, 5, 10, 15]
        min_samples_split: [2, 5, 10]
        min_samples_leaf: [1, 3, 5]
        random_state: [42]
        oob_score: [true]
      classification:
        n_estimators: [100, 200, 300]
        max_depth: [null, 5, 10, 15]
        min_samples_split: [2, 5, 10]
        min_samples_leaf: [1, 3, 5]
        random_state: [42]
        class_weight: ["balanced", null]

    If the required configuration is not found, a default grid is returned.

    Args:
        regression (bool): Whether to load the grid for regression or classification.

    Returns:
        dict: Hyperparameter grid dictionary.
    """
    # Default grid in case the configuration is missing or incomplete.
    default_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        'random_state': [42],
        'oob_score': [True]
    }
    if config is not None:
        rf_config = config.get("random_forest", {})
        grid = rf_config.get("regression", default_grid)
        logging.info("Loaded hyperparameter grid from config.yaml using utilities")
        return grid
    else:
        return default_grid

def load_best_params_rf(
    model_dir: str,
    consumption_type: str
) -> Optional[Dict]:
    """Attempt to load the best hyperparameters from a file, if it exists."""
    params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")
    if os.path.exists(params_path):
        logging.info(f"Loading best RF params from {params_path}")
        return joblib.load(params_path)
    else:
        logging.warning(f"No tuned params found for {consumption_type} at {params_path}. Using default.")
        return None

def save_best_params_rf(
    best_params: Dict,
    model_dir: str,
    consumption_type: str
):
    """Save best hyperparameters to a file for later use."""
    params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")
    joblib.dump(best_params, params_path)
    logging.info(f"Saved best RF params to {params_path}: {best_params}")


# -------------------- Feature Engineering and Preprocessing --------------------

def check_stationarity(series: pd.Series, alpha: float = 0.05) -> bool:
    """
    Perform the Augmented Dickey-Fuller test to check stationarity.

    Parameters:
        series (pd.Series): The time series.
        alpha (float): Significance level.

    Returns:
        bool: True if series is stationary, else False.
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    logging.info(f"ADF test p-value: {p_value:.4f}")
    return p_value < alpha


def difference_if_needed(series: pd.Series, alpha: float = 0.05) -> Tuple[pd.Series, bool]:
    """
    Apply differencing to the series if it is not stationary.

    Parameters:
        series (pd.Series): The original time series.
        alpha (float): Significance level for stationarity.

    Returns:
        Tuple[pd.Series, bool]: The (possibly differenced) series and a flag indicating if differencing was applied.
    """
    if not check_stationarity(series, alpha):
        logging.info("Series is not stationary. Applying first differencing.")
        return series.diff().dropna(), True
    return series, False


# -------------------- Automated Hyperparameter Selection --------------------

def auto_select_arima(series: pd.Series, seasonal: bool = False, m: int = 12) -> Tuple[
    Tuple[int, int, int], Optional[Tuple[int, int, int, int]]]:
    """
    Automatically select ARIMA parameters using pmdarima's auto_arima.

    Parameters:
        series (pd.Series): Time series data.
        seasonal (bool): Whether to search for seasonal parameters.
        m (int): The seasonal period (e.g., 12 for monthly data).

    Returns:
        Tuple containing the ARIMA order and (if seasonal) the seasonal order.
    """
    logging.info("Running auto_arima to select best parameters...")
    model = pm.auto_arima(series, seasonal=seasonal, m=m, trace=True,
                          error_action='ignore', suppress_warnings=True)
    logging.info(f"Selected ARIMA order: {model.order}")
    if seasonal:
        logging.info(f"Selected seasonal order: {model.seasonal_order}")
        return model.order, model.seasonal_order
    return model.order, None


# -------------------- Evaluation Metrics and Cross-Validation --------------------

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def time_series_cv(
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        n_splits: int = 3,
        forecast_horizon: int = 12,
        log_transform: bool = False
) -> Tuple[float, float, float]:
    """
    Perform time series cross-validation.

    Parameters:
        series (pd.Series): The time series.
        order (tuple): ARIMA order.
        seasonal_order (tuple, optional): Seasonal order if applicable.
        n_splits (int): Number of CV splits.
        forecast_horizon (int): Forecast horizon for each split.
        log_transform (bool): Whether forecast/test values are log-transformed.

    Returns:
        Tuple of average RMSE, MAE, and MAPE.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses, maes, mapes = [], [], []
    for train_idx, test_idx in tscv.split(series):
        train, test = series.iloc[train_idx], series.iloc[test_idx]
        if seasonal_order:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        else:
            model = ARIMA(train, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=forecast_horizon)
        # If forecasts were computed on log-scale, revert transformation
        if log_transform:
            forecast = np.exp(forecast)
            test = np.exp(test)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mape = mean_absolute_percentage_error(test.values, forecast.values)
        rmses.append(rmse)
        maes.append(mae)
        mapes.append(mape)
    return np.mean(rmses), np.mean(maes), np.mean(mapes)


# -------------------- Ensemble Forecasting --------------------

def ensemble_forecast(
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        forecast_horizon: int = 12,
        log_transform: bool = False
) -> pd.Series:
    """
    Generate an ensemble forecast by averaging forecasts from an ARIMA/SARIMA model and an ETS model.

    Parameters:
        series (pd.Series): The time series data.
        order (tuple): ARIMA order.
        seasonal_order (tuple, optional): Seasonal order for SARIMA.
        forecast_horizon (int): Number of steps to forecast.
        log_transform (bool): Whether to apply exponentiation to reverse a log transform.

    Returns:
        pd.Series: Ensemble forecast.
    """
    logging.info("Fitting ARIMA/SARIMA model for ensemble forecast...")
    if seasonal_order:
        model_arima = SARIMAX(series, order=order, seasonal_order=seasonal_order).fit()
    else:
        model_arima = ARIMA(series, order=order).fit()
    forecast_arima = model_arima.forecast(steps=forecast_horizon)

    logging.info("Fitting ETS model for ensemble forecast...")
    try:
        model_ets = ExponentialSmoothing(series, seasonal_periods=12,
                                         trend='add', seasonal='add',
                                         initialization_method='estimated').fit()
        forecast_ets = model_ets.forecast(steps=forecast_horizon)
    except Exception as e:
        logging.error(f"ETS model failed: {e}")
        forecast_ets = forecast_arima  # fallback to ARIMA forecast

    ensemble = (forecast_arima + forecast_ets) / 2
    if log_transform:
        ensemble = np.exp(ensemble)
    return ensemble


# -------------------- Drift Detection and Automated Model Updating --------------------

def detect_drift(series: pd.Series, window: int = 12, threshold: float = 0.1) -> bool:
    """
    Detect drift in the time series by comparing the mean of the most recent window with the previous window.

    Parameters:
        series (pd.Series): The time series.
        window (int): Window size to compute means.
        threshold (float): Relative change threshold to flag drift.

    Returns:
        bool: True if drift is detected, else False.
    """
    if len(series) < 2 * window:
        return False
    recent_mean = series.iloc[-window:].mean()
    previous_mean = series.iloc[-2 * window:-window].mean()
    relative_change = np.abs(recent_mean - previous_mean) / previous_mean
    logging.info(f"Drift detection: relative change = {relative_change:.2f}")
    return relative_change > threshold


def update_model_if_needed(
        series: pd.Series,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        model_path: str = "model.pkl",
        log_transform: bool = False
):
    """
    Load an existing model or retrain if drift is detected or no model exists.

    Parameters:
        series (pd.Series): The time series data.
        order (tuple): ARIMA order.
        seasonal_order (tuple, optional): Seasonal order if applicable.
        model_path (str): Path to load/save the model.
        log_transform (bool): Whether the model is trained on log-transformed data.

    Returns:
        A fitted model.
    """
    try:
        existing_model = joblib.load(model_path)
        logging.info("Loaded existing model.")
    except Exception:
        logging.info("No existing model found. Will train a new model.")
        existing_model = None

    if detect_drift(series) or existing_model is None:
        logging.info("Drift detected or no model available. Retraining model.")
        if seasonal_order:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order).fit()
        else:
            model = ARIMA(series, order=order).fit()
        joblib.dump(model, model_path)
    else:
        logging.info("No significant drift detected; using existing model.")
        model = existing_model
    return model


# -------------------- Main Forecasting Pipeline --------------------

def main_forecasting_pipeline(df: pd.DataFrame, consumption_column: str, forecast_horizon: int = 12):
    """
    End-to-end pipeline:
      - Preprocess data (convert, check stationarity, apply differencing if needed).
      - Auto-select ARIMA/SARIMA parameters.
      - Evaluate using time series cross-validation.
      - Update (or train) the model if drift is detected.
      - Generate an ensemble forecast.

    Parameters:
        df (pd.DataFrame): DataFrame with a datetime index and a consumption column.
        consumption_column (str): Column name of the target consumption variable.
        forecast_horizon (int): Number of steps ahead to forecast.

    Returns:
        pd.Series: Ensemble forecast for the next forecast_horizon periods.
    """
    # Preprocessing: convert column to numeric, set datetime index with monthly frequency.
    series = pd.to_numeric(df[consumption_column], errors='coerce')
    series.index = pd.to_datetime(series.index)
    series = series.asfreq('MS').fillna(method='ffill')

    # Check stationarity; if not stationary, difference the series.
    processed_series, differenced = difference_if_needed(series)

    # Auto-select parameters; assume seasonality (e.g. m=12 for monthly data).
    seasonal = True
    order, seasonal_order = auto_select_arima(processed_series, seasonal=seasonal, m=12)

    # Evaluate performance via time series cross-validation.
    rmse, mae, mape = time_series_cv(processed_series, order, seasonal_order, n_splits=3,
                                     forecast_horizon=forecast_horizon)
    logging.info(f"CV Evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")

    # Update (or retrain) model if drift is detected.
    model_path = f"{consumption_column}_model.pkl"
    model = update_model_if_needed(processed_series, order, seasonal_order, model_path=model_path)

    # Generate ensemble forecast.
    ensemble = ensemble_forecast(processed_series, order, seasonal_order, forecast_horizon, log_transform=False)
    logging.info("Ensemble forecast generated.")

    return ensemble


# -------------------- Example Usage --------------------

# if __name__ == "__main__":
#     # For demonstration, we simulate some monthly consumption data.
#     dates = pd.date_range(start="2010-01-01", periods=150, freq='MS')
#     np.random.seed(42)
#     consumption_values = np.random.rand(150) * 100  # simulated consumption values
#     df_example = pd.DataFrame(consumption_values, index=dates, columns=["Consumption"])
#
#     forecast = main_forecasting_pipeline(df_example, "Consumption", forecast_horizon=12)
#     print("Forecast for the next 12 months:")
#     print(forecast)

