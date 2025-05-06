import os

import numpy as np
import pandas as pd
import joblib
from typing import Tuple, Optional, Dict, Union, List
import logging
from typing import Dict, Tuple, Union

# The “canonical” RF params and their validation + defaults
RF_PARAM_NAMES = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "n_jobs",
    "bootstrap",
]

XGB_PARAM_NAMES = [
    "n_estimators",      # 5.0
    "max_depth",         # 2.0
    "learning_rate",     # 0.3
    "subsample",         # 0.3
    "colsample_bytree",  # 0.8
]

PARAM_BOUNDS = {
    "n_estimators":      lambda v: isinstance(v, int)   and v >= 1,
    "max_depth":         lambda v: v is None or (isinstance(v, int) and v >= 1),
    "min_samples_split": lambda v: isinstance(v, int)   and v >= 2,
    "min_samples_leaf":  lambda v: isinstance(v, int)   and v >= 1,
    "n_jobs":            lambda v: v is None or v == -1 or (isinstance(v, int) and v >= 1),
    "bootstrap":         lambda v: isinstance(v, bool),
}
DEFAULTS = {
    "n_estimators":      100,
    "max_depth":         None,
    "min_samples_split": 2,
    "min_samples_leaf":  1,
    "n_jobs":            None,
    "bootstrap":         True,
}
XGB_PARAM_BOUNDS = {
    "n_estimators": lambda v: isinstance(v, int) and v >= 1,
    "max_depth":    lambda v: isinstance(v, int) and v >= 1,
    "learning_rate":lambda v: 0.0 < v <= 1.0,
    "subsample":    lambda v: 0.0 < v <= 1.0,
    "colsample_bytree": lambda v: 0.0 < v <= 1.0,
}

XGB_DEFAULTS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

def regressor_grid_for_pipeline(
    grid: Union[Dict[str, list], Tuple],
    prefix: str = "regressor__",
    param_names: list = None,
) -> Dict[str, list]:
    """
    1) Turn a tuple of 6 values into a dict (zipped against RF_PARAM_NAMES),
    2) Validate each param (or list of params) against sklearn rules,
       substituting a default if invalid,
    3) Prefix every key for Pipeline compatibility.
    """
    # 1️⃣ If user gave a tuple, map it to a dict first
    if isinstance(grid, tuple):
        names = param_names or RF_PARAM_NAMES
        if len(grid) != len(names):
            raise ValueError(
                f"Expected tuple of length {len(names)}, got {len(grid)}"
            )
        grid = dict(zip(names, grid))

    # 2️⃣ Validate & possibly substitute defaults
    validated: Dict[str, list] = {}
    for key, vals in grid.items():
        # wrap single values in a list for uniformity
        values = vals if isinstance(vals, list) else [vals]

        if key in PARAM_BOUNDS:
            check = PARAM_BOUNDS[key]
            good = [v for v in values if check(v)]
            if not good:
                # nothing valid → fall back to default
                default = DEFAULTS[key]
                logging.warning(
                    f"Parameter '{key}' has no valid values {values}; "
                    f"resetting to default {default}"
                )
                validated[key] = [default]
            else:
                validated[key] = good
        else:
            # unknown param – pass it through
            validated[key] = values

    # 3️⃣ Prefix keys
    updated_grid: Dict[str, list] = {}
    for key, val_list in validated.items():
        new_key = key if key.startswith(prefix) else f"{prefix}{key}"
        updated_grid[new_key] = val_list

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
