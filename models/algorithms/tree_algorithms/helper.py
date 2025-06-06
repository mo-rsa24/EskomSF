# helper.py (optional; not used in the final pipeline)

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _recursive_rf_forecast_for_pod(
    model: RandomForestRegressor,
    pod_df: pd.DataFrame,
    forecast_dates: List[pd.Timestamp],
    target_col: str,
    feature_columns: List[str],
    lags: int = 3
) -> pd.Series:
    """
    – Given the last-known pod_df (sorted by ReportingMonth),
      recursively predict one month at a time.
    – Return a pd.Series of length=len(forecast_dates), indexed by forecast_dates.
    """
    df = pod_df.sort_values("ReportingMonth").reset_index(drop=True)
    actual_vals = df[target_col].dropna().tolist()

    # Initialize lag queue (last `lags` observed values, or pad with zeros)
    if len(actual_vals) >= lags:
        lag_queue = actual_vals[-lags:].copy()
    else:
        lag_queue = [0.0] * (lags - len(actual_vals)) + actual_vals.copy()

    template_row = df.iloc[-1].copy()
    preds: List[float] = []

    for date in forecast_dates:
        new_row = template_row.copy()
        new_row["ReportingMonth"] = date
        new_row["month"] = date.month
        new_row["Year"] = date.year
        new_row["Month_sin"] = np.sin(2 * np.pi * date.month / 12)
        new_row["Month_cos"] = np.cos(2 * np.pi * date.month / 12)
        new_row["TimeIndex"] = ((date.year - df["ReportingMonth"].min().year) * 12 + date.month)

        # Build lag columns for this new date
        for i in range(1, lags + 1):
            lag_col = f"{target_col}_lag{i}"
            if len(lag_queue) >= i:
                new_row[lag_col] = lag_queue[-i]
            else:
                new_row[lag_col] = 0.0

        row_df = pd.DataFrame([new_row[feature_columns]]).fillna(0)
        pred_val = model.predict(row_df)[0]
        preds.append(pred_val)
        lag_queue.append(pred_val)

    return pd.Series(data=preds, index=forecast_dates)


def naive_last_value_forecast(
    full_series: pd.Series,
    train_fraction: float,
    forecast_horizon: pd.DatetimeIndex
) -> (pd.Series, dict):
    """
    Splits full_series into train/test by `train_fraction`, computes
    a “last‐value” forecast for the test period and future horizon.
    Returns:
      - in_sample_baseline: pd.Series indexed by the test portion
      - future_baseline: pd.Series indexed by forecast_horizon
    Also returns baseline_metrics = {"MAE":..., "RMSE":..., "R2":...} computed on the test portion.
    """

    # 1) Chronological split
    n = len(full_series)
    test_size = max(int(n * (1 - train_fraction)), 1)
    train_end = n - test_size

    train_series = full_series.iloc[:train_end]
    test_series = full_series.iloc[train_end:]

    if train_series.empty:
        # no training data → baseline is zeros
        baseline_metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
        future_baseline = pd.Series(
            [0.0] * len(forecast_horizon), index=forecast_horizon
        )
        return pd.Series(dtype=float), future_baseline, baseline_metrics

    last_val = train_series.iloc[-1]

    # 2) In‐sample (test) baseline: repeat last_val for each index in test_series
    in_sample_baseline = pd.Series(
        [last_val] * len(test_series), index=test_series.index
    )

    # 3) Compute baseline metrics on test_series vs. in_sample_baseline
    mae_baseline = float(mean_absolute_error(test_series.values, in_sample_baseline.values))
    rmse_baseline = float(np.sqrt(mean_squared_error(test_series.values, in_sample_baseline.values)))
    # R² of a flat‐line model: if test_series is constant, define R² = 1.0; else:
    if np.allclose(test_series.values, last_val):
        r2_baseline = 1.0
    else:
        r2_baseline = float(r2_score(test_series.values, in_sample_baseline.values))

    baseline_metrics = {
        "MAE": mae_baseline,
        "RMSE": rmse_baseline,
        "R2": r2_baseline
    }

    # 4) Future‐horizon baseline: repeat last_val for every date
    future_baseline = pd.Series(
        [last_val] * len(forecast_horizon), index=forecast_horizon
    )

    return in_sample_baseline, future_baseline, baseline_metrics
