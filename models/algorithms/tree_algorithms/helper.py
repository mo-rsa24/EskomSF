from typing import Union

from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


def engineer_data(df: pd.DataFrame, cons: str, lags: list, windows: list):
    df, stl_obj = stl_decompose(df, cons, period=12)
    df = engineer_calendar(df)
    df = engineer_lags(df, "deseasoned", lags)
    df = engineer_rolling(df, "deseasoned", windows)
    df = engineer_interactions(df, lags)
    df, feature_cols = assemble_features(df, lags, windows)
    return df, feature_cols, stl_obj

def stl_decompose(df: pd.DataFrame, target: str, period: int = 12) -> (pd.DataFrame, STL):
    """
    Perform STL seasonal decomposition and add 'seasonal' and 'deseasoned' columns.
    Returns modified df and fitted STL object.
    """
    stl = STL(df[target], period=period, robust=True).fit()
    df["seasonal"] = stl.seasonal
    df["deseasoned"] = df[target] - df["seasonal"]
    return df, stl


def engineer_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical month and binary flags.
    """
    df["month"]     = df.index.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df["is_feb"]    = (df.month == 2).astype(int)
    return df


def engineer_lags(df: pd.DataFrame, base_series: str, lags: list) -> pd.DataFrame:
    """
    Create lag features for the deseasoned series.
    """
    for lag in lags:
        df[f"ds_lag{lag}"] = df[base_series].shift(lag)
    return df


def engineer_rolling(df: pd.DataFrame, base_series: str, windows: list) -> pd.DataFrame:
    """
    Create rolling-mean and -std features.
    """
    for w in windows:
        df[f"ds_roll_mean_{w}m"] = df[base_series].rolling(w).mean().shift(1)
        df[f"ds_roll_std_{w}m"]  = df[base_series].rolling(w).std().shift(1)
    return df


def engineer_interactions(df: pd.DataFrame,  lags: list[int]) -> pd.DataFrame:
    """
    Add custom interaction terms, e.g. lag12 * month_sin.
    """
    year_lags = [lag for lag in lags if lag % 12 == 0]

    for lag in year_lags:
        col = f"ds_lag{lag}"
        # sin interaction
        df[f"{col}_x_sin"] = df[col] * df["month_sin"]
        # cos interaction (optional, but often useful)
        df[f"{col}_x_cos"] = df[col] * df["month_cos"]
    return df


def assemble_features(
    df: pd.DataFrame,
    lags: list[int],
    windows: list[int]
) -> tuple[pd.DataFrame, list[str]]:
    # 1) lag columns
    lag_cols = [f"ds_lag{lag}" for lag in lags]

    # 2) rolling statistics
    roll_cols = []
    for w in windows:
        roll_cols += [f"ds_roll_mean_{w}m", f"ds_roll_std_{w}m"]

    # 3) calendar features
    calendar_cols = ["month_sin", "month_cos", "is_feb"]

    # 4) dynamic yearly-lag interactions
    yearly_lags = [lag for lag in lags if lag % 12 == 0]
    interaction_cols = [f"ds_lag{lag}_x_sin" for lag in yearly_lags]

    # 5) assemble everything
    feature_cols = lag_cols + roll_cols + calendar_cols + interaction_cols

    # 6) drop any rows missing *any* of these features or the target
    df_clean = df.dropna(subset=feature_cols + ["deseasoned"])

    return df_clean, feature_cols



def split_train_test(df: pd.DataFrame, test_months: int = 3) -> (pd.DataFrame, pd.DataFrame):
    """
    Split df into train and test based on last N months.
    """
    hist_end = df.index.max()
    train = df.loc[: hist_end - pd.offsets.MonthBegin(test_months)]
    test  = df.loc[ hist_end - pd.offsets.MonthBegin(test_months-1):]
    return train, test


def train_rf(X: pd.DataFrame, y: pd.Series, rf_params_tuple) -> RandomForestRegressor:
    """
    Fit RandomForestRegressor with given parameters.
    """
    (
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        max_features,
        bootstrap_flag
    ) = rf_params_tuple

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth if max_depth > 0 else None,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap_flag,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf

def train_xgb(X: pd.DataFrame, y: pd.Series, xgb_params_tuple) -> XGBRegressor:
    """
    Fit XGBRegressor with given parameters.
    """
    (n_estimators, max_depth, learning_rate, subsample, colsample_bytree) = xgb_params_tuple

    xgb = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if int(max_depth) > 0 else None,
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror"
    )
    xgb.fit(X, y)
    return xgb


def recursive_forecast(
    df: pd.DataFrame,
    stl_obj: STL,
    rf: Union[RandomForestRegressor | XGBRegressor],
    features: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lags: list,
    windows: list
) -> pd.DataFrame:
    """
    Generate multi-step forecasts by feeding back predictions.
    """
    # 1) build map of average seasonal component by calendar month
    season_map = stl_obj.seasonal.groupby(stl_obj.seasonal.index.month).mean()

    # 2) keep a running deseasoned series for recursive lags
    history_ds = df["deseasoned"].copy()

    # 3) precompute which of your lags are "yearly" (multiples of 12)
    yearly_lags = [lag for lag in lags if lag % 12 == 0]

    # 4) forecast index
    forecast_index = pd.date_range(start=start, end=end, freq="MS")
    results = []
    for date in forecast_index:
        m = date.month
        seasonal = season_map[m]
        # construct features
        row = {}
        for lag in lags:
            row[f"ds_lag{lag}"] = history_ds.loc[date - pd.DateOffset(months=lag)]
        for w in windows:
            window = history_ds.loc[(date - pd.DateOffset(months=w)) : (date - pd.DateOffset(months=1))]
            row[f"ds_roll_mean_{w}m"] = window.mean()
            row[f"ds_roll_std_{w}m"]  = window.std()
        row.update({
            "month_sin": np.sin(2*np.pi*m/12),
            "month_cos": np.cos(2*np.pi*m/12),
            "is_feb":   int(m==2),
        })
        for lag in yearly_lags:
            row[f"ds_lag{lag}_x_sin"] = row[f"ds_lag{lag}"] * np.sin(2 * np.pi * m / 12)
        X_pred = pd.DataFrame([row], index=[date])[features]
        ds_pred = rf.predict(X_pred)[0]
        history_ds.loc[date] = ds_pred
        results.append((date, ds_pred + seasonal))
    return pd.DataFrame(results, columns=["date","forecast"]).set_index("date")

def plot_forecast(pod_df: pd.DataFrame, fc_df: pd.DataFrame, consumption_type: str, end_fc: pd.Timestamp):
    plt.figure(figsize=(10, 4))
    plt.plot(pod_df.index, pod_df[consumption_type], label="Historical")
    plt.plot(fc_df.index, fc_df["forecast"], "--", label="Forecast")
    plt.title(f"{consumption_type}: {pod_df.index.min().date()} → {end_fc}")
    plt.xlabel("ReportingMonth")
    plt.ylabel(consumption_type)
    plt.legend()
    plt.show()

def plot_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_col: str,
    train_pred: np.ndarray,
    test_pred: np.ndarray
):
    """
    Plot actual vs predicted for train and test splits.
    """
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(train_df.index, train_df[y_col], label="Actual")
    ax[0].plot(train_df.index, train_pred, "--", label="Pred")
    ax[0].set_title("Train Set")

    ax[1].plot(test_df.index, test_df[y_col], label="Actual")
    ax[1].plot(test_df.index, test_pred, "--", label="Pred")
    ax[1].set_title("Test Set")

    for a in ax:
        a.set_ylabel(y_col)
        a.legend()
    plt.tight_layout()
    plt.show()






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
