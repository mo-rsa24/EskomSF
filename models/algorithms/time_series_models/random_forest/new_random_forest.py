# new_random_forest.py

import logging
from typing import List, Tuple, Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data.dml import get_forecast_range
from evaluation.performance import RandomForestPerfRow, PodIDPerformanceData
from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _collect_metrics
from models.algorithms.tree_algorithms.helper import naive_last_value_forecast
from models.algorithms.time_series_models.random_forest.config import RFConfig
from models.algorithms.time_series_models.random_forest.feature_engineering import FeatureEngineer
from models.algorithms.time_series_models.random_forest.manager import RFModelManager
from models.algorithms.time_series_models.random_forest.rf_factory import EncoderFactory
from models.algorithms.time_series_models.random_forest.utils import evaluate_regression, train_test_split_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_rf_forecast_pipeline(
    model: Any  # expecting an instance of ForecastModel (with .dataset and .config)
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_raw: pd.DataFrame = model.dataset.processed_df.copy()
    df_raw.reset_index(inplace=True)
    df_raw['ReportingMonth'] = pd.to_datetime(df_raw['ReportingMonth'])

    config_hp = model.config  # HyperParameterConfig
    ufm_config = model.dataset.ufm_config

    # 1) Build the monthly forecast horizon
    forecast_horizon = get_forecast_range(ufm_config)
    forecast_start, forecast_end = forecast_horizon[0], forecast_horizon[-1]

    # 2) Compute horizon length in months relative to last historical date
    last_hist = df_raw["ReportingMonth"].max()
    horizon_months = (
        (pd.to_datetime(forecast_end).year - last_hist.year) * 12 +
        (pd.to_datetime(forecast_end).month - last_hist.month)
    )
    horizon_months = max(horizon_months, 0)

    # 3) Dynamic lag_list: [1,2,3,6] + [12,24,36,...] up to horizon
    base_lags = [1, 2, 3, 6]
    year_lags = [12 * i for i in range(1, (horizon_months // 12) + 1)]
    lag_list = sorted(set(base_lags + year_lags))

    # 4) Dynamic rolling_windows: always include [3,12], add [24,36,...] ≤ horizon
    base_rolls = [3, 12]
    multi_rolls = [12 * i for i in range(2, (horizon_months // 12) + 1)]
    rolling_windows = sorted(set(base_rolls + multi_rolls))

    # 5) Dynamic min_history = max(max(lag_list), max(rolling_windows)+1)
    min_lag = max(lag_list) if lag_list else 0
    min_roll = max(rolling_windows) + 1 if rolling_windows else 0
    min_history = max(min_lag, min_roll)

    # 6) Unpack RF hyperparameters from HyperParameterConfig
    rf_params_tuple = get_model_hyperparameters("randomforest", ufm_config.model_parameters)
    # rf_params_tuple should be: (n_estimators, max_depth, min_samples_split,
    #                             min_samples_leaf, max_features, bootstrap_flag)

    rf_config = RFConfig(
        lag_list=lag_list,
        rolling_windows=rolling_windows,
        encoder_method=getattr(config_hp, "encoder_method", "frequency"),
        rf_params=rf_params_tuple,
        min_history=min_history,
        test_fraction=getattr(config_hp, "test_fraction", 0.2),
    )

    # 7) Identify which consumption column to model
    consumption_col = config_hp.consumption_types[0]

    # 8) Core group / date columns
    group_cols = ["CustomerID", "PodID"]
    date_col = "ReportingMonth"

    # 11) Train per‐series models
    manager = RFModelManager(
        df=df_raw,
        group_cols=group_cols,
        date_col=date_col,
        consumption_col=consumption_col,
        config=rf_config
    )

    rf_global, global_perf_row = manager.train_global()
    global_perf_df = pd.DataFrame([global_perf_row.to_row()])

    series_perf_rows, in_sample_forecasts = manager.train_all_series()
    series_perf_df = pd.DataFrame([row.to_row() for row in series_perf_rows])
    # 12) Combine global + series performance
    perf_df_all = pd.concat([global_perf_df, series_perf_df], ignore_index=True)
    perf_df_all.fillna(0, inplace=True)

    # 13) Combine in‐sample test forecasts
    combined_in_sample = (
        pd.concat(in_sample_forecasts, ignore_index=True)
        if in_sample_forecasts else pd.DataFrame()
    )

    # 14) Generate future horizons for each trained series
    future_forecasts: List[pd.DataFrame] = []
    for (cust_id, pod_id), rf_model in manager.models.items():
        df_fut = manager.forecast_for_pod(cust_id, pod_id, forecast_horizon)
        future_forecasts.append(df_fut)
    combined_future = (
        pd.concat(future_forecasts, ignore_index=True)
        if future_forecasts else pd.DataFrame()
    )

    # 15) Final concatenation: in‐sample test + future
    forecast_df_all = pd.concat([combined_in_sample, combined_future], ignore_index=True)

    return perf_df_all, forecast_df_all


def prepare_global_training_data(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    consumption_col: str,
    config: RFConfig
) -> Tuple[pd.DataFrame, List[str]]:
    fe = FeatureEngineer(
        group_cols=group_cols,
        date_col=date_col,
        consumption_cols=[consumption_col],
        lag_list=config.lag_list,
        rolling_windows=config.rolling_windows,
        seasonal=True,
        drop_na=True
    )
    feats_df, feature_cols = fe.generate(df)

    # 1) If TariffID exists, copy it over, then encode and drop the raw
    if "TariffID" in df.columns:
        feats_df["TariffID"] = df["TariffID"]

    encoder = EncoderFactory.get_encoder(config.encoder_method)
    new_id_cols = []
    for col in group_cols + ["TariffID"]:
        if col in feats_df.columns:
            enc_col = f"{col}_enc"
            if config.encoder_method == "frequency":
                feats_df[enc_col] = encoder(feats_df[col])
            else:
                feats_df[enc_col] = encoder(feats_df, col, consumption_col)
            new_id_cols.append(enc_col)

    # 2) Drop raw TariffID now that “TariffID_enc” exists
    if "TariffID" in feats_df.columns:
        feats_df = feats_df.drop(columns=["TariffID"])

    feature_cols.extend(new_id_cols)
    return feats_df, feature_cols

def train_global_rf_models(
    feats_df: pd.DataFrame,
    feat_cols: List[str],
    consumption_col: str,
    date_col: str,
    config: RFConfig
) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    """
    Splits feats_df into train/test by date, fits one RF, evaluates,
    and returns the model + a one‐row perf DataFrame.
    """
    train_df, test_df = train_test_split_time(feats_df, date_col, config.test_fraction)
    X_train = train_df[feat_cols]
    y_train = train_df[consumption_col]
    X_test = test_df[feat_cols]
    y_test = test_df[consumption_col]

    (n_estimators, max_depth, min_samples_split,
     min_samples_leaf, max_features, bootstrap_flag) = config.rf_params

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
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    perf = evaluate_regression(y_test.values, preds)

    perf_row = RandomForestPerfRow(
        cust_id="ALL", pod_id="ALL",
        mse=perf["MSE"], mae=perf["MAE"], r2=perf["R2"],
        n_train=len(X_train), n_test=len(X_test)
    )
    perf_df = pd.DataFrame([perf_row.to_row()])
    logger.info(
        f"[Global RF] train_count={len(X_train)}, test_count={len(X_test)}, R2={perf['R2']:.3f}"
    )
    return rf, perf_df


def forecast_rf_for_podel_id(
    df: pd.DataFrame,
    customer_id: str,
    pod_id: str,
    consumption_types: List[str],
    ufm_config: Any,
    rf_config_template: RFConfig,
    forecast_dates: pd.DatetimeIndex
) -> PodIDPerformanceData:
    rows = []

    # 1) Filter down to this customer‐pod combination
    mask = (df["CustomerID"] == customer_id) & (df["PodID"] == pod_id)
    df_series = df.loc[mask].sort_values("ReportingMonth").reset_index(drop=True)

    if df_series.empty:
        # If no history, return zero‐filled forecasts/baselines for each consumption_type
        for ctype in consumption_types:
            zero_forecast = pd.Series([0.0] * len(forecast_dates), index=forecast_dates)
            metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
            baseline_metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
            row = _collect_metrics(pod_id, customer_id, ctype, zero_forecast, metrics, baseline_metrics)
            rows.append(row)

        return PodIDPerformanceData(
            pod_id=pod_id,
            forecast_method_name=ufm_config.forecast_method_name,
            customer_id=customer_id,
            user_forecast_method_id=ufm_config.user_forecast_method_id,
            performance_data_frame=pd.DataFrame(rows)
        )

    # 2) For each consumption_type, train & forecast
    for ctype in consumption_types:
        series = df_series.set_index("ReportingMonth")[ctype].copy()

        # Skip if invalid (all‐zero or constant)
        if series.nunique() <= 1:
            logger.warning(
                f"⚠️ Series for Pod {pod_id}, {ctype} is constant or too short. Using zero forecast."
            )
            zero_forecast = pd.Series([0.0] * len(forecast_dates), index=forecast_dates)
            metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
            baseline_metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}
            row = _collect_metrics(pod_id, customer_id, ctype, zero_forecast, metrics, baseline_metrics)
            rows.append(row)
            continue

        # 2A) Build an RFConfig for this consumption_type by cloning rf_config_template,
        #     but setting its consumption_cols = [ctype].  (That’s how FeatureEngineer knows which column to lag.)
        rf_config = RFConfig(
            lag_list=rf_config_template.lag_list,
            rolling_windows=rf_config_template.rolling_windows,
            encoder_method=rf_config_template.encoder_method,
            rf_params=rf_config_template.rf_params,
            min_history=rf_config_template.min_history,
            test_fraction=rf_config_template.test_fraction,
        )

        # 2B) Train a ONE‐SERIES model
        #     We instantiate a manager that uses only this one consumption column.
        manager = RFModelManager(
            df=df_series[["ReportingMonth", "CustomerID", "PodID", "TariffID", ctype]].copy(),
            group_cols=["CustomerID", "PodID"],
            date_col="ReportingMonth",
            consumption_col=ctype,
            config=rf_config
        )
        # Train global is not used – we want exactly one series model.
        # So call train_all_series(), but it will create exactly one (cust,pod).
        perf_rows, in_sample_forecasts = manager.train_all_series()
        # There is exactly one perf_row for this (cust,pod,ctype).
        perf_row = perf_rows[0] if perf_rows else None

        # Extract in‐sample metrics from perf_row (if available)
        if perf_row and not perf_row.error:
            metrics = {
                "MAE": perf_row.mae,
                "RMSE": np.sqrt(perf_row.mse) if perf_row.mse >= 0 else 0.0,
                "R2": perf_row.r2
            }
        else:
            metrics = {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0}

        # 2C) Baseline: train_fraction = 1 – test_fraction
        train_frac = 1.0 - rf_config.test_fraction
        in_sample_baseline, future_baseline, baseline_metrics = naive_last_value_forecast(
            full_series=series,
            train_fraction=train_frac,
            forecast_horizon=forecast_dates
        )

        # 2D) Collect future forecast (recursive multi‐step)
        df_future = manager.forecast_for_pod(customer_id, pod_id, list(forecast_dates))
        # df_future has columns: ["CustomerID","PodID","ReportingMonth","PredictedConsumption"]
        forecast_series = pd.Series(
            df_future["PredictedConsumption"].values,
            index=pd.DatetimeIndex(df_future["ReportingMonth"])
        )

        # 2E) Build the row dict via _collect_metrics(...)
        row = _collect_metrics(pod_id, customer_id, ctype, forecast_series, metrics, baseline_metrics)
        rows.append(row)

    # 3) Return PodIDPerformanceData
    return PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=pd.DataFrame(rows)
    )
