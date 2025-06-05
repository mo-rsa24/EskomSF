from datetime import datetime
import os
from typing import Optional, Any, Tuple

import pandas as pd

from hyperparameters import get_model_hyperparameters
from models.algorithms.helper import _convert_to_model_performance_row, _convert_forecast_map_to_df, \
   _aggregate_forecast_outputs, _collect_metrics
import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from evaluation.performance import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
   StratifiedKFold, GridSearchCV, KFold)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from db.queries import ForecastConfig
from data.dml import convert_column, create_lag_features, prepare_lag_features, create_month_and_year_columns, \
   prepare_features_and_target, create_month_and_year_columns_
from models.algorithms.utilities import evaluate_predictions
from models.base import ForecastModel

# Setup logger with basic configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib._store_backends")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")
warnings.filterwarnings("ignore", category=ConvergenceWarning)


cache_dir = os.path.join(os.getcwd(), 'pipeline_cache')
memory = joblib.Memory(location=cache_dir, verbose=0)

def is_series_valid(series: pd.Series, min_length: int = 3) -> Tuple[bool, str]:
   if series.isnull().all():
       return False, "All NaN values"
   if series.nunique() <= 1:
       return False, "Constant or all zero series"
   if len(series) < min_length:
       return False, "Too few observations"
   return True, "Series valid"

import os
import logging
import joblib
import pandas as pd
import numpy as np

from typing import Dict, Tuple, List, Any
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def prepare_global_training_data(model) -> Dict[str, Any]:
   """
   Prepares the dataset for global model training.
   Returns full df, feature df, and feature column list.
   """
   consumption_types = getattr(model.dataset, 'variable_ids', None) or model.config.consumption_types
   df = model.dataset.processed_df.copy()

   # Encode categorical IDs
   df['CustomerID_encoded'] = LabelEncoder().fit_transform(df['CustomerID'].astype(str))
   df['PodID_encoded'] = LabelEncoder().fit_transform(df['PodID'].astype(str))

   # Temporal features
   df = create_month_and_year_columns_(df)

   # Lag features
   df = create_lag_features(df, model.config.selected_columns, lags=3)
   df, lag_features = prepare_lag_features(df, lag_columns=consumption_types)

   # Select feature columns (no metadata)
   feature_columns = lag_features + ['Month_sin', 'Month_cos', 'TimeIndex', 'CustomerID_encoded', 'PodID_encoded']
   feature_df = df[feature_columns]
   feature_df = feature_df.loc[:, (feature_df != 0).any(axis=0)]
   feature_columns = feature_df.columns.tolist()

   return {
       "full_df": df,
       "feature_df": feature_df,
       "feature_columns": feature_columns
   }


def train_global_rf_models(
   data: Dict[str, pd.DataFrame],
   model
) -> Dict[str, RandomForestRegressor]:
   """
   Trains or loads one RandomForestRegressor per consumption type.
   Saves each trained model under:
     model_configuration/{forecast_method_name}/global/{consumption_type}.pkl
   Returns:
     { consumption_type : RandomForestRegressor }
   """
   ufm_config = model.dataset.ufm_config
   consumption_types = getattr(model.dataset, "variable_ids", None) or model.config.consumption_types

   full_df = data["full_df"]
   feature_df = data["feature_df"]

   base_dir = os.path.join("model_configuration", ufm_config.forecast_method_name, "global")
   os.makedirs(base_dir, exist_ok=True)

   trained_models: Dict[str, RandomForestRegressor] = {}

   rf_params_tuple = get_model_hyperparameters("randomforest", ufm_config.model_parameters)
   # Expecting something like: (50, 10, 2, 1, 3, True)
   (n_estimators,
    max_depth,
    min_samples_split,
    min_samples_leaf,
    max_features,
    bootstrap_flag) = rf_params_tuple

   for ctype in consumption_types:
       model_path = os.path.join(base_dir, f"{ctype}.pkl")

       if os.path.exists(model_path):
           logger.info(f"📦 Loading existing RF model for '{ctype}' from {model_path}")
           trained_models[ctype] = joblib.load(model_path)
           continue

       mask = full_df[ctype].notna()
       y = full_df.loc[mask, ctype]
       X = data["feature_df"].loc[mask]

       if X.empty or y.empty:
           logger.warning(f"⚠️ Skipping '{ctype}'—no data available.")
           continue

       if y.nunique() <= 1 or len(y) < 3:
           logger.warning(f"⚠️ Skipping '{ctype}'—series invalid or too short.")
           continue

       logger.info(f"🚀 Training RF model for '{ctype}' (n_samples={len(y)})")
       # rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
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

       joblib.dump(rf, model_path)
       logger.info(f"💾 Saved RF model for '{ctype}' to {model_path}")
       trained_models[ctype] = rf

   return trained_models


def recursive_rf_forecast_for_pod(
   model: RandomForestRegressor,
   last_known_df: pd.DataFrame,
   forecast_dates: List[pd.Timestamp],
   target_col: str,
   feature_columns: List[str],
   lags: int = 3
) -> pd.Series:
   """
   Performs multi-step recursive forecasting for a single pod (last_known_df sorted by ReportingMonth).
   Returns a pd.Series indexed by forecast_dates containing predicted values.
   """
   df = last_known_df.sort_values("ReportingMonth").reset_index(drop=True)
   actual_vals = df[target_col].dropna().tolist()
   if len(actual_vals) >= lags:
       lag_queue = actual_vals[-lags:].copy()
   else:
       lag_queue = [0.0] * (lags - len(actual_vals)) + actual_vals.copy()

   template = df.iloc[-1].copy()
   preds: List[float] = []

   for date in forecast_dates:
       new_row = template.copy()
       new_row["ReportingMonth"] = date
       new_row["Month"] = date.month
       new_row["Year"] = date.year
       new_row["Month_sin"] = np.sin(2 * np.pi * date.month / 12)
       new_row["Month_cos"] = np.cos(2 * np.pi * date.month / 12)
       new_row["TimeIndex"] = ((date.year - df["ReportingMonth"].min().year) * 12 + date.month)

       for i in range(1, lags + 1):
           lag_col = f"{target_col}_lag{i}"
           new_row[lag_col] = lag_queue[-i] if len(lag_queue) >= i else 0.0

       row_df = pd.DataFrame([new_row[feature_columns]]).fillna(0)
       pred = model.predict(row_df)[0]
       preds.append(pred)
       lag_queue.append(pred)

   return pd.Series(data=preds, index=forecast_dates)


def make_continuous_index(pod_df: pd.DataFrame) -> pd.DatetimeIndex:
   """
   Given a pod's DataFrame with ReportingMonth column, return a continuous monthly index
   from min to max ReportingMonth.
   """
   start = pod_df["ReportingMonth"].min()
   end = pod_df["ReportingMonth"].max()
   return pd.date_range(start=start, end=end, freq="MS")


def build_forecast_rows(
    full_df: pd.DataFrame,
    global_models: Dict[str, RandomForestRegressor],
    feature_columns: List[str],
    forecast_dates: List[pd.Timestamp],
    consumption_types: List[str],
    user_forecast_id: Any,
    gap: str = "skip"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Iterates over each (CustomerID, PodID) group in full_df, applies gap logic,
    and returns two lists of dicts:
      1) forecast_rows: each entry has
         { "CustomerID","PodID","UserForecastMethodID","ReportingMonth","ConsumptionType","Forecast" }
      2) perf_rows: each entry is exactly `_collect_metrics(...)` for that PodID×ctype.
    """
    forecast_rows: List[Dict[str, Any]] = []
    perf_rows: List[Dict[str, Any]] = []

    grouped = full_df.groupby(["CustomerID", "PodID"], sort=False)
    for (cust_id, pod_id), pod_df in grouped:
        pod_df = pod_df.sort_values("ReportingMonth").reset_index(drop=True)
        continuous_index = make_continuous_index(pod_df)
        actual_index = pod_df["ReportingMonth"]

        if gap == "skip":
            if not actual_index.isin(continuous_index).all() or len(actual_index) != len(continuous_index):
                logger.warning(f"⚠️ Skipping Pod={pod_id} (Customer={cust_id}) due to gaps.")
                continue

        elif gap == "fill":
            pod_df = pod_df.set_index("ReportingMonth").reindex(continuous_index)
            pod_df.index.name = "ReportingMonth"
            pod_df = pod_df.ffill().reset_index()

        else:
            raise ValueError("gap must be one of ['fill','skip'].")

        # For each consumption type, compute in-sample metrics + future forecast
        for ctype in consumption_types:
            rf_model = global_models.get(ctype)
            if rf_model is None or pod_df[ctype].dropna().empty:
                # If no model or no actual values, append zero‐forecast and zero‐metrics
                zero_forecast = pd.Series([0.0] * len(forecast_dates), index=forecast_dates)
                perf_rows.append(
                    _collect_metrics(pod_id, cust_id, ctype, zero_forecast)
                )
                for date, _ in zero_forecast.items():
                    forecast_rows.append({
                        "CustomerID": cust_id,
                        "PodID": pod_id,
                        "UserForecastMethodID": user_forecast_id,
                        "ReportingMonth": date,
                        "ConsumptionType": ctype,
                        "Forecast": 0.0
                    })
                continue

            # --- 1) In‐sample prediction & metrics (last window of actuals) ---
            series = pod_df.set_index("ReportingMonth")[ctype]
            evaluation_window = min(len(series), len(forecast_dates))
            test_actual = series[-evaluation_window:]

            df_idx = pod_df.set_index("ReportingMonth")
            X_in = df_idx.loc[test_actual.index, feature_columns]
            test_pred = rf_model.predict(X_in)

            # Compute metrics exactly like ARIMA did
            metrics, baseline_metrics = evaluate_predictions(test_actual.values, test_pred)

            # --- 2) Multi‐step future forecast ---
            future_forecast = recursive_rf_forecast_for_pod(
                model=rf_model,
                last_known_df=pod_df,
                forecast_dates=forecast_dates,
                target_col=ctype,
                feature_columns=feature_columns,
                lags=3
            )

            # --- 3) Collect metrics via _collect_metrics (it returns a dict row) ---
            perf_dict = _collect_metrics(
                pod_id,
                cust_id,
                ctype,
                future_forecast,
                metrics,
                baseline_metrics
            )
            perf_rows.append(perf_dict)

            # --- 4) Append forecast rows for each date/value pair ---
            for date, val in future_forecast.items():
                forecast_rows.append({
                    "CustomerID": cust_id,
                    "PodID": pod_id,
                    "UserForecastMethodID": user_forecast_id,
                    "ReportingMonth": date,
                    "ConsumptionType": ctype,
                    "Forecast": val
                })

    return forecast_rows, perf_rows


def pivot_forecast_rows(
    forecast_rows: List[Dict[str, Any]],
    consumption_types: List[str]
) -> pd.DataFrame:
    """
    Takes a list of dicts with keys:
      "CustomerID","PodID","UserForecastMethodID","ReportingMonth","ConsumptionType","Forecast"
    and pivots it so each row is:
      [PodID, UserForecastMethodID, CustomerID, ReportingMonth, <ctype1>, <ctype2>, …]

    Any missing consumption‐type values become 0.0.
    """
    if not forecast_rows:
        columns = ["PodID", "UserForecastMethodID", "CustomerID", "ReportingMonth"] + consumption_types
        return pd.DataFrame(columns=columns)

    flat = pd.DataFrame(forecast_rows)
    pivoted = flat.pivot_table(
        index=["CustomerID", "PodID", "UserForecastMethodID", "ReportingMonth"],
        columns="ConsumptionType",
        values="Forecast",
        aggfunc="first"
    ).reset_index()

    for ctype in consumption_types:
        if ctype not in pivoted.columns:
            pivoted[ctype] = np.nan

    pivoted[consumption_types] = pivoted[consumption_types].fillna(0.0)

    final_df = pivoted[
        ["PodID", "UserForecastMethodID", "CustomerID", "ReportingMonth"] + consumption_types
    ]
    return final_df


def build_perf_df(
    perf_rows: List[Dict[str, Any]],
    consumption_types: List[str],
    model_name: str,
    user_forecast_id: Any,
    data_brick_id: Any
) -> pd.DataFrame:
    """
    Builds a DataFrame matching Performance.csv schema, based on the list of
    dicts returned by _collect_metrics for each (pod,ctype). We simply concat all rows
    for each pod, then aggregate into one row per pod with the required columns:

      [CustomerID, PodID, DataBrickID, UserForecastMethodID, ModelName,
       RMSE_Avg, R2_Avg,
       RMSE_<ctype1>, R2_<ctype1>, RMSE_<ctype2>, R2_<ctype2>, …
      ]

    Because `_collect_metrics(...)` already returns a dict containing:
       {
         'pod_id': …,
         'customer_id': …,
         'consumption_type': …,
         'forecast': pd.Series,
         'RMSE': …,
         'MAE': …,
         'R2': …,
         'baseline_RMSE': …,
         'baseline_MAE': …,
         'baseline_R2': …
       }

    We need to pivot those metrics into one row per pod.
    """
    if not perf_rows:
        # Build empty DataFrame with correct columns
        cols = [
            "CustomerID", "PodID", "DataBrickID", "UserForecastMethodID", "ModelName",
            "RMSE_Avg", "R2_Avg"
        ]
        for ctype in consumption_types:
            cols += [f"RMSE_{ctype}", f"R2_{ctype}"]
        return pd.DataFrame(columns=cols)

    df_perf = pd.DataFrame(perf_rows)

    # For each (pod_id, consumption_type), compute metrics
    # Then aggregate per pod: RMSE_<ctype>, R2_<ctype>, and compute RMSE_Avg, R2_Avg across all types
    records: List[Dict[str, Any]] = []
    grouped = df_perf.groupby(["pod_id", "customer_id"], sort=False)

    for (pod_id, cust_id), sub in grouped:
        record: Dict[str, Any] = {
            "CustomerID": cust_id,
            "PodID": pod_id,
            "DataBrickID": data_brick_id,
            "UserForecastMethodID": user_forecast_id,
            "ModelName": model_name
        }

        # Pull out RMSE and R2 for each ctype
        rmse_list = []
        r2_list = []
        for ctype in consumption_types:
            row = sub[sub["consumption_type"] == ctype]
            if row.empty:
                # no metrics for this ctype → set to 0
                record[f"RMSE_{ctype}"] = 0.0
                record[f"R2_{ctype}"] = 0.0
            else:
                rmse_val = float(row["RMSE"].iloc[0])
                r2_val = float(row["R2"].iloc[0])
                record[f"RMSE_{ctype}"] = rmse_val
                record[f"R2_{ctype}"] = r2_val
                rmse_list.append(rmse_val)
                r2_list.append(r2_val)

        # Compute averages across all consumption types that appeared
        record["RMSE_Avg"] = float(np.mean(rmse_list)) if rmse_list else 0.0
        record["R2_Avg"] = float(np.mean(r2_list)) if r2_list else 0.0

        records.append(record)

    return pd.DataFrame(records)

def run_rf_forecast_pipeline(
    model,
    gap: str = "fill"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates:
      1) prepare_global_training_data
      2) train_global_rf_models
      3) build_forecast_rows → pivot to forecast_df + collect perf_rows
      4) build_perf_df

    Returns:
      - perf_df matching Performance.csv
      - forecast_df matching Forecast.csv
    """
    # 1) Global training data
    data_dict = prepare_global_training_data(model)
    full_df: pd.DataFrame = data_dict["full_df"]
    feature_columns: List[str] = data_dict.get("feature_columns")

    # 2) Train/load global models
    logger.info("🔧 Training/loading global RF models …")
    global_models = train_global_rf_models(data_dict, model)

    ufm_config = model.dataset.ufm_config
    model_name = ufm_config.forecast_method_name
    user_forecast_id = ufm_config.user_forecast_method_id
    data_brick_id = ufm_config.databrick_task_id

    # 3) Build forecast + perf rows
    logger.info("🔮 Building forecast & performance rows …")
    consumption_types = getattr(model.dataset, "variable_ids", None) or model.config.consumption_types
    forecast_dates = model.dataset.forecast_dates

    forecast_rows, perf_rows = build_forecast_rows(
        full_df=full_df,
        global_models=global_models,
        feature_columns=feature_columns,
        forecast_dates=forecast_dates,
        consumption_types=consumption_types,
        user_forecast_id=user_forecast_id,
        gap=gap
    )

    # 4) Pivot into final forecast_df
    forecast_df = pivot_forecast_rows(forecast_rows, consumption_types)

    # 5) Build perf_df
    perf_df = build_perf_df(
        perf_rows=perf_rows,
        consumption_types=consumption_types,
        model_name=model_name,
        user_forecast_id=user_forecast_id,
        data_brick_id=data_brick_id
    )

    logger.info("✅ RF pipeline complete. Returning perf_df & forecast_df.")
    return perf_df, forecast_df
