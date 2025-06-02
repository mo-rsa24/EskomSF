from datetime import datetime
import os
from typing import Optional, Any, Tuple
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
    prepare_features_and_target
from hyperparameters import get_model_hyperparameters, get_pipeline_config, get_cv_config, load_hyperparameter_grid
from models.algorithms.autoarima import evaluate_predictions
from models.algorithms.utilities import load_hyperparameter_grid_rf, regressor_grid_for_pipeline, save_best_params_rf, \
    load_best_params_rf
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

def most_frequent_params(param_list):
    """Return the most frequent parameter configuration."""
    from collections import Counter
    counter = Counter(tuple(sorted(p.items())) for p in param_list)
    most_common = counter.most_common(1)[0][0]
    return dict(most_common)


def build_rf_pipeline(random_state: int = 42) -> Pipeline:
    """
    Construct a baseline pipeline for Random Forest using external configuration.

    Returns:
        Pipeline: The constructed pipeline.
    """
    pipeline_config = get_pipeline_config("randomforest")
    imputer_strategy = pipeline_config.get("imputer_strategy", "mean")
    rfecv_config = pipeline_config.get("rfecv", {"step": 0.1, "cv": 3, "scoring": "neg_mean_absolute_error"})

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=RandomForestRegressor(random_state=random_state),
            step=rfecv_config.get("step", 0.1),
            cv=rfecv_config.get("cv", 3),
            scoring=rfecv_config.get("scoring", "neg_mean_absolute_error")
        )),
        ('regressor', RandomForestRegressor(random_state=random_state))
    ])
    return pipeline

def random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    mode: str,
    model_dir: str,
    consumption_type: str,
    param_grid: Optional[Dict]=None,
    scoring: str='neg_mean_absolute_error',
    forecast_dates: Optional[pd.DatetimeIndex] = None,
    feature_columns: Optional[List[str]] = None
) -> (Pipeline, pd.Series, Dict[str,float], Dict[str,float]):
    """
    1) If mode='validation':
         - Run nested or single-level cross-validation to find best params.
         - It picks the best parameters from that single training set, then re-fits on that same set, and calls it a day.
         - Save best params to disk.
         - Train final pipeline on sub_train => Forecast on validation => Return metrics.
    2) If mode='train':
         - Load best params if present, else default.
         - Train final pipeline on sub_train+validation => optional check => Save final model.
    3) If mode='test':
         - Load best params
         - Train on sub_train+validation => Forecast on final_test => Return metrics.
    Returns: (model, forecast, metrics, baseline_metrics)
    """

    # Early exit if no data
    if X.empty or y.empty:
        logging.warning(f"No data for consumption_type={consumption_type}. Skipping.")
        return None, pd.Series([], dtype=float), {}, {}

    # Create model dir if needed
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if mode == 'validation':
        # 1) Possible load or define parameter grid
        if param_grid is None:
            raw_grid = load_hyperparameter_grid_rf(None)
            param_grid = regressor_grid_for_pipeline(raw_grid)

        # 2) Simple approach: single-level CV or nested. Here we do single-level for brevity.
        pipeline = build_rf_pipeline()
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_params = grid_search.best_params_

        # Save best_params to file
        save_best_params_rf(best_params, model_dir, consumption_type)

        # Evaluate on the same dataset for now (since this is just sub_train or sub_train subset)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)

        return best_model, pd.Series(y_pred, index=X.index), metrics, baseline_metrics, pd.Series([], dtype=float)
    elif mode == 'train':
        # 1) Load best_params if exist
        best_params = load_best_params_rf(model_dir, consumption_type)
        if best_params is None:
            # Build defaults
            default_grid = load_hyperparameter_grid_rf(None)
            default_grid = regressor_grid_for_pipeline(default_grid)
            # e.g. pick the first in each list
            best_params = {k: v[0] for k, v in default_grid.items() if k.startswith('regressor__')}

        # 2) Final pipeline, set best params
        pipeline = build_rf_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X, y)

        # 3) Optionally evaluate in-sample
        y_pred = pipeline.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)

        # 4) Save final model
        final_model_path = os.path.join(model_dir, f"{consumption_type}.pkl")
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'hyperparams': best_params
        }
        joblib.dump({'model': pipeline, 'metadata': metadata}, final_model_path)
        logging.info(f"Saved final model to {final_model_path}")

        return pipeline, pd.Series(y_pred, index=X.index), metrics, baseline_metrics, pd.Series([], dtype=float)
    elif mode == 'test':
        # 1) Load best_params
        best_params ={}
        # 1) Try loading saved model
        model_path = os.path.join(model_dir, f"{consumption_type}.pkl")
        if os.path.exists(model_path):
            logging.info(f"🔁 Loading saved RF model for {consumption_type} from {model_path}")
            saved = joblib.load(model_path)
            pipeline = saved.get('model', saved)
        else:
            # fallback: train on full data
            logging.warning("No saved model found. Training on full data for test.")
            best_params = load_best_params_rf(model_dir, consumption_type)
            pipeline = build_rf_pipeline()
            pipeline.set_params(**best_params)
            pipeline.fit(X, y)

        # 2) In-sample evaluation
        y_in = pipeline.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_in)

        # 3) Future forecast
        try:
            future_X = make_future_features(forecast_dates, feature_columns)
            y_future = pipeline.predict(future_X)
            future_forecast = pd.Series(y_future, index=forecast_dates)
        except Exception as e:
            logging.error(f"Failed to build future features or predict: {e}")
            future_forecast = pd.Series([], dtype=float)

        return pipeline, pd.Series(y_in, index=X.index), metrics, baseline_metrics, future_forecast
    else:
        raise ValueError("mode must be one of ['validation', 'train', 'test']")

def train_random_forest_for_podel_id(
    df: pd.DataFrame,
    feature_columns: List[str],
    consumption_types: List[str],
    ufm_config: ForecastConfig,
    customer_id: str,
    pod_id: str,
    mode: str = 'train',  # 'validation','train','test'
    param_grid: Optional[Dict] = None,
    forecast_dates: Optional[pd.DatetimeIndex] = None  # <-- ADD THIS
) -> PodIDPerformanceData:
    """
    Simpler function that:
      1) Loops over each consumption_type.
      2) Prepares X,y via a helper.
      3) Calls train_or_validate_rf(...)
      4) Aggregates results in a PodIDPerformanceData object.

    The entire logic for cross-validation, loading hyperparams, saving final model
    is inside train_or_validate_rf(...), so we keep this short & modular.
    """

    logging.info("Random Forest forecasting for Pod=%s, mode=%s", pod_id, mode)
    metric_keys = {'RMSE', 'MAE', 'R2'}
    data_rows = []

    # Directory structure for saving/loading
    model_dir = os.path.join(
        "model_configuration",
        ufm_config.forecast_method_name,
        f"customer_{customer_id}",
        f"pod_{pod_id}"
    )
    os.makedirs(model_dir, exist_ok=True)

    for consumption_type in consumption_types:
        # Step 1) Prepare the dataset
        X, y= prepare_features_and_target(df, feature_columns, consumption_type)
        if X.empty or y.empty:
            logging.info(f"No data for consumption type = {consumption_type}. Skipping")
            continue

        # Step 2) Train or Validate or Test
        model, in_sample_forecast, metrics, baseline_metrics, future_forecast = random_forest(X, y,
            mode=mode,
            model_dir=model_dir,
            consumption_type=consumption_type,
            param_grid=param_grid,
            forecast_dates=forecast_dates,
            feature_columns=feature_columns)

        # Step 3) Aggregate results
        row = {
            'pod_id': pod_id,
            'customer_id': customer_id,
            'consumption_type': consumption_type,
            'in_sample_forecast': in_sample_forecast,
            'forecast': future_forecast
        }
        for key in metric_keys:
            row[key] = metrics.get(key, None)
            row[f"{key}_baseline"] = baseline_metrics.get(key, None)

        data_rows.append(row)

    # Convert to PodIDPerformanceData
    performance_df = pd.DataFrame(data_rows)
    pod_performance_data = PodIDPerformanceData(
        pod_id=pod_id,
        forecast_method_name=ufm_config.forecast_method_name,
        customer_id=customer_id,
        user_forecast_method_id=ufm_config.user_forecast_method_id,
        performance_data_frame=performance_df
    )
    return pod_performance_data

# Add near top of random_forest.py:

def make_future_features(
    forecast_dates: pd.DatetimeIndex,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Build a simplistic feature matrix for future dates.
    Here we just produce zeros (or you can implement lag-based).
    """
    # zero‐fill all features for each future date
    return pd.DataFrame(0, index=forecast_dates, columns=feature_columns)


def train_random_forest_for_single_customer(model: ForecastModel) -> pd.DataFrame:
    """
    Train Random Forest models for a single customer across multiple pods using the given ForecastModel instance.

    This function performs the following steps:
      1. Extracts required parameters (data, customer id, config, selected columns, consumption types, mode) from the model.
      2. Sets default values for selected_columns, consumption_types, and lag_features if not already provided.
      3. Preprocesses the DataFrame by converting date columns and customer IDs.
      4. Filters the data for the target customer and sorts it by PodID.
      5. Iterates over each unique pod ID:
          - Creates lag features and prepares the feature set.
          - Calls the per-pod training function to obtain performance metrics.
          - Aggregates results into a performance container.
      6. Combines and returns the performance results as a long-format DataFrame.

    Args:
        model (ForecastModel): ForecastModel instance that encapsulates the dataset, configuration,
                               customer ID, selected columns, consumption types, and mode.

    Returns:
        pd.DataFrame: Aggregated performance metrics and forecasts for each pod.

    Raises:
        ValueError: If the dataset is empty or necessary data fields are missing.
    """
    df: pd.DataFrame = model.dataset.processed_df
    # customer_ids, variable_ids = model.dataset.parse_identifiers()
    # customer_id: str = customer_ids[0]
    customer_id: str = "6632769797"
    ufm_config = model.dataset.ufm_config
    mode: str = model.config.mode

    # Set default feature/consumption columns if not provided.
    selected_columns: List[str] = model.config.selected_columns if model.config.selected_columns is not None else \
        ["StandardConsumption", "OffpeakConsumption", "PeakConsumption"]
    consumption_types: List[str] = model.config.consumption_types if model.config.consumption_types is not None else [
        "PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
        "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"
    ]
    lag_features: List[str] = model.config.selected_columns if model.config.selected_columns is not None else ['StandardConsumption']

    logger.info(f"💡 [RF] Starting training for customer {customer_id} in mode '{mode}'.")

    # Retrieve hyperparameters for Random Forest from the config (or defaults).
    param_grid: Any = get_model_hyperparameters("randomforest", ufm_config.model_parameters)
    param_grid = regressor_grid_for_pipeline(param_grid)
    logger.info(f"💡 [RF] Hyperparameters: {param_grid}")

    # Preprocess the DataFrame: set month/year columns and convert customer IDs.
    df = create_month_and_year_columns(df)
    customer_data: pd.DataFrame = df[df['CustomerID'] == customer_id].sort_values('PodID')
    customer_data = convert_column(customer_data)

    # Initialize a container for performance metrics.
    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)

    # Determine unique Pod IDs for the customer.
    unique_pod_ids: List[str] = list(customer_data["PodID"].unique())
    logger.info(f"📊 [RF] Found {len(unique_pod_ids)} pod(s) for customer {customer_id}.")

    forecast_map: Dict[str, pd.Series] = {}
    all_forecasts = []
    random_forest_rows: List[ModelPodPerformance] = []
    # Process each pod.
    for pod_id in unique_pod_ids:
        # Filter pod data and sort by ReportingMonth.
        podel_df: pd.DataFrame = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        podel_df = create_lag_features(podel_df, lag_features, lags=3)
        podel_df, feature_columns = prepare_lag_features(podel_df, lag_features)

        performance_data = train_random_forest_for_podel_id(
            podel_df,
            feature_columns,
            consumption_types,
            ufm_config,
            customer_id,
            pod_id,
            mode=mode,
            param_grid=param_grid,
            forecast_dates=model.dataset.forecast_dates
        )
        consumer_performance_data.pod_by_id_performance.append(performance_data)

        # now convert the long-format DataFrame to a wide row:
        df_long = performance_data.performance_data_frame
        wide = finalize_model_performance_df(
            df_long,
            model_name=ufm_config.forecast_method_name,
            databrick_id=getattr(ufm_config, 'databrick_task_id', None),
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
            DataBrickID=getattr(ufm_config, "databrick_task_id", None),
            UserForecastMethodID=ufm_config.user_forecast_method_id,
            metrics=metrics
        )
        random_forest_rows.append(mpp)

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
        logger.info(f"✅ [RF] Processed pod {pod_id} for customer {customer_id}.")

    # Aggregate the per-pod performance results.
    all_performance_df: pd.DataFrame = consumer_performance_data.get_pod_performance_data()
    random_forest_rows_performance_df = pd.DataFrame([m.to_row() for m in random_forest_rows])
    final_performance_df = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)
    forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
    pod_id_performance_data: pd.DataFrame = consumer_performance_data.convert_pod_id_performance_data(
        all_performance_df)
    logger.info("✅ [RF] Forecast aggregation complete.")
    return pod_id_performance_data

def is_series_valid(series: pd.Series, min_length: int = 3) -> Tuple[bool, str]:
    if series.isnull().all():
        return False, "All NaN values"
    if series.nunique() <= 1:
        return False, "Constant or all zero series"
    if len(series) < min_length:
        return False, "Too few observations"
    return True, "Series valid"

def train_random_forest_globally_forecast_locally_with_aggregation(model: ForecastModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates global RF training and local forecasting, returns performance and forecast outputs.
    """
    df, feature_columns = prepare_global_training_data(model)
    global_models = train_global_rf_models(df, feature_columns, model)
    rf_performance_df, forecast_combined_df = forecast_locally_with_global_models(df, feature_columns, global_models, model)
    return rf_performance_df, forecast_combined_df


def prepare_global_training_data(model: ForecastModel) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepares the dataset for global model training: encodes IDs, creates lag and temporal features.
    """
    consumption_types = getattr(model.dataset, 'variable_ids', None) or model.config.consumption_types
    df = model.dataset.processed_df.copy()
    df['CustomerID_encoded'] = LabelEncoder().fit_transform(df['CustomerID'].astype(str))
    df['PodID_encoded'] = LabelEncoder().fit_transform(df['PodID'].astype(str))
    df = create_month_and_year_columns(df)
    df = create_lag_features(df, model.config.selected_columns, lags=3)
    df, lag_features = prepare_lag_features(df, consumption_types)
    feature_columns = lag_features + ['Month', 'Year', 'CustomerID_encoded', 'PodID_encoded']
    return df, feature_columns


def train_global_rf_models(df: pd.DataFrame, feature_columns: List[str], model: ForecastModel) -> Dict[str, Any]:
    """
    Trains global Random Forest models for each consumption type.
    """
    global_models = {}
    ufm_config = model.dataset.ufm_config
    mode = model.config.mode
    consumption_types = getattr(model.dataset, 'variable_ids', None) or model.config.consumption_types
    for consumption_type in consumption_types:
        X, y = prepare_features_and_target(df, feature_columns, consumption_type)
        if X.empty or y.empty:
            logger.warning(f"Skipping {consumption_type} due to insufficient data.")
            continue

        # ✅ Add Global Series Validation Here
        valid, reason = is_series_valid(y)
        if not valid:
            logger.warning(f"⚠️ Skipping global model training for {consumption_type}: {reason}.")
            continue

        param_grid = get_model_hyperparameters("randomforest", ufm_config.model_parameters)
        param_grid = regressor_grid_for_pipeline(param_grid)
        model_dir = os.path.join("model_configuration", ufm_config.forecast_method_name, "global")
        os.makedirs(model_dir, exist_ok=True)

        pipeline = build_rf_pipeline()
        best_model = None
        if mode == 'validation':
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(X, y)
            best_model = search.best_estimator_
            save_best_params_rf(search.best_params_, model_dir, consumption_type)
        elif mode in ['train', 'test']:
            best_params = load_best_params_rf(model_dir, consumption_type)
            if best_params is None:
                best_params = {k: v[0] for k, v in param_grid.items()}
            pipeline.set_params(**best_params)
            pipeline.fit(X, y)
            best_model = pipeline
            if mode == 'train':
                joblib.dump({'model': best_model}, os.path.join(model_dir, f"{consumption_type}.pkl"))
        global_models[consumption_type] = best_model
    return global_models


def forecast_locally_with_global_models(df: pd.DataFrame, feature_columns: List[str], global_models: Dict[str, Any], model: ForecastModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs local forecasts per customer/pod using globally trained models. Aggregates results.
    """

    ufm_config = model.dataset.ufm_config
    forecast_dates = model.dataset.forecast_dates
    consumption_types = getattr(model.dataset, 'variable_ids', None) or model.config.consumption_types

    consumer_perf_data = CustomerPerformanceData(customer_id="GLOBAL", columns=consumption_types)
    model_perf_rows = []
    all_forecasts = []

    unique_customers = df['CustomerID'].unique()
    for customer_id in unique_customers:
        customer_df = df[df['CustomerID'] == customer_id]
        for pod_id in customer_df['PodID'].unique():
            pod_df = customer_df[customer_df['PodID'] == pod_id]
            pod_perf_rows = []

            for consumption_type in consumption_types:
                X_pod, y_pod = prepare_features_and_target(pod_df, feature_columns, consumption_type)
                if X_pod.empty or y_pod.empty:
                    continue

                # ✅ Add Series Validation Here
                valid, reason = is_series_valid(y_pod)
                if not valid:
                    logger.warning(f"⚠️ Skipping {consumption_type} for pod {pod_id}: {reason}.")
                    forecast = pd.Series([0] * len(forecast_dates), index=forecast_dates)
                    pod_perf_rows.append(_collect_metrics(
                        pod_id, customer_id, consumption_type, forecast
                    ))
                    continue

                model_for_type = global_models.get(consumption_type)
                y_pred = model_for_type.predict(X_pod)
                metrics, baseline = evaluate_predictions(y_pod, y_pred)

                # Future Forecast
                try:
                    future_X = pd.DataFrame(0, index=forecast_dates, columns=feature_columns)
                    future_X[['CustomerID_encoded', 'PodID_encoded']] = [
                        pod_df['CustomerID_encoded'].iloc[0],
                        pod_df['PodID_encoded'].iloc[0]
                    ]
                    y_future = model_for_type.predict(future_X)
                    future_forecast = pd.Series(y_future, index=forecast_dates)
                except Exception as e:
                    logger.error(f"⚠️ Forecast failed for {customer_id}-{pod_id}-{consumption_type}: {e}")
                    future_forecast = pd.Series([], dtype=float)

                # Collect performance row
                pod_perf_rows.append({
                    'pod_id': pod_id,
                    'customer_id': customer_id,
                    'consumption_type': consumption_type,
                    'forecast': future_forecast,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2'],
                    'RMSE_baseline': baseline['RMSE'],
                    'MAE_baseline': baseline['MAE'],
                    'R2_baseline': baseline['R2']
                })

            if pod_perf_rows:
                performance_df = pd.DataFrame(pod_perf_rows)
                pod_perf_data = PodIDPerformanceData(
                    pod_id=pod_id,
                    forecast_method_name=ufm_config.forecast_method_name,
                    customer_id=customer_id,
                    user_forecast_method_id=ufm_config.user_forecast_method_id,
                    performance_data_frame=performance_df
                )
                consumer_perf_data.pod_by_id_performance.append(pod_perf_data)
                model_perf_rows.append(_convert_to_model_performance_row(pod_perf_data, customer_id, pod_id, ufm_config))
                all_forecasts.append(_convert_forecast_map_to_df(pod_perf_data, customer_id, pod_id, ufm_config))

    # Aggregate performance and forecast outputs
    all_perf_df, final_perf_df, rf_performance_df, forecast_combined_df = _aggregate_forecast_outputs(
        consumer_perf_data, model_perf_rows, all_forecasts
    )
    logger.info("✅ Forecasting and aggregation complete.")
    return rf_performance_df, forecast_combined_df
