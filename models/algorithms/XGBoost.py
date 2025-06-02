from datetime import datetime
import os
from typing import List, Any, Dict, Optional

import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from db.queries import ForecastConfig
from data.dml import convert_column, create_lag_features, prepare_lag_features, create_month_and_year_columns, \
    prepare_features_and_target
from evaluation.performance import CustomerPerformanceData, PodIDPerformanceData, generate_diagnostics, \
    ModelPodPerformance, build_forecast_df, finalize_model_performance_df
from hyperparameters import get_model_hyperparameters, get_pipeline_config, get_cv_config, load_hyperparameter_grid
from models.algorithms.autoarima import evaluate_predictions
from models.algorithms.helper import _collect_metrics
from models.algorithms.utilities import *
from models.base import ForecastModel

# (Assuming that your utilities.py has been loaded as part of your environment)

# Setup logger
# Setup logger with basic configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib._store_backends")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Setup joblib cache directory
cache_dir = os.path.join(os.getcwd(), 'pipeline_cache')
memory = joblib.Memory(location=cache_dir, verbose=0)

# ------------------ Hyperparameter Grid for XGBoost ------------------

DEFAULT_XGB_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'random_state': [42]
}

def load_hyperparameter_grid_xgb(cfg=None): # Still need to configure hyperparameters in config
    """
    Load the hyperparameter grid for XGBoost from the configuration.
    If the configuration contains a key "xgboost", it will be used; otherwise,
    a default grid is returned.
    """
    if cfg is not None and "xgboost" in cfg:
        xgb_config = cfg.get("xgboost", {})
        grid = xgb_config.get("regression", DEFAULT_XGB_GRID)
        logging.info("Loaded XGBoost hyperparameter grid from config.")
        return grid
    else:
        return DEFAULT_XGB_GRID

# ------------------ Save/Load Hyperparameters for XGBoost ------------------
def load_best_params_xgb(model_dir: str, consumption_type: str):
    """Load the best hyperparameters for XGBoost from file if available."""
    params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")
    if os.path.exists(params_path):
        logging.info(f"Loading best XGBoost params from {params_path}")
        return joblib.load(params_path)
    else:
        logging.warning(f"No tuned params found for {consumption_type} at {params_path}. Using default.")
        return None

def save_best_params_xgb(best_params: dict, model_dir: str, consumption_type: str):
    """Save best hyperparameters for XGBoost to file."""
    params_path = os.path.join(model_dir, f"{consumption_type}_params.pkl")
    joblib.dump(best_params, params_path)
    logging.info(f"Saved best XGBoost params to {params_path}: {best_params}")

def most_frequent_params(param_list):
    """Return the most frequent parameter configuration from a list."""
    from collections import Counter
    counter = Counter(tuple(sorted(p.items())) for p in param_list)
    most_common = counter.most_common(1)[0][0]
    return dict(most_common)

# ------------------ Model Training Functions ------------------

def build_xgb_pipeline(random_state: int = 42) -> Pipeline:
    """
    Construct a baseline pipeline for XGBoost using external configuration.

    Returns:
        Pipeline: The constructed pipeline.
    """
    pipeline_config = get_pipeline_config("xgboost")
    imputer_strategy = pipeline_config.get("imputer_strategy", "mean")
    rfecv_config = pipeline_config.get("rfecv", {"step": 0.1, "cv": 3, "scoring": "neg_mean_absolute_error"})

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=imputer_strategy)),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=XGBRegressor(objective="reg:squarederror", random_state=random_state),
            step=rfecv_config.get("step", 0.1),
            cv=rfecv_config.get("cv", 3),
            scoring=rfecv_config.get("scoring", "neg_mean_absolute_error")
        )),
        ('regressor', XGBRegressor(objective="reg:squarederror", random_state=random_state))
    ], memory=memory)
    return pipeline


def create_inner_cv() -> KFold:
    """
    Create a KFold cross-validation object based on the configuration.

    Returns:
        KFold: The configured KFold cross-validator.f
    """
    cv_config = get_cv_config()
    return KFold(
        n_splits=cv_config.get("n_splits", 3),
        shuffle=cv_config.get("shuffle", True),
        random_state=cv_config.get("random_state", 42)
    )
def nested_cv_xgb(X, y, param_grid=None, scoring='neg_mean_absolute_error', perform_validation=False,
                  stored_hyperparams=None, interpret: bool = False):
    """
    Perform nested cross-validation using a pipeline that includes:
      - Imputation
      - Scaling
      - Feature selection (RFECV)
      - XGBoost Regression

    When perform_validation is True, the function tunes hyperparameters via nested CV
    and aggregates outer fold metrics. Otherwise, stored or default hyperparameters are used
    and the final pipeline is evaluated on the full dataset.

    Returns:
      - final_pipeline: Pipeline refitted on the entire dataset.
      - scores: A dictionary of evaluation metrics (MAE, RMSE, R2).
      - y_pred: Predictions on X.
      - selected_features: Features selected by RFECV.
    """
    if perform_validation:
        if param_grid is None:
            param_grid = load_hyperparameter_grid_xgb() # Still need to configure hyperparameters in config
            param_grid = regressor_grid_for_pipeline(param_grid)
        # Create stratified bins for regression (using quantile binning, converted to integer codes)
        strat_bins = pd.qcut(y, q=10, duplicates='drop').cat.codes
        outer_cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

        outer_scores_list = []
        best_params_list = []

        # Define pipeline with caching enabled.
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('feature_selection', RFECV(
                estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
                step=0.1, cv=3, scoring=scoring
            )),
            ('regressor', XGBRegressor(objective="reg:squarederror", random_state=42))
        ], memory=memory)

        # Outer CV loop for hyperparameter tuning
        for train_idx, test_idx in outer_cv.split(X, strat_bins):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            inner_cv = KFold(n_splits=3, random_state=42, shuffle=True)
            grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=-1, scoring=scoring)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred_fold = best_model.predict(X_test)
            fold_scores = {
                "mae": mean_absolute_error(y_test, y_pred_fold),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred_fold)),
                "r2": r2_score(y_test, y_pred_fold)
            }
            outer_scores_list.append(fold_scores)
            best_params_list.append(grid_search.best_params_)
            logging.info(f"Fold scores: {fold_scores}")

        best_overall_params = most_frequent_params(best_params_list)
        logging.info(f"Most frequent hyperparameters: {best_overall_params}")
    else:
        if stored_hyperparams is not None:
            best_overall_params = stored_hyperparams
            logging.info("Using stored hyperparameters for XGBoost.")
        else:
            default_grid = load_hyperparameter_grid_xgb() # Still need to configure hyperparameters in config
            default_grid = regressor_grid_for_pipeline(default_grid)
            best_overall_params = {k: v[0] for k, v in default_grid.items() if k.startswith('regressor__')}
            logging.info(f"Using default hyperparameters for XGBoost: {best_overall_params}.")
        outer_scores_list = None

    # Build the final pipeline with the chosen hyperparameters.
    # Filter out duplicate 'random_state' parameter.
    xgb_params = {k.split('__')[1]: v for k, v in best_overall_params.items()
                  if k.startswith('regressor__') and k.split('__')[1] != 'random_state'}

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=XGBRegressor(objective="reg:squarederror", random_state=42),
            step=0.1, cv=3, scoring=scoring
        )),
        ('regressor', XGBRegressor(objective="reg:squarederror", random_state=42, **xgb_params))
    ], memory=memory)

    final_pipeline.fit(X, y)
    selector = final_pipeline.named_steps['feature_selection']
    selected_features = X.columns[selector.get_support()]

    y_pred = final_pipeline.predict(X)
    eval_scores = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred)
    }

    if perform_validation and outer_scores_list:
        agg_scores = {metric: np.mean([fold[metric] for fold in outer_scores_list])
                      for metric in outer_scores_list[0].keys()}
        scores = agg_scores
        logging.info(f"Aggregated outer CV scores for XGBoost: {scores}")
    else:
        scores = eval_scores
        logging.info(f"Evaluation scores on training dataset for XGBoost: {scores}")

    if interpret:
        # Optionally generate diagnostics (e.g., feature importance, SHAP analysis)
        generate_diagnostics(final_pipeline, X, y, selected_features)

    return final_pipeline, scores, y_pred, selected_features


# ------------------ Unified XGBoost Forecast Function ------------------
def xgboost_forecast(X: pd.DataFrame, y: pd.Series, mode: str, model_dir: str, consumption_type: str,
                     param_grid: dict = None, scoring: str = 'neg_mean_absolute_error',
                    forecast_dates: Optional[pd.DatetimeIndex] = None,
                    feature_columns: Optional[List[str]] = None
                     ):
    """
    Depending on the mode, perform:
      - 'validation': Run grid search using cross-validation to tune hyperparameters and save the best.
      - 'train': Load tuned hyperparameters (or use defaults), train final pipeline on full dataset, evaluate and save model.
      - 'test': Load tuned hyperparameters, train on full dataset and return test evaluation.

    Returns:
      - final_pipeline: Trained pipeline.
      - forecast: Predictions on training dataset (or test if applicable).
      - metrics: Evaluation metrics dictionary.
      - baseline_metrics: Baseline metrics (here same as metrics for simplicity).
    """
    if X.empty or y.empty:
        logging.warning(f"No dataset for consumption_type={consumption_type}. Skipping.")
        return None, pd.Series([], dtype=float), {}, {}

    os.makedirs(model_dir, exist_ok=True)

    if mode == 'validation':
        if param_grid is None:
            raw_grid = load_hyperparameter_grid_xgb() # Still need to configure hyperparameters in config
            param_grid = regressor_grid_for_pipeline(raw_grid)

        # Use KFold CV for tuning (similar to random_forest.py)
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        pipeline = build_xgb_pipeline()
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        save_best_params_xgb(best_params, model_dir, consumption_type)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)
        return best_model, pd.Series(y_pred, index=X.index), metrics, baseline_metrics, pd.Series(y_pred, index=X.index)

    elif mode == 'train':
        best_params = load_best_params_xgb(model_dir, consumption_type)
        if best_params is None:
            # Use default grid values (first value from each list)
            default_grid = load_hyperparameter_grid_xgb() # Still need to configure hyperparameters in config
            default_grid = regressor_grid_for_pipeline(default_grid)
            best_params = {k: v[0] for k, v in default_grid.items() if k.startswith('regressor__')}
            logging.info(f"Using default hyperparameters for XGBoost: {best_params}.")

        pipeline = build_xgb_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X, y) # Demonstrate that is sub_train + fake_train

        y_pred = pipeline.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)

        # Save final model with metadata.
        final_model_path = os.path.join(model_dir, f"{consumption_type}.pkl")
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'hyperparams': best_params
        }
        joblib.dump({'model': pipeline, 'metadata': metadata}, final_model_path)
        logging.info(f"Saved final XGBoost model to {final_model_path}")

        # 3) Future forecast
        try:
            future_X = make_future_features(forecast_dates, feature_columns)
            y_future = pipeline.predict(future_X)
            future_forecast = pd.Series(y_future, index=forecast_dates)
        except Exception as e:
            logging.error(f"Failed to build future features or predict: {e}")
            future_forecast = pd.Series([], dtype=float)

        return pipeline, pd.Series(y_pred, index=X.index), metrics, baseline_metrics,  future_forecast


    elif mode == 'test':
        best_params = {}
        # 1) Try loading saved model
        model_path = os.path.join(model_dir, f"{consumption_type}.pkl")

        if os.path.exists(model_path):
            logging.info(f"🔁 Loading saved RF model for {consumption_type} from {model_path}")
            saved = joblib.load(model_path)
            pipeline = saved.get('model', saved)
        else:
            logging.warning("No saved model found. Training on full data for test.")
            best_params = load_best_params_xgb(model_dir, consumption_type)
            pipeline = build_xgb_pipeline()
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
# ------------------ Per-Pod Training ------------------
def train_forecast_per_pod_xgb(podel_df: pd.DataFrame, customer_id: str, pod_id: str, feature_columns: list,
                               consumption_types: list, ufm_config: ForecastConfig, mode: str = 'train',
                               param_grid: dict = None,
                               forecast_dates: Optional[pd.DatetimeIndex] = None):
    """
    Train XGBoost models for each consumption type for a given pod.
    This function:
      1) Prepares X and y based on features.
      2) Calls xgboost_forecast() for training/validation/testing.
      3) Aggregates performance and forecasts into a summary DataFrame.
    """
    logging.info("Training XGBoost for POD: %s", pod_id)
    metric_keys = {'RMSE', 'MAE', 'R2'}
    data = []

    # Directory structure for saving/loading model_configuration.
    model_dir = os.path.join("model_configuration", "XGBoost",
                             f"customer_{customer_id}", f"pod_{pod_id}")
    os.makedirs(model_dir, exist_ok=True)

    for consumption_type in consumption_types:
        try:
            logging.info(f"Customer: {customer_id}, POD: {pod_id}, Consumption Type: {consumption_type}")
            # Skip if target column is completely NaN
            if podel_df[consumption_type].isnull().all():
                continue

            X = podel_df[feature_columns]
            y = podel_df[consumption_type]

            model, in_sample_forecast, metrics, baseline_metrics, future_forecast  = xgboost_forecast(
                X, y, mode=mode, model_dir=model_dir, consumption_type=consumption_type, param_grid=param_grid,
                forecast_dates=forecast_dates,
                feature_columns=feature_columns
            )
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

            data.append(row)

            # Save the final model with metadata.
            model_file = os.path.join(model_dir, f"{consumption_type}.pkl")
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "hyperparameters": model.named_steps['regressor'].get_params()
            }
            joblib.dump({"model": model, "metadata": metadata}, model_file)
            logging.info(f"Saved XGBoost model to {model_file} with metadata: {metadata}")
        except Exception as e:
            logging.exception(f"Error during training for POD {pod_id}, Consumption Type {consumption_type}: {e}")

    performance_data_frame = pd.DataFrame(data)
    pod_id_performance_data = PodIDPerformanceData(pod_id, "XGBoost", customer_id,
                                                   ufm_config.user_forecast_method_id, performance_data_frame)
    return pod_id_performance_data


def train_xgboost_for_single_customer(model: ForecastModel) -> pd.DataFrame:
    """
    Train XGBoost models for a single customer across multiple pods using a unified API.

    This function performs the following steps:
      1. Extracts required parameters from the ForecastModel instance, including:
         - The processed DataFrame.
         - The customer identifier.
         - Forecast configuration (which includes start/end dates and hyperparameter string).
         - Selected consumption columns, consumption types, and operational mode.
      2. Sets default values for selected_columns, consumption_types, and lag_features if not provided.
      3. Preprocesses the DataFrame by creating month/year columns and converting customer IDs.
      4. Filters the dataset for the target customer and sorts by PodID.
      5. Iterates over each unique PodID for the customer:
            - Creates lag features and prepares the feature set.
            - Calls the per-pod training function for XGBoost (train_forecast_per_pod_xgb).
            - Aggregates performance and forecast results into a performance container.
      6. Combines and returns all per-pod performance data as a long-format DataFrame.

    Args:
        model (ForecastModel): An instance that encapsulates the ForecastDataset and configuration.

    Returns:
        pd.DataFrame: Aggregated forecast performance metrics and predictions.

    Raises:
        ValueError: If the dataset is empty or required fields are missing.
    """
    # Extract required attributes from the model object.
    df: pd.DataFrame = model.dataset.processed_df
    customer_id: str = "8460296087"
    ufm_config = model.dataset.ufm_config
    mode: str = model.config.mode



    # Set defaults if not provided.
    selected_columns: List[str] = model.config.selected_columns if model.config.selected_columns is not None else \
        ["StandardConsumption", "OffpeakConsumption", "PeakConsumption"]
    consumption_types: List[str] = model.config.consumption_types if model.config.consumption_types is not None else [
        "PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
        "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"
    ]
    lag_features: List[str] = model.config.selected_columns if model.config.selected_columns is not None else ['StandardConsumption']

    logger.info(f"🚀 [XGBoost] Starting training for customer {customer_id} in mode '{mode}'.")

    # Extract or default hyperparameters using our centralized helper.

    # param_grid: Any = get_model_hyperparameters("xgboost", ufm_config.model_parameters)
    param_grid: Any = get_model_hyperparameters("xgboost", ufm_config.model_parameters)
    param_grid = regressor_grid_for_pipeline(
        param_grid,
        prefix="regressor__",
        param_names=XGB_PARAM_NAMES  # <-- override RF_PARAM_NAMES
    )
    logger.info(f"💡 [XGBoost] Hyperparameters: {param_grid}")

    # Preprocess the DataFrame:
    df = create_month_and_year_columns(df)
    customer_data: pd.DataFrame = df[df['CustomerID'] == customer_id].sort_values('PodID')
    customer_data = convert_column(customer_data)

    # Initialize a performance container.
    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)

    unique_pod_ids: List[str] = list(customer_data["PodID"].unique())
    logger.info(f"📊 [XGBoost] Found {len(unique_pod_ids)} pod(s) for customer {customer_id}.")

    forecast_map: Dict[str, pd.Series] = {}
    all_forecasts = []
    xgboost_rows: List[ModelPodPerformance] = []
    for pod_id in unique_pod_ids:
        pod_df: pd.DataFrame = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        pod_df = create_lag_features(pod_df, lag_features, lags=3)
        pod_df, feature_columns = prepare_lag_features(pod_df, lag_features)

        performance_data = train_forecast_per_pod_xgb(
            pod_df, customer_id, pod_id, feature_columns, consumption_types, ufm_config, mode=mode,
            param_grid=param_grid, forecast_dates=model.dataset.forecast_dates
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
        xgboost_rows.append(mpp)

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
        logger.info(f"✅ [XGBoost] Processed pod {pod_id} for customer {customer_id}.")

    # Aggregate performance results.
    all_performance_df: pd.DataFrame = consumer_performance_data.get_pod_performance_data()
    xgboost_rows_performance_df = pd.DataFrame([m.to_row() for m in xgboost_rows])
    final_performance_df = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)
    forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
    pod_id_performance_data: pd.DataFrame = consumer_performance_data.convert_pod_id_performance_data(
        all_performance_df)
    logger.info("✅ [XGBoost] Forecast aggregation complete.")
    return pod_id_performance_data

def train_xgboost_globally_forecast_locally_with_aggregation(model: ForecastModel) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates global XGBoost training and local forecasting. Returns performance and forecast outputs.
    """
    df, feature_columns = prepare_global_training_data(model)
    global_models = train_global_xgboost_models(df, feature_columns, model)
    xgb_performance_df, forecast_combined_df = forecast_locally_with_global_xgboost_models(df, feature_columns, global_models, model)
    return xgb_performance_df, forecast_combined_df


def prepare_global_training_data(model: ForecastModel) -> Tuple[pd.DataFrame, List[str]]:
    df = model.dataset.processed_df.copy()

    # Drop constant or all-zero consumption columns
    usable = [col for col in model.config.selected_columns if df[col].nunique() > 1 and not (df[col] == 0).all()]
    df = df.drop(columns=[col for col in model.config.selected_columns if col not in usable])

    # Add Month and Year
    df = create_month_and_year_columns(df)

    # Encode IDs for grouping only; do not use in feature_columns
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['PodID'] = df['PodID'].astype(str)

    # Create lag features (but keep original ID fields for grouping later)
    df, lag_features = prepare_lag_features(df, lag_columns=usable, base_features=["Month", "Year"])

    # Final feature columns (exclude identifiers)
    feature_columns = [f for f in lag_features if f not in ['CustomerID', 'PodID']]
    return df, feature_columns


def train_global_xgboost_models(df: pd.DataFrame, feature_columns: List[str], model: ForecastModel) -> Dict[str, Any]:
    global_models = {}
    mode = model.config.mode
    ufm_config = model.dataset.ufm_config
    consumption_types = [ct for ct in model.config.consumption_types if ct in df.columns]

    for consumption_type in consumption_types:
        X, y = prepare_features_and_target(df, feature_columns, consumption_type)

        # ✅ Add Global Series Validation Here
        valid, reason = is_series_valid(y)
        if not valid:
            logger.warning(f"⚠️ Skipping global model training for {consumption_type}: {reason}.")
            continue

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])

        param_grid = get_model_hyperparameters('xgboost', ufm_config.model_parameters)
        param_grid = regressor_grid_for_pipeline(param_grid, prefix='regressor__', param_names=XGB_PARAM_NAMES)
        model_dir = os.path.join('model_configuration', ufm_config.forecast_method_name, 'global')
        os.makedirs(model_dir, exist_ok=True)

        if mode == 'validation':
            search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
            search.fit(X, y)
            best_model = search.best_estimator_
            save_best_params_xgb(search.best_params_, model_dir, consumption_type)
        else:
            best_params = load_best_params_xgb(model_dir, consumption_type) or {k: v[0] for k, v in param_grid.items()}
            pipeline.set_params(**best_params)
            pipeline.fit(X, y)
            best_model = pipeline
            # if mode == 'train':
            #     joblib.dump({'model': best_model}, os.path.join(model_dir, f"{consumption_type}.pkl"))

        global_models[consumption_type] = best_model

    return global_models


def is_series_valid(series: pd.Series, min_length: int = 30) -> Tuple[bool, str]:
    if series.isnull().all():
        return False, "All NaN values"
    if len(series) < min_length:
        return False, "Too few observations"
    return True, "Series valid"

def forecast_locally_with_global_xgboost_models(
    df: pd.DataFrame,
    feature_columns: List[str],
    global_models: Dict[str, Any],
    model: ForecastModel
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from models.algorithms.helper import _convert_to_model_performance_row, _convert_forecast_map_to_df, _aggregate_forecast_outputs
    ufm_config = model.dataset.ufm_config
    forecast_dates = model.dataset.forecast_dates
    trained_types = global_models.keys()

    consumer_perf_data = CustomerPerformanceData("GLOBAL", columns=list(trained_types))
    model_perf_rows = []
    all_forecasts = []

    for cust_id in df['CustomerID'].unique():
        cust_df = df[df['CustomerID'] == cust_id]
        for pod_id in cust_df['PodID'].unique():
            pod_df = cust_df[cust_df['PodID'] == pod_id]
            pod_perf_rows = []

            for ct in trained_types:
                if ct not in pod_df.columns: continue
                X_pod, y_pod = prepare_features_and_target(pod_df, feature_columns, ct)
                if X_pod.empty or y_pod.empty: continue

                model_for_ct = global_models[ct]
                y_pred = model_for_ct.predict(X_pod)
                metrics, baseline = evaluate_predictions(y_pod, y_pred)

                future_X = pd.DataFrame(0, index=forecast_dates, columns=feature_columns)
                for col in ['Month', 'Year']:
                    if col in feature_columns:
                        future_X[col] = [d.month if col == 'Month' else d.year for d in forecast_dates]

                future_forecast = model_for_ct.predict(future_X)
                future_series = pd.Series(future_forecast, index=forecast_dates)

                pod_perf_rows.append({
                    'pod_id': pod_id,
                    'customer_id': cust_id,
                    'consumption_type': ct,
                    'forecast': future_series,
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'R2': metrics['R2'],
                    'RMSE_baseline': baseline['RMSE'],
                    'MAE_baseline': baseline['MAE'],
                    'R2_baseline': baseline['R2']
                })

            if pod_perf_rows:
                df_perf = pd.DataFrame(pod_perf_rows)
                perf_data = PodIDPerformanceData(pod_id, ufm_config.forecast_method_name, cust_id, ufm_config.user_forecast_method_id, df_perf)
                consumer_perf_data.pod_by_id_performance.append(perf_data)
                model_perf_rows.append(_convert_to_model_performance_row(perf_data, cust_id, pod_id, ufm_config))
                all_forecasts.append(_convert_forecast_map_to_df(perf_data, cust_id, pod_id, ufm_config))

    if not model_perf_rows or not all_forecasts:
        logger.warning("No forecasts were generated due to invalid series.")
        return pd.DataFrame(), pd.DataFrame()
    _, _, xgb_perf_df, forecast_combined_df = _aggregate_forecast_outputs(
        consumer_perf_data, model_perf_rows, all_forecasts)
    forecast_combined_df = fill_missing_consumption_columns_wide(forecast_combined_df, model)
    return xgb_perf_df, forecast_combined_df

def fill_missing_consumption_columns_wide(df, model):
    desired_order = ["PodID", "UserForecastMethodID", "CustomerID", "ReportingMonth",
        "OffPeakConsumption", "StandardConsumption", "PeakConsumption",
        "Block1Consumption", "Block2Consumption", "Block3Consumption",
        "Block4Consumption", "NonTOUConsumption"
    ]

    # Fill missing consumption columns with zeros
    for col in model.config.consumption_types:
        if col not in df.columns:
            df[col] = 0.0

    # Reorder columns (only include those that exist in the DataFrame)
    reordered_cols = [col for col in desired_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in reordered_cols]
    return df[reordered_cols + other_cols]

