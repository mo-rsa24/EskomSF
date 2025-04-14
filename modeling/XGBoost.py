from datetime import datetime
import os
import joblib
import logging
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from db.queries import ForecastConfig
from dml.dml import convert_column, create_lag_features, prepare_lag_features, create_month_and_year_columns
from evaluation.performance import CustomerPerformanceData, PodIDPerformanceData, generate_diagnostics
from modeling.autoarima import evaluate_predictions
from modeling.utilities import regressor_grid_for_pipeline
from db.utilities import config
# (Assuming that your utilities.py has been loaded as part of your environment)

# Setup logger
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib._store_backends")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")

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

def load_hyperparameter_grid_xgb(cfg=None):
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

def build_xgb_pipeline(random_state=42) -> Pipeline:
    """
    Construct a baseline pipeline for XGBoost including:
      - Simple Imputation
      - Scaling
      - Feature Selection using RFECV
      - XGBoost Regressor
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=XGBRegressor(objective="reg:squarederror", random_state=random_state),
            step=0.1, cv=3, scoring='neg_mean_absolute_error'
        )),
        ('regressor', XGBRegressor(objective="reg:squarederror", random_state=random_state))
    ], memory=memory)
    return pipeline

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
            param_grid = load_hyperparameter_grid_xgb(config)
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
            default_grid = load_hyperparameter_grid_xgb(config)
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
        logging.info(f"Evaluation scores on training data for XGBoost: {scores}")

    if interpret:
        # Optionally generate diagnostics (e.g., feature importance, SHAP analysis)
        generate_diagnostics(final_pipeline, X, y, selected_features)

    return final_pipeline, scores, y_pred, selected_features


# ------------------ Unified XGBoost Forecast Function ------------------
def xgboost_forecast(X: pd.DataFrame, y: pd.Series, mode: str, model_dir: str, consumption_type: str,
                     param_grid: dict = None, scoring: str = 'neg_mean_absolute_error'):
    """
    Depending on the mode, perform:
      - 'validation': Run grid search using cross-validation to tune hyperparameters and save the best.
      - 'train': Load tuned hyperparameters (or use defaults), train final pipeline on full data, evaluate and save model.
      - 'test': Load tuned hyperparameters, train on full data and return test evaluation.

    Returns:
      - final_pipeline: Trained pipeline.
      - forecast: Predictions on training data (or test if applicable).
      - metrics: Evaluation metrics dictionary.
      - baseline_metrics: Baseline metrics (here same as metrics for simplicity).
    """
    if X.empty or y.empty:
        logging.warning(f"No data for consumption_type={consumption_type}. Skipping.")
        return None, pd.Series([], dtype=float), {}, {}

    os.makedirs(model_dir, exist_ok=True)

    if mode == 'validation':
        if param_grid is None:
            raw_grid = load_hyperparameter_grid_xgb(config)
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
        return best_model, pd.Series(y_pred, index=X.index), metrics, baseline_metrics

    elif mode == 'train':
        best_params = load_best_params_xgb(model_dir, consumption_type)
        if best_params is None:
            # Use default grid values (first value from each list)
            default_grid = load_hyperparameter_grid_xgb(config)
            default_grid = regressor_grid_for_pipeline(default_grid)
            best_params = {k: v[0] for k, v in default_grid.items() if k.startswith('regressor__')}
            logging.info(f"Using default hyperparameters for XGBoost: {best_params}.")

        pipeline = build_xgb_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X, y)
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

        return pipeline, pd.Series(y_pred, index=X.index), metrics, baseline_metrics

    elif mode == 'test':
        best_params = load_best_params_xgb(model_dir, consumption_type)
        if best_params is None:
            logging.warning("No best params found in test mode. Using default.")
            best_params = {}

        pipeline = build_xgb_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)
        return pipeline, pd.Series(y_pred, index=X.index), metrics, baseline_metrics

    else:
        raise ValueError("mode must be one of ['validation', 'train', 'test']")


# ------------------ Per-Pod Training ------------------
def train_forecast_per_pod_xgb(podel_df: pd.DataFrame, customer_id: str, pod_id: str, feature_columns: list,
                               consumption_types: list, ufm_config: ForecastConfig, mode: str = 'train',
                               param_grid: dict = None):
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

    # Directory structure for saving/loading models.
    model_dir = os.path.join("models", ufm_config.forecast_method_name,
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

            model, y_pred, scores, baseline_metrics = xgboost_forecast(
                X, y, mode=mode, model_dir=model_dir, consumption_type=consumption_type, param_grid=param_grid
            )
            row = {
                'pod_id': pod_id,
                'customer_id': customer_id,
                'consumption_type': consumption_type,
                'forecast': y_pred
            }
            for key in metric_keys:
                row[key] = scores.get(key, None)
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
    pod_id_performance_data = PodIDPerformanceData(pod_id, ufm_config.forecast_method_name, customer_id,
                                                   ufm_config.user_forecast_method_id, performance_data_frame)
    return pod_id_performance_data


def train_xgboost_for_single_customer(df: pd.DataFrame, customer_id: str, ufm_config: ForecastConfig,
                                      column: str = 'CustomerID', lag_features: list = None,
                                      selected_columns: list = None, consumption_types: list = None,
                                      mode: str = 'train', param_grid: dict = None):
    """
    Train XGBoost models for a single customer across multiple pods.
    Preprocess the data, create lag features, and call the per-pod training function.
    """
    if selected_columns is None:
        selected_columns = ["StandardConsumption", "OffpeakConsumption", "PeakConsumption"]
    if consumption_types is None:
        consumption_types = [
            "PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
            "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"
        ]
    if lag_features is None:
        lag_features = ['StandardConsumption']

    df = create_month_and_year_columns(df)
    customer_data = df[df[column] == customer_id].sort_values('PodID')
    customer_data = convert_column(customer_data)

    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)
    unique_pod_ids = list(customer_data["PodID"].unique())

    for pod_id in unique_pod_ids:
        podel_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        podel_df = create_lag_features(podel_df, lag_features, lags=3)
        podel_df, feature_columns = prepare_lag_features(podel_df, lag_features)
        performance_data = train_forecast_per_pod_xgb(
            podel_df, customer_id, pod_id, feature_columns, consumption_types, ufm_config, mode=mode,
            param_grid=param_grid
        )
        consumer_performance_data.pod_by_id_performance.append(performance_data)

    all_performance_df = consumer_performance_data.get_pod_performance_data()
    pod_id_performance_data = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)
    return pod_id_performance_data
