from datetime import datetime
import os
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline
from evaluation.performance import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, KFold)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from db.queries import ForecastConfig
from dml.dml import convert_column, create_lag_features, prepare_lag_features, create_month_and_year_columns, \
    prepare_features_and_target
from modeling.autoarima import evaluate_predictions
from modeling.utilities import load_hyperparameter_grid_rf, regressor_grid_for_pipeline, save_best_params_rf, \
    load_best_params_rf
from db.utilities import config

# Setup logger
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib._store_backends")
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")

cache_dir = os.path.join(os.getcwd(), 'pipeline_cache')
memory = joblib.Memory(location=cache_dir, verbose=0)

def most_frequent_params(param_list):
    """Return the most frequent parameter configuration."""
    from collections import Counter
    counter = Counter(tuple(sorted(p.items())) for p in param_list)
    most_common = counter.most_common(1)[0][0]
    return dict(most_common)

def build_rf_pipeline(random_state=42) -> Pipeline:
    """Construct a baseline pipeline for Random Forest + optional feature selection."""
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=RandomForestRegressor(random_state=random_state),
            step=0.1, cv=3, scoring='neg_mean_absolute_error'
        )),
        ('regressor', RandomForestRegressor(random_state=random_state))
    ])
    return pipeline


def nested_cv_rf(X, y, param_grid=None, scoring='neg_mean_absolute_error', perform_validation=False,
                 stored_hyperparams=None, interpret: bool = False):
    """
    Perform nested cross-validation using a pipeline that includes:
      - Imputation
      - Scaling
      - Feature selection (RFECV)
      - Random Forest Regression

    If perform_validation is True, the function conducts nested CV (outer and inner loops)
    to determine the best hyperparameters and aggregates the outer fold metrics.
    If False, it will skip the CV process and use stored hyperparameters (if provided) or
    default hyperparameters from the configuration, then perform a quick evaluation.

    does two levels of cross-validation (the “outer” for unbiased performance estimates, and the “inner” for hyperparameter selection).
    It then picks an overall best hyperparameter set, re-fits on the full data, and optionally returns aggregated outer CV performance.
    That’s more complex and yields a more robust estimate of how well your chosen hyperparameters generalize.

    Returns:
      - final_pipeline: A pipeline refitted on the entire dataset with the chosen hyperparameters.
      - scores: A dictionary with evaluation metrics (MAE, RMSE, R²).
      - selected_features: Features selected by the RFECV step.
    """
    if perform_validation:
        # Load the hyperparameter grid if not provided.
        if param_grid is None:
            param_grid = load_hyperparameter_grid_rf(config)
            param_grid = regressor_grid_for_pipeline(param_grid)

        # Create stratified bins for regression using quantile binning (converted to integer codes)
        strat_bins = pd.qcut(y, q=10, duplicates='drop').cat.codes
        outer_cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

        outer_scores_list = []
        best_params_list = []

        # Define the pipeline with caching enabled.
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('feature_selection', RFECV(
                estimator=RandomForestRegressor(random_state=42),
                step=0.1, cv=3, scoring=scoring
            )),
            ('regressor', RandomForestRegressor(random_state=42))
        ], memory=memory)

        # Outer CV loop for hyperparameter tuning
        for train_idx, test_idx in outer_cv.split(X, strat_bins):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Inner CV: using KFold (non-stratified) to reduce computational overhead
            inner_cv = KFold(n_splits=3, random_state=42, shuffle=True)

            grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, n_jobs=-1, scoring=scoring)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)

            fold_scores = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred)
            }
            outer_scores_list.append(fold_scores)
            best_params_list.append(grid_search.best_params_)
            logging.info(f"Fold scores: {fold_scores}")

        # Determine the most frequent hyperparameters across folds
        best_overall_params = most_frequent_params(best_params_list)
        logging.info(f"Most frequent hyperparameters: {best_overall_params}")

    else:
        # Skip nested CV and use stored or default hyperparameters
        if stored_hyperparams is not None:
            best_overall_params = stored_hyperparams
            logging.info("Using stored hyperparameters.")
        else:
            # Load the grid and choose default values (e.g., first option in each list)
            default_grid = load_hyperparameter_grid_rf(config)
            default_grid = regressor_grid_for_pipeline(default_grid)
            best_overall_params = {k: v[0] for k, v in default_grid.items() if k.startswith('regressor__')}
            logging.info(f"Using default hyperparameters: {best_overall_params}.")
        # When skipping validation, we don't have outer CV fold scores.
        outer_scores_list = None

    # Build the final pipeline using the chosen hyperparameters.
    # Remove duplicate random_state if present in best_overall_params.
    rf_params = {k.split('__')[1]: v for k, v in best_overall_params.items()
                 if k.startswith('regressor__') and k.split('__')[1] != 'random_state'}

    final_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selection', RFECV(
            estimator=RandomForestRegressor(random_state=42),
            step=0.1, cv=3, scoring=scoring
        )),
        ('regressor', RandomForestRegressor(**rf_params))
    ], memory=memory)

    final_pipeline.fit(X, y)

    # Extract the features selected by RFECV.
    selector = final_pipeline.named_steps['feature_selection']
    selected_features = X.columns[selector.get_support()]

    # Compute evaluation scores on the full dataset for the final pipeline.
    y_pred = final_pipeline.predict(X)
    eval_scores = {
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "R2": r2_score(y, y_pred)
    }

    # If nested CV was performed, aggregate outer fold scores; otherwise, use the evaluation scores.
    if perform_validation and outer_scores_list:
        # Average the fold metrics.
        agg_scores = {metric: np.mean([fold[metric] for fold in outer_scores_list])
                      for metric in outer_scores_list[0].keys()}
        scores = agg_scores
        logging.info(f"Aggregated outer CV scores: {scores}")
    else:
        scores = eval_scores
        logging.info(f"Evaluation scores on training data: {scores}")

    if interpret:
        # Generate diagnostics (e.g., feature importance, SHAP analysis)
        generate_diagnostics(final_pipeline, X, y, selected_features)

    return final_pipeline, scores, y_pred, selected_features


def random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    mode: str,
    model_dir: str,
    consumption_type: str,
    param_grid: Optional[Dict]=None,
    scoring: str='neg_mean_absolute_error'
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

        # Evaluate on the same data for now (since this is just sub_train or sub_train subset)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)

        return best_model, pd.Series(y_pred, index=X.index), metrics, baseline_metrics
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

        return pipeline, pd.Series(y_pred, index=X.index), metrics, baseline_metrics
    elif mode == 'test':
        # 1) Load best_params
        best_params = load_best_params_rf(model_dir, consumption_type)
        if best_params is None:
            logging.warning("No best params found in test mode. Using default.")
            best_params = {}

        # 2) Fit pipeline
        pipeline = build_rf_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X, y)

        # 3) Evaluate
        y_pred = pipeline.predict(X)
        metrics, baseline_metrics = evaluate_predictions(y, y_pred)

        return pipeline, pd.Series(y_pred, index=X.index), metrics, baseline_metrics

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
    param_grid: Optional[Dict] = None
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
        "models",
        ufm_config.forecast_method_name,
        f"customer_{customer_id}",
        f"pod_{pod_id}"
    )
    os.makedirs(model_dir, exist_ok=True)

    for consumption_type in consumption_types:
        # Step 1) Prepare the data
        X, y= prepare_features_and_target(df, feature_columns, consumption_type)
        if X.empty or y.empty:
            logging.info(f"No data for consumption type = {consumption_type}. Skipping")
            continue

        # Step 2) Train or Validate or Test
        model, forecast, metrics, baseline_metrics = random_forest(X, y,
            mode=mode,
            model_dir=model_dir,
            consumption_type=consumption_type,
            param_grid=param_grid)

        # Step 3) Aggregate results
        row = {
            'pod_id': pod_id,
            'customer_id': customer_id,
            'consumption_type': consumption_type,
            'forecast': forecast
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

def train_random_forest_for_single_customer( df: pd.DataFrame,
        customer_id: str,
        ufm_config: ForecastConfig,
        column: str = 'CustomerID',
        lag_features: List[str] = None,
        selected_columns: List[str] = None,
        consumption_types: List[str] = None,
        mode: str = None, param_grid: Optional[Dict] = None):
    if selected_columns is None:
        selected_columns = ["StandardConsumption","OffpeakConsumption" ,"PeakConsumption"]

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

    # Initialize a CustomerPerformanceData object to store performance per pod.
    consumer_performance_data = CustomerPerformanceData(customer_id=customer_id, columns=selected_columns)

    # Get unique PodIDs for this customer.
    unique_pod_ids: List[str] = list(customer_data["PodID"].unique())

    # Process each pod: forecast consumption for each consumption type.
    for pod_id in unique_pod_ids:
        # Sort pod data by reporting month.
        podel_df = customer_data[customer_data["PodID"] == pod_id].sort_values('ReportingMonth')
        podel_df = create_lag_features(podel_df, lag_features, lags=3)
        podel_df, feature_columns = prepare_lag_features(podel_df, lag_features)
        performance_data = train_random_forest_for_podel_id(podel_df,feature_columns,
    consumption_types,ufm_config, customer_id, pod_id, mode=mode, param_grid = param_grid)
        consumer_performance_data.pod_by_id_performance.append(performance_data)


    all_performance_df: pd.DataFrame = consumer_performance_data.get_pod_performance_data()

    pod_id_performance_data = consumer_performance_data.convert_pod_id_performance_data(all_performance_df)
    return pod_id_performance_data
