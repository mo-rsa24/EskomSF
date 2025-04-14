from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List
import shap
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import logging
# Setup logger
logging.basicConfig(level=logging.INFO)
@dataclass
class PerformanceData:
    forecast_method_name: str
    customer_id: str
    pod_id: str
    user_forecast_method_id: int
    metrics: Dict[str, float ] = field(default_factory=dict)

    def log_metric(self, metric: str, value: float, alternative: str = None, consumption_type: str = None):
        if alternative is None:
            key = f'{metric}_{consumption_type}' if consumption_type is not None else f'{metric}'
            self.metrics[key] = value
        else:
            key = f'{metric}_{consumption_type}_{alternative}' if consumption_type is not None else f'{metric}'
            self.metrics[key] = value


@dataclass
class PodIDPerformanceData:
    pod_id: str
    forecast_method_name: str
    customer_id: str
    user_forecast_method_id: int
    performance_data_frame: pd.DataFrame

@dataclass
class CustomerPerformanceData:
    customer_id: str
    columns: List[str]
    pod_by_id_performance: List[PodIDPerformanceData] = field(default_factory=list)

    def get_pod_performance_data(self) -> pd.DataFrame:
        dataframes = [
            pod_data.performance_data_frame
            for pod_data in self.pod_by_id_performance
            if hasattr(pod_data, 'performance_data_frame')
        ]
        # Combine all DataFrames into one.
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def convert_pod_id_performance_data(self, df: pd.DataFrame, consumption_filter: List[str] = None) -> pd.DataFrame:
        """
        Convert a wide-format performance DataFrame into a long-format DataFrame, filtering by specified consumption types.

        The input DataFrame is expected to contain at least the following columns:
            - 'pod_id'
            - 'customer_id'
            - 'consumption_type'
            - Metric columns: 'RMSE', 'MAE', 'R2', 'forecast'

        The function performs the following steps:
            1. Creates a pivot table using 'pod_id' and 'customer_id' as the index and the
               'consumption_type' values as columns. Metric values are aggregated using the first occurrence.
            2. Stacks the consumption type level to convert the DataFrame from wide to long format.
            3. Filters the long-format DataFrame to retain only rows where the consumption type is in the
               provided `consumption_filter`.

        Parameters:
            df (pd.DataFrame): The input DataFrame in long format with required columns.
            consumption_filter (List[str], optional): A list of allowed consumption types to retain.
                Defaults to ['PeakConsumption', 'StandardConsumption', 'OffPeakConsumption'].

        Returns:
            pd.DataFrame: A long-format DataFrame where each row corresponds to a unique combination
                          of (pod_id, customer_id, consumption_type) and contains the corresponding metric values.
        """
        if consumption_filter is None:
            consumption_filter = ['PeakConsumption', 'StandardConsumption', 'OffPeakConsumption']

        # Pivot the DataFrame: Each (pod_id, customer_id) pair becomes a unique index,
        # with consumption types as columns for each metric.
        pod_df = df.pivot_table(
            index=['pod_id', 'customer_id'],
            columns='consumption_type',
            values=['RMSE', 'MAE', 'R2', 'forecast'],
            aggfunc='first'  # In case of duplicates, take the first occurrence.
        )

        # Stack the consumption type level to convert the wide format into a long format.
        pod_pf = pod_df.stack(level=1).reset_index()

        # Filter rows to include only the allowed consumption types.
        pod_filtered = pod_pf[pod_pf['consumption_type'].isin(consumption_filter)]

        return pod_filtered

def get_performance_data(forecast_method_name: str, customer_id: str, pod_id: str, user_forecast_method_id: int) -> PerformanceData:
    return PerformanceData(forecast_method_name, customer_id, pod_id, user_forecast_method_id)


def most_frequent_params(params_list):
    """Select most common params from CV results"""
    param_counts = {}
    for params in params_list:
        for k, v in params.items():
            param_counts.setdefault(k, {}).update({v: param_counts.get(k, {}).get(v, 0) + 1})
    return {k: max(v.items(), key=lambda x: x[1])[0] for k, v in param_counts.items()}

def generate_diagnostics(model, X, y, feature_names, verbose: bool = False):
    """Comprehensive model interpretation"""

    # Gini Importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'gini': model.feature_importances_,
        'permutation': permutation_importance(model, X, y, n_repeats=10).importances_mean
    }).sort_values(by='gini', ascending=False)
    print("Feature Importance:", importance_df)

    # SHAP Analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)

    # Partial Dependence
    for feature in feature_names[:3]:
        PartialDependenceDisplay.from_estimator(
            model, X, [feature],
            kind='both',
            subsample=1000,
            n_jobs=-1
        )
    if verbose:
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')
        plt.plot(train_sizes, -np.mean(train_scores, axis=1), label='Training error')
        plt.plot(train_sizes, -np.mean(test_scores, axis=1), label='Validation error')
        plt.xlabel('Training size')
        plt.ylabel('MSE')
        plt.legend()
        plt.title('Learning Curve')
        plt.show()

        # OOB Error Tracking
        oob_scores = [estimator.oob_score_ for estimator in model.estimators_]
        plt.plot(oob_scores)
        plt.title('OOB Error During Training')
        plt.xlabel('Number of Trees')
        plt.ylabel('OOB Score')

# Performance Reporting
def report_performance(scores):
    """
    Report and log the average performance metrics over outer folds.

    Returns:
      - dict: Aggregated performance metrics (MAE, RMSE, R², MAPE).
    """
    mae_avg = np.mean([s["mae"] for s in scores])
    rmse_avg = np.mean([s["rmse"] for s in scores])
    r2_avg = np.mean([s["r2"] for s in scores])

    performance = {
        "MAE": mae_avg,
        "RMSE": rmse_avg,
        "R2": r2_avg
    }
    logging.info(f"Final Model Performance: {performance}")
    return performance