from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_time_index(df: pd.DataFrame, date_col: str) -> pd.Series:
    """
    Compute TimeIndex: number of months since the earliest date in date_col.
    """
    earliest = df[date_col].min()
    return ((df[date_col] - earliest).dt.days // 30).astype(int)


def frequency_encode(series: pd.Series) -> pd.Series:
    """
    Frequency‐encode a categorical Series: (count / total_count).
    Unseen categories map to 0.0.
    """
    freqs = series.value_counts() / len(series)
    return series.map(freqs).fillna(0.0)


def target_encode(
    df: pd.DataFrame, col: str, target: str, smoothing: float = 1.0
) -> pd.Series:
    """
    Target‐encode with smoothing:
      enc = (mean_cat * count_cat + global_mean * smoothing) / (count_cat + smoothing)
    """
    agg = df.groupby(col)[target].agg(["mean", "count"])
    global_mean = df[target].mean()
    agg["enc"] = ((agg["mean"] * agg["count"] + global_mean * smoothing)
                  / (agg["count"] + smoothing))
    return df[col].map(agg["enc"]).fillna(global_mean)


def train_test_split_time(
    df: pd.DataFrame, date_col: str, test_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split: last `test_fraction` of rows by date.
    """
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n_total = len(df_sorted)
    test_size = max(int(n_total * test_fraction), 1)
    train_end = n_total - test_size
    return df_sorted.iloc[:train_end], df_sorted.iloc[train_end:]


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Return MSE, MAE, R2 as a dict.
    """
    return {
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }
