from typing import List, Tuple

import numpy as np
import pandas as pd

from models.algorithms.tree_algorithms.random_forest.utils import compute_time_index


class FeatureEngineer:
    """
    Generates time‐series features (lags, rolling, seasonal, time index) grouped by keys.
    """

    def __init__(
        self,
        group_cols: List[str],
        date_col: str,
        consumption_cols: List[str],
        lag_list: List[int],
        rolling_windows: List[int],
        seasonal: bool = True,
        drop_na: bool = True
    ):
        self.group_cols = group_cols
        self.date_col = date_col
        self.consumption_cols = consumption_cols
        self.lag_list = lag_list
        self.rolling_windows = rolling_windows
        self.seasonal = seasonal
        self.drop_na = drop_na

    def generate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Returns:
          - df_feat: DataFrame with features + original columns
          - feature_cols: List[str] of input columns for the model
        """
        df.reset_index(inplace=True)
        df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'])
        df["Year"] = df["ReportingMonth"].dt.year
        df_sorted = df.copy().sort_values(self.group_cols + [self.date_col])

        # 1) Time index
        df_sorted["TimeIndex"] = compute_time_index(df_sorted, self.date_col)

        # 2) Seasonal features
        if self.seasonal:
            df_sorted["month"] = df_sorted[self.date_col].dt.month
            df_sorted["Month_sin"] = np.sin(2 * np.pi * df_sorted["month"] / 12)
            df_sorted["Month_cos"] = np.cos(2 * np.pi * df_sorted["month"] / 12)
        else:
            df_sorted["Month_sin"] = 0.0
            df_sorted["Month_cos"] = 0.0

        # 3) Lag features
        for cons in self.consumption_cols:
            for lag in self.lag_list:
                df_sorted[f"{cons}_lag{lag}"] = (
                    df_sorted
                    .groupby(self.group_cols)[cons]
                    .shift(lag)
                )

        # 4) Rolling stats
        for cons in self.consumption_cols:
            for window in self.rolling_windows:
                rm_col = f"{cons}_roll{window}_mean"
                rs_col = f"{cons}_roll{window}_std"
                df_sorted[rm_col] = (
                    df_sorted
                    .groupby(self.group_cols)[cons]
                    .shift(1)
                    .rolling(window=window)
                    .mean()
                )
                df_sorted[rs_col] = (
                    df_sorted
                    .groupby(self.group_cols)[cons]
                    .shift(1)
                    .rolling(window=window)
                    .std()
                    .fillna(0.0)
                )

        # 5) Drop rows with NaNs in lag columns
        lag_cols = [f"{cons}_lag{lag}"
                    for cons in self.consumption_cols
                    for lag in self.lag_list]
        if self.drop_na:
            df_sorted = df_sorted.dropna(subset=lag_cols).reset_index(drop=True)

        # 6) Build feature_cols (exclude group_cols, date_col, consumption_cols)
        feature_cols: List[str] = []
        for col in df_sorted.columns:
            if col in self.group_cols or col == self.date_col:
                continue
            if col in self.consumption_cols:
                continue
            feature_cols.append(col)

        # 7) Remove any feature column that is constant (e.g., all zeros, or all a single value)
        #    This avoids having useless constant features in X_train.
        constant_cols = []
        for col in feature_cols:
            # If only one unique value in the column, it is constant:
            if df_sorted[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)

        # Drop constant columns from the DataFrame and from feature_cols
        if constant_cols:
            df_sorted = df_sorted.drop(columns=constant_cols)
            feature_cols = [col for col in feature_cols if col not in constant_cols]
        return df_sorted, feature_cols