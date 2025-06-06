# manager.py

import logging
from typing import Any, List, Tuple, Dict

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from evaluation.performance import RandomForestPerfRow
from models.algorithms.tree_algorithms.random_forest.config import RFConfig
from models.algorithms.tree_algorithms.random_forest.feature_engineering import FeatureEngineer
from models.algorithms.tree_algorithms.random_forest.rf_factory import EncoderFactory
from models.algorithms.tree_algorithms.random_forest.utils import evaluate_regression, train_test_split_time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RFModelManager:
    """
    Trains and manages Random Forest models for Eskom forecasting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        date_col: str,
        consumption_col: str,
        config: RFConfig
    ):
        self.df_raw = df.copy()
        self.group_cols = group_cols
        self.date_col = date_col
        self.consumption_col = consumption_col
        self.config = config

        self.feature_engineer = FeatureEngineer(
            group_cols=group_cols,
            date_col=date_col,
            consumption_cols=[consumption_col],
            lag_list=config.lag_list,
            rolling_windows=config.rolling_windows,
            seasonal=True,
            drop_na=True
        )
        self.encoder = EncoderFactory.get_encoder(config.encoder_method)
        self.models: Dict[Tuple[Any, Any], RandomForestRegressor] = {}
        self.feat_cols_map: Dict[Tuple[Any, Any], List[str]] = {}
        self.perf_rows: List[RandomForestPerfRow] = []

    def _encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enc = df.copy()
        for col in self.group_cols + ["TariffID"]:
            if col not in df_enc.columns:
                continue
            if self.config.encoder_method == "frequency":
                df_enc[f"{col}_enc"] = self.encoder(df_enc[col])
            else:
                df_enc[f"{col}_enc"] = self.encoder(
                    df_enc, col, self.consumption_col
                )
        return df_enc

    def train_all_series(self) -> Tuple[List[RandomForestPerfRow], List[pd.DataFrame]]:
        """
        Trains one RF model per (CustomerID, PodID). Populates self.models and self.perf_rows.
        Returns:
          - all_perf_rows: List[RandomForestPerfRow]
          - all_forecast_dfs: List[pd.DataFrame] (each containing in‐sample test forecasts)
        """
        df_all = self.df_raw.copy().sort_values(self.group_cols + [self.date_col])
        unique_pairs = df_all[self.group_cols].drop_duplicates().to_records(index=False)

        all_forecast_dfs: List[pd.DataFrame] = []

        for (cust_id, pod_id) in unique_pairs:
            # Filter raw series
            mask = (
                (df_all[self.group_cols[0]] == cust_id) &
                (df_all[self.group_cols[1]] == pod_id)
            )
            df_series = df_all.loc[mask].copy().sort_values(self.date_col)

            # Skip if insufficient history
            if len(df_series) < self.config.min_history:
                self.perf_rows.append(
                    RandomForestPerfRow(
                        cust_id, pod_id, 0, 0, 0, 0, 0,
                        error=f"insufficient_history (n={len(df_series)})"
                    )
                )
                continue

            # Feature generation
            feats_df, feat_cols = self.feature_engineer.generate(df_series)

            # Include TariffID if present
            if "TariffID" in df_series.columns:
                feats_df["TariffID"] = df_series["TariffID"]

            # Encode IDs
            feats_df, new_id_cols = self._encode_ids_and_tariff(feats_df)
            feat_cols.extend(new_id_cols)



            for col in self.group_cols + ["TariffID"]:
                if col in feats_df.columns:
                    feat_cols.append(f"{col}_enc")

            if 'level_0' in feats_df.columns:
                feats_df.drop(columns=['level_0'], inplace=True)
            if 'level_0' in feat_cols:
                feat_cols.remove("level_0")
            self.feat_cols_map[(cust_id, pod_id)] = feat_cols.copy()
            train_df, test_df = train_test_split_time(
                feats_df, self.date_col, self.config.test_fraction
            )
            if train_df.empty or test_df.empty:
                self.perf_rows.append(
                    RandomForestPerfRow(
                        cust_id, pod_id, 0, 0, 0, 0, 0,
                        error="train/test split failed"
                    )
                )
                continue

            X_train = train_df[feat_cols]
            y_train = train_df[self.consumption_col]
            X_test = test_df[feat_cols]
            y_test = test_df[self.consumption_col]

            (n_estimators, max_depth, min_samples_split,
             min_samples_leaf, max_features, bootstrap_flag) = self.config.rf_params

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

            # Record performance and store model
            self.models[(cust_id, pod_id)] = rf
            perf_row = RandomForestPerfRow(
                cust_id, pod_id,
                mse=perf["MSE"], mae=perf["MAE"], r2=perf["R2"],
                n_train=len(X_train), n_test=len(X_test)
            )
            self.perf_rows.append(perf_row)

            # In‐sample test forecasts (for debugging/inspection)
            df_forecast = test_df[[*self.group_cols, self.date_col, self.consumption_col]].copy()
            df_forecast["Predicted"] = preds
            all_forecast_dfs.append(df_forecast)

            logger.info(
                f"[Series RF] Cust={cust_id}, Pod={pod_id} | "
                f"train={len(X_train)}, test={len(X_test)}, R2={perf['R2']:.3f}"
            )

        return self.perf_rows, all_forecast_dfs

    def train_global(self) -> Tuple[RandomForestRegressor, RandomForestPerfRow]:
        """
        Trains a single RF model across all series combined.
        Returns:
          - rf_model
          - RandomForestPerfRow summarizing global performance
        """
        feats_df, feat_cols = self.feature_engineer.generate(self.df_raw)

        # Add TariffID if present, then encode all IDs + TariffID
        if "TariffID" in self.df_raw.columns:
            feats_df["TariffID"] = self.df_raw["TariffID"]

        feats_df, new_id_cols = self._encode_ids_and_tariff(feats_df)
        feat_cols.extend(new_id_cols)
        feat_cols.remove("TariffID")

        self.feat_cols_map[("ALL", "ALL")] = feat_cols.copy()

        train_df, test_df = train_test_split_time(
            feats_df, self.date_col, self.config.test_fraction
        )
        X_train = train_df[feat_cols]
        y_train = train_df[self.consumption_col]
        X_test = test_df[feat_cols]
        y_test = test_df[self.consumption_col]

        (n_estimators, max_depth, min_samples_split,
         min_samples_leaf, max_features, bootstrap_flag) = self.config.rf_params

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
        logger.info(
            f"[Global RF] train={len(X_train)}, test={len(X_test)}, R2={perf['R2']:.3f}"
        )
        return rf, perf_row

    def _encode_ids_and_tariff(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Encodes CustomerID, PodID—and if present, TariffID—into numeric columns.
        Returns (df_encoded, new_feature_cols_to_append).
        """
        df_enc = df.copy()
        new_cols = []

        # Encode CustomerID and PodID
        for col in self.group_cols:
            df_enc[f"{col}_enc"] = self.encoder(df_enc[col])
            new_cols.append(f"{col}_enc")

        # Encode TariffID if present
        if "TariffID" in df_enc.columns:
            df_enc["TariffID_enc"] = self.encoder(df_enc["TariffID"])
            new_cols.append("TariffID_enc")
            # ** Drop raw TariffID now that we've created TariffID_enc **
            df_enc = df_enc.drop(columns=["TariffID"])

        return df_enc, new_cols
    def forecast_for_pod(
        self,
        cust_id: Any,
        pod_id: Any,
        forecast_horizon: List[pd.Timestamp]
    ) -> pd.DataFrame:
        """
        Given a trained model for (cust_id, pod_id), generate forecasts
        for each ReportingMonth in `forecast_horizon`.
        Ensures that X_pred uses exactly the feat_cols that were used during training.
        Returns a DataFrame with columns:
          [CustomerID, PodID, ReportingMonth, PredictedConsumption]
        """
        key = (cust_id, pod_id)
        if key not in self.models:
            raise KeyError(f"No model found for Customer={cust_id}, Pod={pod_id}")

        rf = self.models[key]
        stored_feats = self.feat_cols_map[key]

        # 1) Rebuild combined historical+future DataFrame
        df_all = self.df_raw.copy().sort_values(self.group_cols + [self.date_col])
        mask = (
            (df_all[self.group_cols[0]] == cust_id) &
            (df_all[self.group_cols[1]] == pod_id)
        )
        df_series = df_all.loc[mask].copy().sort_values(self.date_col)

        future_df = pd.DataFrame({
            self.group_cols[0]: cust_id,
            self.group_cols[1]: pod_id,
            self.date_col: forecast_horizon,
        })
        if "TariffID" in df_all.columns:
            future_df["TariffID"] = df_series["TariffID"].iloc[-1]

        df_combined = pd.concat([df_series, future_df], ignore_index=True)

        # 2) Re‐generate features on combined data
        feats_df, _ = self.feature_engineer.generate(df_combined)

        # 3) Add TariffID from combined, then encode + drop raw
        if "TariffID" in df_combined.columns:
            feats_df["TariffID"] = df_combined["TariffID"]
        feats_df, _ = self._encode_ids_and_tariff(feats_df)

        # 4) Ensure feats_df has all stored feature columns; if any are missing, add as zeros
        for col in stored_feats:
            if col not in feats_df.columns:
                feats_df[col] = 0.0

        # 5) X_pred = feats_df[stored_feats]
        pred_mask = feats_df[self.date_col].isin(forecast_horizon)
        X_pred = feats_df.loc[pred_mask, stored_feats]

        # 6) Predict
        preds = rf.predict(X_pred)

        # 7) Build output DataFrame
        df_out = feats_df.loc[pred_mask, [*self.group_cols, self.date_col]].copy()
        df_out["PredictedConsumption"] = preds
        return df_out
