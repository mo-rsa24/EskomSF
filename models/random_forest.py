import logging

import pandas as pd

from models.algorithms.tree_algorithms.random_forest.random_forest import run_rf_forecast_pipeline
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RandomForestModel(ForecastModel):

    def train(self) -> pd.DataFrame:
        logger.info("🚀 [RFModel] Starting training...")

        perf_df_all, forecast_df_all = run_rf_forecast_pipeline(self)
        logger.info("✅ [RFModel] Training complete.")

        self.performance_df = perf_df_all
        self.forecast_df = forecast_df_all
        return perf_df_all