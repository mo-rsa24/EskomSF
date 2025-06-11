import logging

import pandas as pd

from models.algorithms.XGBoost import train_XGBoost_globally_forecast_locally_with_aggregation
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XGBoostModel(ForecastModel):
    def train(self) -> pd.DataFrame:
        logger.info("🚀 [XGBoostModel] Starting training...")
        result = train_XGBoost_globally_forecast_locally_with_aggregation(self)
        logger.info("✅ [XGBoostModel] Training complete.")
        return result