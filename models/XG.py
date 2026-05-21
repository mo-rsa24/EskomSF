import logging

import pandas as pd

from models.algorithms.XGBoost import train_XGBoost_globally_forecast_locally_with_aggregation
from models.algorithms.tree_algorithms.xgb import forecast_xgb_for_single_customer
from models.base import ForecastModel

logger = logging.getLogger(__name__)

class XGBoostModel(ForecastModel):
    def train(self) -> pd.DataFrame:
        logger.info("🚀 [XGBoostModel] Starting training...")
        # result = train_XGBoost_globally_forecast_locally_with_aggregation(self)
        result = forecast_xgb_for_single_customer(self)
        logger.info("✅ [XGBoostModel] Training complete.")
        return result
