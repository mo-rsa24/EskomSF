import logging

import pandas as pd

from models.algorithms.XGBoost import train_xgboost_for_single_customer
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XGBoostModel(ForecastModel):
    def train(self) -> pd.DataFrame:
        logger.info("🚀 [XGBoostModel] Starting training...")
        result = train_xgboost_for_single_customer(self)
        logger.info("✅ [XGBoostModel] Training complete.")
        return result