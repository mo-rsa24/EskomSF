import logging

import pandas as pd

from models.algorithms.XGBoost import run_xgb_forecast_pipeline
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XGBoostModel(ForecastModel):
    def train(self) -> pd.DataFrame:
        logger.info("🚀 [XGBoostModel] Starting training...")
        result = run_xgb_forecast_pipeline(self)
        logger.info("✅ [XGBoostModel] Training complete.")
        return result