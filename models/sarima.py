import logging

import pandas as pd

from models.algorithms.autoarima import forecast_arima_for_single_customer
from models.base import ForecastModel

logger = logging.getLogger(__name__)

class SARIMAModel(ForecastModel):
    def train(self) -> pd.DataFrame:
        logger.info("🚀 [SARIMAModel] Starting training...")
        result = forecast_arima_for_single_customer(self)
        logger.info("✅ [SARIMAModel] Training complete.")
        return result
