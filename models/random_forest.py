import logging

import pandas as pd

from models.algorithms.tree_algorithms.random_forest import train_random_forest_for_single_customer
from models.base import ForecastModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RandomForestModel(ForecastModel):
    def train(self) -> pd.DataFrame:
        logger.info("🚀 [RFModel] Starting training...")
        result = train_random_forest_for_single_customer(self)
        logger.info("✅ [RFModel] Training complete.")
        return result