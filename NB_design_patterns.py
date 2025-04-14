from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Domain-specific imports
from db.queries import get_user_forecast_data, row_to_config
from db.utilities import env, logger
from etl.etl import *
from modeling.autoarima import *


# --------------------------
# 🏗️ Structural Patterns
# --------------------------

class ForecastConfigBuilder:
    """Builder Pattern for complex configuration object"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._config = ForecastConfig()

    def with_dates(self, start: datetime, end: datetime):
        self._config.start_date = start
        self._config.end_date = end
        return self

    def with_model_params(self, params: Dict):
        self._config.model_parameters = params
        return self

    def build(self) -> 'ForecastConfig':
        config = self._config
        self.reset()
        return config


@dataclass
class ForecastConfig:
    """Data class for configuration"""
    start_date: datetime = None
    end_date: datetime = None
    model_parameters: Dict = None
    # Add other config fields as needed


# --------------------------
# 💡 Behavioral Patterns
# --------------------------

class ForecastStrategy(ABC):
    """Strategy Pattern for different forecast algorithms"""

    @abstractmethod
    def execute(self, data: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
        pass


class ARIMAStrategy(ForecastStrategy):
    def execute(self, data: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
        arima_order, seasonal_order = extract_sarimax_params(config.model_parameters)
        return forecast_arima_for_single_customer(
            data, config.customer_id, config, order=arima_order
        )


# --------------------------
# 🏭 Creational Patterns
# --------------------------

class ForecastFactory:
    """Factory Pattern for forecast object creation"""

    @staticmethod
    def create_forecaster(method: str) -> ForecastStrategy:
        if method == 'autoarima':
            return ARIMAStrategy()
        # Add other forecast methods
        raise ValueError(f"Unknown forecast method: {method}")


# --------------------------
# 📦 Main Application Class
# --------------------------

class ForecastEngine:
    """Facade Pattern for complex forecasting workflow"""

    def __init__(self, task_id: int):
        self.task_id = task_id
        self.config = self._load_config()
        self.data_processor = DataProcessor()
        self.forecaster = ForecastFactory.create_forecaster(
            self.config.forecast_method_name
        )

    def _load_config(self) -> ForecastConfig:
        """Template Method for config loading"""
        ufm_df = get_user_forecast_data(databrick_task_id=self.task_id)
        return row_to_config(ufm_df.iloc[0])

    def execute_forecast(self) -> pd.DataFrame:
        """Template Method Pattern for workflow steps"""
        try:
            # 1. Data Preparation
            df = self.data_processor.load_data(self.config)

            # 2. Validation
            if df.empty:
                raise DataLoadingError("Empty DataFrame detected")

            # 3. Forecasting Execution
            return self.forecaster.execute(df, self.config)

        except ForecastError as e:
            logger.error(f"Forecast failed: {str(e)}")
            raise


# --------------------------
# 🛡️ Error Handling
# --------------------------

class ForecastError(Exception):
    """Base exception for forecast-related errors"""
    pass


class DataLoadingError(ForecastError):
    """Custom exception for data issues"""
    pass


# --------------------------
# 🔄 Client Code Usage
# --------------------------

if __name__ == "__main__":
    logger.info(f"Running on {env}")

    try:
        engine = ForecastEngine(databrick_task_id=1)
        result = engine.execute_forecast()
        logger.info("✅ Forecast completed successfully")

    except ForecastError as e:
        logger.critical(f"🚫 Critical failure: {str(e)}")
        exit(1)
