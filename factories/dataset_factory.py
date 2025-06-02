import os

from data.dataset import ForecastDataset
from data.spark_forecast_dataset import SparkForecastDataset


class DatasetFactory:
    @staticmethod
    def create(databrick_task_id: int, save: bool = False, spark=None) -> SparkForecastDataset | ForecastDataset:
        env = os.getenv("ENV", "LOCAL").upper()
        if env == "DATABRICKS":
            if spark is None:
                raise ValueError("SparkSession must be provided in Databricks environment.")
            return SparkForecastDataset(databrick_task_id, spark, save)
        return ForecastDataset(databrick_task_id, save)
