from pyspark.sql import SparkSession

from data.base_dataset import BaseDataset
from data.dataset import ForecastDataset


class SparkForecastDataset(BaseDataset):
    def __init__(self, databrick_task_id: int, spark: SparkSession, save: bool = False):
        self.internal = ForecastDataset(databrick_task_id=databrick_task_id, spark=spark, save=save)

    def load_data(self) -> None:
        self.internal.load_data()

    def extract_metadata(self) -> dict:
        return self.internal.extract_metadata()

    def preprocess(self) -> dict:
        return self.internal.preprocess()
