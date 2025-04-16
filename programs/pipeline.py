from config_loader import HyperParameterConfig
from data.dataset import ForecastDataset
from db.queries import ForecastConfig
from models.XG import XGBoostModel
from models.arima import ARIMAModel
from models.random_forest import RandomForestModel
from models.sarima import SARIMAModel


class ForecastPipeline:
    def __init__(self, dataset: ForecastDataset, config: HyperParameterConfig):
        self.model = self.get_model(dataset, config)

    def get_model(self, dataset: ForecastDataset,config: HyperParameterConfig):
        name = dataset.ufm_config.forecast_method_name.lower().replace(" ","")
        if name == "arima":
            return ARIMAModel(dataset, config)
        elif name == "sarima":
            return SARIMAModel(dataset, config)
        elif name == "randomforest":
            return RandomForestModel(dataset, config)
        elif name == "xgboost":
            return XGBoostModel(dataset, config)
        else:
            raise ValueError(f"Unknown model type: {name}")

    def run(self):
        self.model.prepare_data()
        return self.model.train()
