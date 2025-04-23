import yaml
from dataclasses import dataclass
from typing import List

@dataclass
class HyperParameterConfig:
    selected_columns: List[str]
    consumption_types: List[str]
    mode: str
    log: bool

    def get(self, key: str, default=None):
        return getattr(self, key, default)

def load_config(path: str = 'config.yaml') -> HyperParameterConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    # Filter out keys used for the forecasting pipeline (ignore environment definitions)
    config_keys = ["selected_columns", "consumption_types", "mode", "log"]
    filtered_data = { key: data[key] for key in config_keys if key in data }
    return HyperParameterConfig(**filtered_data)
