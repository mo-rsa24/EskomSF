from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseDataset(ABC):
    @abstractmethod
    def load_data(self) -> None:
        ...

    @abstractmethod
    def extract_metadata(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def preprocess(self) -> Dict[str, Any]:
        ...
