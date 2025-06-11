from typing import List, Any, Tuple


class RFConfig:
    """
    Holds configuration for the RF pipeline.
    """
    rf_params: tuple[Any]

    def __init__(
        self,
        lag_list: List[int],
        rolling_windows: List[int],
        encoder_method: str,
        rf_params: Tuple[Any],
        min_history: int,
        test_fraction: float
    ):
        self.lag_list = lag_list
        self.rolling_windows = rolling_windows
        self.encoder_method = encoder_method
        self.rf_params = rf_params
        self.min_history = min_history
        self.test_fraction = test_fraction