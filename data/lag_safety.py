import logging
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def validate_lag_vs_horizon(
    lag_hours: List[int],
    forecast_horizon: int,
    fail_on_violation: bool = False
) -> None:
    """
    Risk:
    If forecast_horizon = 48 months, but you've created lags of 12, 24, 36, and 48, then:

        -  During training, these lags work

        - But during inference, those lag values are not available unless:

        - You recursively predict (risk: error propagation)

    You "cheat" with future info (leakage)

    Warn (or error) if any lag values exceed the forecastable horizon.

    Args:
        lag_hours (List[int]): List of lag values (e.g., [168, 336, 504])
        forecast_horizon (int): Forecast window length (e.g., 48 months)
        fail_on_violation (bool): If True, raise an error instead of just warning


    *Working with monthly data:*
        - Common lags: [3,12,24,36] This can scale up nicely

    """
    violations = [lag for lag in lag_hours if lag > forecast_horizon]
    if violations:
        msg = (
            f"⚠️ Lags {violations} exceed forecast horizon ({forecast_horizon}). "
            f"These cannot be used during inference unless recursively predicted."
        )
        if fail_on_violation:
            raise ValueError(msg)
        else:
            logger.warning(msg)


def describe_lagging_strategy(use_lag_features: bool, train_mode: str = "direct") -> str:
    """
    Provide a human-readable explanation of the forecasting strategy.

    Args:
        use_lag_features (bool): Whether the model is trained using lag features.
        train_mode (str): 'direct' (single-step) or 'recursive' (multi-step forecast)

    Returns:
        str: Explanation of modeling tradeoff
    """
    if not use_lag_features:
        return (
            "🧭 Lag-Free Forecasting Enabled:\n"
            "  - Excludes historical lags (y[t-k])\n"
            "  - Ideal for long-horizon stability\n"
            "  - Uses only features known at forecast time\n"
            f"  - Forecasting Mode: {train_mode}"
        )
    else:
        return (
            "🔁 Lag-Based Forecasting Enabled:\n"
            "  - Includes lagged target values (e.g., y[t-12])\n"
            "  - Improves short-term accuracy but risks compounding error\n"
            "  - Use 'recursive' mode for longer horizons\n"
            f"  - Forecasting Mode: {train_mode}"
        )


def log_lag_strategy(use_lag_features: bool, train_mode: str = "direct"):
    logger.info(describe_lagging_strategy(use_lag_features, train_mode))
