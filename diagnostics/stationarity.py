# diagnostics/stationarity.py
import logging
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


def adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'statistic': result[0],
        'pvalue': result[1],
        'critical_values': result[4],
        'stationary': result[1] < 0.05
    }


def kpss_test(series: pd.Series, regression: str = 'c') -> dict:
    result = kpss(series.dropna(), regression=regression, nlags="auto")
    return {
        'statistic': result[0],
        'pvalue': result[1],
        'critical_values': result[3],
        'stationary': result[1] > 0.05
    }


def run_stationarity_tests(series: pd.Series, title="") -> dict:
    """
        For example, if:
        ADF says non-stationary (p > 0.05)

        KPSS says non-stationary (p < 0.05)
        → You have strong evidence of non-stationarity

        But if:

        ADF says stationary (p < 0.05)

        KPSS says stationary (p > 0.05)
        → You have strong evidence of stationarity
    """
    logger.info(f"🧪 Running stationarity tests for: {title}")
    adf_result = adf_test(series)
    try:
        kpss_result = kpss_test(series)
    except Exception as e:
        logger.warning(f"⚠️ KPSS test failed for {title}: {e}")
        kpss_result = {'statistic': None, 'pvalue': None, 'critical_values': {}, 'stationary': None}

    return {
        'ADF': adf_result,
        'KPSS': kpss_result
    }
