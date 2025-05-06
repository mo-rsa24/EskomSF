# diagnostics/visualization.py

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_rolling_statistics(series: pd.Series, window: int = 12, title: str = ""):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()

    plt.figure(figsize=(12, 6))
    plt.plot(series, label="Original", color="black")
    plt.plot(roll_mean, label=f"Rolling Mean ({window})", linestyle="--", color="blue")
    plt.plot(roll_std, label=f"Rolling Std ({window})", linestyle=":", color="orange")
    plt.title(f"Rolling Statistics - {title}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 40, title: str = ""):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    plot_acf(series.dropna(), ax=ax[0], lags=lags)
    ax[0].set_title(f"ACF - {title}")

    plot_pacf(series.dropna(), ax=ax[1], lags=lags, method='ywm')
    ax[1].set_title(f"PACF - {title}")

    plt.tight_layout()
    plt.show()
