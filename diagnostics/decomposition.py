# diagnostics/decomposition.py
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import pandas as pd

def plot_stl_decomposition(series: pd.Series, period: int = 12, title: str = ""):
    series = series.dropna()
    stl = STL(series, period=period)
    result = stl.fit()
    result.plot()
    plt.suptitle(f"STL Decomposition - {title}", fontsize=14)
    plt.tight_layout()
    plt.show()
    return result
