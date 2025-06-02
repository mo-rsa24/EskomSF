import pandas as pd

from profiler.timer.ProfilerTimer import ProfilerTimer,get_logger

logger = get_logger(engine="mariadb", env="dev")

with ProfilerTimer("etl", "load_csv", logger, "Reading input data"):
    df = pd.read_csv("input.csv")
