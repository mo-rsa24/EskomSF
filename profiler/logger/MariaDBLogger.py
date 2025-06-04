# MariaDB Strategy
import pandas as pd


from db.utilities import get_mariadb_connection_from_config
from profiler.logger.BaseLogger import BaseLogger


class MariaDBLogger(BaseLogger):
    def __init__(self, env: str = None):
        self.env = env

    def log(self, record: dict):
        conn = get_mariadb_connection_from_config()
        try:
            from db.queries import insert_profiling_log
            return insert_profiling_log(record)
        finally:
            conn.close()

# Spark Strategy
