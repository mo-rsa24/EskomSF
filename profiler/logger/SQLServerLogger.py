from db.utilities import get_db_conn_str
from profiler.logger.BaseLogger import BaseLogger


class SQLServerLogger(BaseLogger):
    def __init__(self, env: str = None):
        self.env = env

    def log(self, record: dict):
        conn_str = get_db_conn_str()
        import pyodbc
        conn = pyodbc.connect(conn_str)
        try:
            from db.queries import insert_profiling_log
            return insert_profiling_log(record)  # Pass open conn
        finally:
            conn.close()

    def log_error(self, log_id, error, traceback, error_type=None, severity=None, component=None):
        import pyodbc
        conn = pyodbc.connect(get_db_conn_str())
        try:
            from db.queries import insert_profiling_error  # Assumes function is engine-aware
            return insert_profiling_error(
                log_id=log_id,
                error=error,
                traceback=traceback,
                error_type=error_type,
                severity=severity,
                component=component,
                conn=conn
            )
        finally:
            conn.close()