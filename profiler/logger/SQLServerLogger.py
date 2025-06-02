from db.queries import insert_profiling_log, insert_profiling_error
from db.utilities import with_db_connection
from profiler.logger.BaseLogger import BaseLogger


class SQLServerLogger(BaseLogger):
    def __init__(self, env: str = None):
        self.env = env

    @with_db_connection(engine="sqlserver")
    def log(self, conn, record: dict) -> int:
        return insert_profiling_log(record)

    def log_error(self, log_id, error, traceback, error_type=None, severity=None, component=None):
        return insert_profiling_error(
            log_id=log_id,
            error=error,
            traceback=traceback,
            error_type=error_type,
            severity=severity,
            component=component
        )