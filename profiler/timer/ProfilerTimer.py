# Profiler context manager
import threading
import traceback
from datetime import timedelta
import time
from functools import wraps
import socket

import pandas as pd
from db.utilities import logger as global_logger
from docstring.validate_category import category_validator
from profiler.logger.MariaDBLogger import MariaDBLogger
from profiler.logger.SparkLogger import SparkLogger


class ProfilerTimer:
    def __init__(
        self,
        module,
        function,
        logger_backend,
        message="",
        category="general",
        run_id=None,
        app_name=None,
        context=None
    ):
        self.module = module
        self.function = function
        self.message = message
        self.logger = logger_backend
        self.hostname = socket.gethostname()
        self.thread_id = threading.get_ident()
        self.category = category
        if not category_validator.validate(category):
            self.category = "unknown"
        self.run_id = run_id
        self.app_name = app_name
        self.log_id = None  # ✅ Will store log insert ID for FK to errors
        self.context = context or {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration_ms = round((end_time - self.start_time) * 1000, 3)
        duration_readable = str(timedelta(milliseconds=duration_ms)).split(".")[0]

        self.status = "failed" if exc_type else "completed"
        error_msg = str(exc_val) if exc_type else None
        trace = traceback.format_exc() if exc_type else None

        record = {
            "run_id": self.run_id,
            "app_name": self.app_name,
            "module": self.module,
            "function": self.function,
            "message": self.message,
            "start_time": pd.Timestamp.fromtimestamp(self.start_time),
            "end_time": pd.Timestamp.fromtimestamp(end_time),
            "duration_ms": duration_ms,
            "duration_readable": duration_readable,
            "error": str(exc_val) if exc_type else None,
            "traceback": traceback.format_exc() if exc_type else None,
            "hostname": self.hostname,
            "thread_id": self.thread_id,
            "status": self.status,
            "category": self.category  # ✅ NEW
        }

        record = {**record, **self.context}

        try:
            # ✅ Log to main table
            self.log_id = self.logger.log(record)  # should return log_id for relational backends
            from db.error_logger import insert_profiling_error
            # ✅ If failed, also write to profiling_errors
            if self.status == "failed" and self.log_id:
                insert_profiling_error(
                    log_id=self.log_id,
                    error=error_msg,
                    traceback=trace
                )

        except Exception as logging_error:
            global_logger.warning(f"[Profiler] Logging failed for {self.function}: {logging_error}")


    @staticmethod
    def timer(logger_backend, module, function, message=""):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with ProfilerTimer(module, function, logger_backend, message):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# logger Factory
def get_logger(engine="mariadb", spark_session=None, env=None):
    if engine == "mariadb":
        return MariaDBLogger(env=env)
    elif engine == "sqlserver":
        from profiler.logger.SQLServerLogger import SQLServerLogger
        return SQLServerLogger(env=env)
    elif engine == "pyspark":
        if spark_session is None:
            raise ValueError("Spark session required for SparkLogger")
        return SparkLogger(spark_session)
    else:
        raise ValueError(f"Unsupported logging engine: {engine}")