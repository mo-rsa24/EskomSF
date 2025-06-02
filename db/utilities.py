# utilities.py
import os
import logging
from functools import wraps
from typing import Optional, Callable

import numpy as np
import yaml
import pyodbc

from config.config_service import get_profiling_engine
from db.db_connection_pool import get_connection
from db.error_logger import insert_profiling_error
from profiler.errors.utils import get_error_metadata
from utils.exit_handler import safe_exit
from os import getenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PROFILING_LOG_COLUMNS = {
    "run_id", "app_name", "module", "function", "message",
    "start_time", "end_time", "duration_ms", "duration_readable",
    "error", "traceback", "hostname", "thread_id", "status",
    "category", "forecast_method_id", "forecast_method_name",
    "databrick_task_id", "user_forecast_method_id"
}


def load_yaml_config(path="config.yaml"):
    """Load YAML configuration from the given path."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

import os
from dotenv import load_dotenv


def get_environment_config(config_path="config.yaml"):
    """
    Returns a tuple (env_name, full_env_config) for the current ENV.

    If LOCAL is selected, supports:
    - profiler: local connection details
    - datasource: alias to another environment (e.g., QA or DEV) for data fetching
    """
    load_dotenv()
    config = load_yaml_config(config_path)

    env = os.getenv("ENV", "LOCAL").upper()
    if env not in config:
        raise ValueError(f"Environment '{env}' not found in config file.")

    base_config = config[env]

    # Resolve alias if LOCAL has 'datasource' pointing to another environment
    if env == "LOCAL" and "datasource" in base_config["profiling_cfg"]:
        ds_key = base_config["profiling_cfg"]["datasource"]
        if ds_key not in config:
            from db.error_logger import insert_profiling_error
            meta = get_error_metadata("ConfigMissing", {"field": ds_key})
            insert_profiling_error(
                log_id=None,
                error=meta["message"],
                traceback="",  # or traceback.format_exc()
                error_type="ConfigMissing",
                severity=meta["severity"],
                component=meta["component"]
            )
            safe_exit(meta["code"], meta["message"])
        base_config["datasource_config"] = config[ds_key]

    return env, base_config

def with_db_connection(engine=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            conn = get_connection(engine or "sqlserver")  # default fallback
            try:
                return func(conn, *args, **kwargs)
            finally:
                pass  # connection reused
        return wrapper
    return decorator

def clean_text(text: str) -> str:
    return (
        text.encode("utf-8", errors="replace")
            .decode("utf-8")
            .replace("\uFFFD", "?")
            .replace("\n", " ")
            .replace("\r", "")
            .strip()
    )


def normalize_value(value):
    """Casts NumPy scalar types to native Python types for DB compatibility."""
    if isinstance(value, (np.generic,)):
        return value.item()
    return value

def get_valid_columns(table_name: str, field_mappings: dict) -> set:
    """
    Returns valid source-side columns for the given table context.

    - For mariadb (LOCAL dev), reverse map local-side keys.
    - For sqlserver (DEV/QA), return destination DB columns directly.
    """
    engine = get_profiling_engine()

    if engine == "mariadb":
        # Reverse map: DB column -> source field
        reverse_map = {v: k for k, v in field_mappings[table_name].items() if v}
        return set(reverse_map.values())
    else:
        # SQL Server engine → use DB-side column names directly
        return set(field_mappings[table_name].values())

def build_insert_query_and_values(table: str, record: dict, valid_columns: set):
    # Escape reserved words with square brackets
    SQL_SERVER_RESERVED_WORDS = ['FUNCTION']
    def escape_column(col: str) -> str:
        return f"[{col}]" if col.upper() in SQL_SERVER_RESERVED_WORDS else col
    clean_record = {k: v for k, v in record.items() if v is not None and k in valid_columns}
    if 'message' in clean_record:
        clean_record['message'] = clean_text(clean_record.get('message'))
    escaped_columns = [escape_column(col) for col in clean_record.keys()]
    columns = ", ".join(escaped_columns)
    placeholders = ", ".join(["?"] * len(clean_record))
    values = [normalize_value(v) for v in clean_record.values()]
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    return query, values

