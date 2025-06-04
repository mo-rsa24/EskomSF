# utilities.py
import os
import logging
from functools import wraps
from typing import Optional, Callable

import numpy as np
import yaml
import pyodbc

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


def get_mariadb_connection_from_config(config_path="config.yaml"):
    """
    Connect to the profiler database (MariaDB), specifically from the LOCAL config.
    """
    env, env_config = get_environment_config(config_path)

    if env != "LOCAL" or "profiler" not in env_config:
        raise ValueError("MariaDB profiler config is only available in LOCAL environment.")

    db_conf = env_config["profiler"]

    conn_str = (
        f"DRIVER={db_conf['driver']};"
        f"SERVER={db_conf['host']};"
        f"PORT={db_conf['port']};"
        f"DATABASE={db_conf['database']};"
        f"UID={db_conf['user']};"
        f"PWD={db_conf['password']};"
        f"OPTION=3;"
        f"CHARSET=UTF8MB4;"
    )
    return pyodbc.connect(conn_str)


def get_db_conn_str(config_path="config.yaml"):
    """
    Build and return the connection string for the data source (e.g., QA or DEV) using ODBC.
    This is used for the main SQL Server database, not the profiler.
    """
    env, env_config = get_environment_config(config_path)

    # Resolve which config to use: full config in non-LOCAL, or nested in LOCAL
    if env == "LOCAL":
        ds_config = env_config.get("datasource_config", {})
    else:
        ds_config = env_config

    user = os.getenv("DB_USER", "fortrackSQL")
    password = os.getenv("DB_PASSWORD", "vuxpapyvu@2024")

    # Extract server details safely
    server = ds_config.get("server", "").split("//")[-1]
    database = ds_config["database"]

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
    )
    try:
        return conn_str
    except KeyError as e:
        meta = get_error_metadata("ConfigMissing", {"field": str(e)})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback="",
            error_type="ConfigMissing",
            severity=meta["severity"],
            component=meta["component"]
        )
        safe_exit(meta["code"], meta["message"])

def with_db_connection(conn_provider: Optional[Callable[[], pyodbc.Connection]] = None):
    """
    Decorator for injecting a database connection into a function.

    Parameters:
    - conn_provider: Callable that returns a DB connection (defaults to get_db_conn_str-based remote)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            provider = conn_provider or default_connection_provider
            conn = provider()
            try:
                result = func(conn, *args, **kwargs)
            finally:
                conn.close()
            return result
        return wrapper
    return decorator

# 🔌 Default connection provider (uses your current remote config)
def default_connection_provider():
    conn_str = get_db_conn_str()
    return pyodbc.connect(conn_str)


def clean_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8").replace("\uFFFD", "?")


def normalize_value(value):
    """Casts NumPy scalar types to native Python types for DB compatibility."""
    if isinstance(value, (np.generic,)):
        return value.item()
    return value

def get_valid_columns(table_name: str, field_mappings: dict) -> set:
    """Returns valid source-side columns for the given table context."""
    env = getenv("ENV", "LOCAL").upper()
    if env == "LOCAL":
        # Get all local-side keys from field mapping that are mapped to non-null DB columns
        reverse_map = {v: k for k, v in field_mappings[table_name].items() if v}
        return set(reverse_map.values())
    else:
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

