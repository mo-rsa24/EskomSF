# connection_pool.py
"""
Memoized DB connection pool for profiling (MariaDB and SQL Server).
Avoids repeated instantiation by engine/environment.
"""

import pyodbc
from config.config_service import get_profiling_engine, get_env_config, get_datasource_config
from threading import Lock

_connection_cache = {}
_lock = Lock()


def _build_sqlserver_conn_str() -> str:
    config = get_datasource_config()
    server = config.get("server", "localhost").split("//")[-1]
    database = config.get("database")
    user = config.get("user", "fortrackSQL")
    password = config.get("password", "vuxpapyvu@2024")

    return (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
    )

def _build_mariadb_conn_str() -> str:
    profiler = get_env_config().get("profiler", {})
    return (
        f"DRIVER={profiler['driver']};"
        f"SERVER={profiler['host']};"
        f"PORT={profiler['port']};"
        f"DATABASE={profiler['database']};"
        f"UID={profiler['user']};"
        f"PWD={profiler['password']};"
        f"OPTION=3;CHARSET=UTF8MB4;"
    )

def get_connection(engine: str = None) -> pyodbc.Connection:
    from profiler.errors.utils import get_error_metadata
    from db.queries import insert_profiling_error
    from utils.exit_handler import safe_exit
    import traceback

    engine = engine or get_profiling_engine()

    with _lock:
        if engine not in _connection_cache:
            try:
                if engine == "mariadb":
                    conn_str = _build_mariadb_conn_str()
                elif engine == "sqlserver":
                    conn_str = _build_sqlserver_conn_str()
                else:
                    raise ValueError(f"Unsupported engine: {engine}")

                _connection_cache[engine] = pyodbc.connect(conn_str)

            except Exception as e:
                meta = get_error_metadata("DBConnectionError", {"engine": engine, "exception": str(e)})
                insert_profiling_error(
                    log_id=None,
                    error=meta["message"],
                    traceback=traceback.format_exc(),
                    error_type="DBConnectionError",
                    severity=meta["severity"],
                    component=meta["component"]
                )
                safe_exit(meta["code"], meta["message"])

        return _connection_cache[engine]

