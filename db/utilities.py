# utilities.py
import os
import logging
import yaml
import pyodbc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yaml_config(path="config.yaml"):
    """Load YAML configuration from the given path."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_environment_config(config_path="config.yaml"):
    """Returns the current environment name and its config block."""
    config = load_yaml_config(config_path)
    env = os.getenv("ENV")
    if env not in config:
        raise ValueError(f"Environment '{env}' not found in config file.")
    return env, config[env]

def get_db_conn_str(config_path="config.yaml"):
    """Build and return the connection string for the current environment."""
    env, env_config = get_environment_config(config_path)
    # Credentials: try environment vars first
    user = os.getenv("DB_USER", "fortrackSQL")
    password = os.getenv("DB_PASSWORD", "vuxpapyvu@2024")
    # Remove any jdbc:// prefix if present
    server = env_config["server"].split("//")[1]  # Strip jdbc:
    database = env_config["database"]
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
    )
    return conn_str

def with_db_connection(fn):
    """Higher-order function to handle DB connection and errors."""
    def wrapper(*args, **kwargs):
        try:
            # Build the connection string only when needed.
            conn_str = get_db_conn_str()
            with pyodbc.connect(conn_str) as conn:
                return fn(conn, *args, **kwargs)
        except pyodbc.Error as e:
            logger.error(f"Database operation failed: {e}")
            return None
    return wrapper
