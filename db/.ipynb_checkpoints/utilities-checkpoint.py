import os
import logging
import yaml
import pyodbc
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load YAML Config
def load_config(env="DEV"):
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if env not in config:
        raise ValueError(f"Environment '{env}' is not found in  the config")
    return config[env]

# Determine environment
env = os.getenv("ENV", "DEV")
config = load_config(env)

# Database credentials from env vars (fallback to defaults if not set)

# Authentication (Use environment variables for security)
user = os.getenv("DB_USER", "fortrackSQL")
password = os.getenv("DB_PASSWORD", "vuxpapyvu@2024")

# SQL Server Details
server = config["server"].split("//")[1]  # Strip jdbc:
database = config["database"]

conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={user};"
    f"PWD={password};"
)

def with_db_connection(fn):
    """Higher-order function to handle DB connection and errors."""
    def wrapper(*args, **kwargs):
        try:
            with pyodbc.connect(conn_str) as conn:
                return fn(conn, *args, **kwargs)
        except pyodbc.Error as e:
            logging.error(f"Database operation failed: {e}")
            return None
    return wrapper
