# utilities/exit_handler.py
from config.environment import is_databricks

def safe_exit(code: str, message: str):
    full_msg = f"{code}: {message}"
    if is_databricks():
        try:
            import dbutils
            dbutils.notebook.exit(f"[Notebook Exit] {full_msg}")
        except Exception:
            pass  # dbutils or context not ready
    else:
        raise SystemExit(f"[Local Exit] {full_msg}")
