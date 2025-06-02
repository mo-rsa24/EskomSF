# main.py
# import sys
# sys.path.append("/Workspace/Shared")
from profiler.init_error_hooks import init_error_hooks
from programs.pipeline import ForecastPipeline
from utils.logger_factory import get_logger
from utils.mlflow_control import disable_all_autologgers
from utils.safe_import import safe_import
import os

from config.init_runtime import init_spark
from config_loader import load_config
from config.parameter_resolver import get_parameter
from config.environment import is_databricks

import sys
sys.stdout.reconfigure(encoding='utf-8')

from factories.dataset_factory import DatasetFactory
init_error_hooks()
# ──────────────────────────────────────────────
# 1️⃣ Logging setup
logger = get_logger(__name__)
disable_all_autologgers()
# Safe Spark setup
spark_imports = lambda: (
    __import__('pyspark.sql').sql.SparkSession,
    __import__('pyspark.dbutils').dbutils.DBUtils
)
SparkSession, DBUtils = safe_import(spark_imports, logger) or (None, None)

# ──────────────────────────────────────────────

# 2️⃣ Initialize Spark early — safe and idempotent
spark = init_spark() if is_databricks() else None

# 3️⃣ Resolve Parameters
task_id = int(get_parameter("DATABRICK_TASK_ID", default="1"))

# 4️⃣ Load YAML config
config = load_config("config.yaml")

# 5️⃣ Instantiate Dataset (local vs databricks)
dataset = DatasetFactory.create(
    databrick_task_id=task_id,
    save=False,
    spark=spark
)

# 6️⃣ Run preprocessing
dataset.load_data()
dataset.parse_identifiers()
info = dataset.preprocess()
dataset.define_forecast_range()

logger.info(f"📦 Metadata extracted: {info['metadata']}")

# 7️⃣ Run Forecasting Pipeline

pipeline = ForecastPipeline(dataset=dataset, config=config)
arima_performance, forecast_combined_df = pipeline.run()

# 8️⃣ Persist Results
if is_databricks():
    logger.info("📡 Writing to Databricks JDBC tables")
    write_props = {
        "user": config.user,
        "password": config.password,
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
    }
    spark.createDataFrame(arima_performance).write.jdbc(
        url=config.write_url,
        table=config.tables["performance_metrics_table"],
        mode="append",
        properties=write_props
    )
    spark.createDataFrame(forecast_combined_df).write.jdbc(
        url=config.write_url,
        table=config.tables["target_table_name"],
        mode="append",
        properties=write_props
    )
else:
    logger.info("💾 Writing locally to CSV")
    os.makedirs("outputs", exist_ok=True)
    arima_performance.to_csv("outputs/performance.csv", index=False)
    forecast_combined_df.to_csv("outputs/forecast_combined.csv", index=False)
    logger.info("✅ Results saved to /outputs/")
