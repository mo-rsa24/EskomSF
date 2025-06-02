import os
import traceback

import pandas as pd
import pyodbc
from pyspark.sql import SparkSession

from config.config_service import get_profiling_engine
from docstring.utilities import profiled_function
from profiler.errors.utils import get_error_metadata
from profiler.profiler_switch import profiling_switch
from utils.exit_handler import safe_exit
from .db_connection_pool import get_connection
from .utilities import with_db_connection, logger, build_insert_query_and_values
from dataclasses import dataclass
import yaml

# Load field mappings from a YAML file
with open("db/field_mappings.yaml", "r") as f:
    FIELD_MAPPINGS = yaml.safe_load(f)

def map_fields(record, field_map):
    return {field_map[k]: v for k, v in record.items() if k in field_map and field_map[k] is not None}


@with_db_connection()
def get_all_query(conn, query: str):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        logger.info("✅ Query executed successfully.")
        return cursor.fetchall()
    except Exception as e:
        meta = get_error_metadata("❌ ConnectionRefused", {"resource": "query engine", "exception": str(e)})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback=traceback.format_exc(),
            error_type="ConnectionRefused",
            severity=meta["severity"],
            component=meta["component"]
        )
        safe_exit(meta["code"], meta["message"])


def get_sample_rows(table_name="dbo.DataBrickTasks", limit=5):
    query = f"SELECT TOP {limit} * FROM {table_name}"
    return get_all_query(query)


@profiled_function(category="database_call",enabled=profiling_switch.enabled)
def get_predictive_data(UFMID=64): # This dataset that we fine-tune
    query_act_cons = f"""
                         ( select * from dbo.PredictiveInputData({UFMID}))
                        """
    return query_to_df(query_act_cons)

def get_actual_data(rows=5000):
    query = f"""
select top {rows}
	ReportingMonth
	,CustomerID
	,PodID
	,sum(OffpeakConsumption) as OffpeakConsumption
	,sum(StandardConsumption) as StandardConsumption
	,sum(PeakConsumption) as PeakConsumption
	,sum(Block1Consumption) as Block1Consumption
	,sum(Block2Consumption) as Block2Consumption
	,sum(Block3Consumption) as Block3Consumption
	,sum(Block4Consumption) as Block4Consumption
	,sum(NonTOUConsumption) as NonTOUConsumption
from ActualData
group by
	ReportingMonth
	,CustomerID
	,PodID"""
    return query_to_df(query)

@with_db_connection()
def query_to_df(conn, query: str):
    try:
        return pd.read_sql(query, conn)
    except Exception as e:
        meta = get_error_metadata("❌ PandasLoadError", {"query": query, "exception": str(e)})
        insert_profiling_error(
            log_id=None,
            error=meta["message"],
            traceback=traceback.format_exc(),
            error_type="PandasLoadError",
            severity=meta["severity"],
            component=meta["component"]
        )
        safe_exit(meta["code"], meta["message"])

@profiled_function(category="database_call",enabled=profiling_switch.enabled,)
def get_user_forecast_data(databrick_task_id=39, spark: SparkSession = None ):
    """
    Function: Retrieves the most recent user forecast method configuration ( DataBrickTasks, UserForecastMethod, and DimForecastMethod) details associated with a specific Databrick task.

    This query joins the DataBrickTasks, UserForecastMethod, and DimForecastMethod tables to extract
    relevant forecast configuration information. It returns one record (the earliest by CreationDate)
    matching the specified DatabrickID.

    Parameters:
    databrick_task_id (int): The ID of the Databrick task to filter the query.
    Args:
        databrick_task_id:

    """
    query = f"""
    SELECT TOP 1
        ufm.StartDate, ufm.EndDate, ufm.Parameters, ufm.Region, ufm.Status,
        ufm.ForecastMethodID, ufm.UserForecastMethodID,
        ufm.JSONCustomer as CustomerJSON, ufm.varJSON,
        dfm.Method, dbt.DatabrickID
    FROM 
        [dbo].[DataBrickTasks] AS dbt
    INNER JOIN 
        [dbo].[UserForecastMethod] AS ufm ON dbt.UserForecastMethodID = ufm.UserForecastMethodID
    INNER JOIN 
        [dbo].[DimForecastMethod] AS dfm ON ufm.ForecastMethodID = dfm.ForecastMethodID
    WHERE dbt.DatabrickID = {databrick_task_id}
    ORDER BY dbt.CreationDate
    """
    return query_to_df(query)

def insert_profiling_log(record: dict):
    try:
        engine = get_profiling_engine()
        table = "profiling_logs" if engine == "mariadb" else "PredictiveProfilingLogs"

        # Step 2: Field mapping if SQL Server
        if engine == "sqlserver":
            field_map = FIELD_MAPPINGS["profiling_logs"]
            record = {field_map[k]: v for k, v in record.items() if k in field_map and field_map[k] is not None}

        # Step 3: Validate fields and build query
        from db.utilities import get_valid_columns
        valid_cols = get_valid_columns('profiling_logs', FIELD_MAPPINGS)
        query, values = build_insert_query_and_values(table, record, valid_cols)

        # Step 4: Execute insert and return ID
        conn = get_connection(engine)
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        try:
            cursor.execute("SELECT SCOPE_IDENTITY()" if engine == "sqlserver" else "SELECT LAST_INSERT_ID()")
            result = cursor.fetchone()
            if result and result[0]:
                return int(result[0])
        except Exception as id_err:
            logger.warning(f"[Profiler] ID function failed: {id_err}")

            # ⛑ Fallback to ORDER BY
        id_column = "id" if engine == "mariadb" else "ID"  # adjust as per schema
        cursor.execute(f"SELECT TOP 1 {id_column} FROM {table} ORDER BY {id_column} DESC")
        result = cursor.fetchone()
        return int(result[0]) if result else None

    except Exception as e:
        meta = get_error_metadata("ProfilerInsertFailure", {"exception": str(e)})
        logger.error(meta["message"])
        return None

def insert_profiling_error(
    log_id,
    error,
    traceback,
    error_type: str = None,
    severity: str = None,
    component: str = None
):
    try:
        engine = get_profiling_engine()

        # Step 1: Table & Field Map
        table = "profiling_errors" if engine == "mariadb" else "PredictiveProfilingErrors"
        record = {
            "profiling_log_id": log_id,
            "error": error,
            "traceback": traceback,
            "error_type": error_type,
            "severity": severity,
            "component": component
        }

        # Step 2: Field Mapping if SQL Server
        if engine == "sqlserver":
            field_map = FIELD_MAPPINGS["profiling_errors"]
            record = {field_map[k]: v for k, v in record.items() if k in field_map and field_map[k] is not None}

        from db.utilities import get_valid_columns
        valid_cols = get_valid_columns("profiling_errors", FIELD_MAPPINGS)
        query, values = build_insert_query_and_values(table, record, valid_cols)

        # Step 4: Execute insert
        conn = get_connection(engine)
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()

    except Exception as e:
        meta = get_error_metadata("ProfilerInsertFailure", {"exception": str(e)})
        logger.error(meta["message"])
        return None



@dataclass
class ForecastConfig:
    forecast_method_id: int
    forecast_method_name: str
    model_parameters: str
    region: str
    status: str
    user_forecast_method_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    databrick_task_id: int

def row_to_config(row: pd.Series) -> ForecastConfig:
    return ForecastConfig(
        forecast_method_id=row["ForecastMethodID"],
        forecast_method_name=row["Method"],
        model_parameters=row["Parameters"],
        region=row["Region"],
        status=row["Status"],
        user_forecast_method_id=row["UserForecastMethodID"],
        start_date=row["StartDate"],
        end_date=row["EndDate"],
        databrick_task_id=row["DatabrickID"],
    )

