import pandas as pd
from .utilities import *
from dataclasses import dataclass

@with_db_connection
def get_all_query(conn, query):
    cursor = conn.cursor()
    cursor.execute(query)
    logging.info("Query executed successfully.")
    return cursor.fetchall()


def get_sample_rows(table_name="dbo.DataBrickTasks", limit=5):
    query = f"SELECT TOP {limit} * FROM {table_name}"
    return get_all_query(query)

@with_db_connection
def query_to_df(conn, query):
    return pd.read_sql(query, conn)

def get_user_forecast_data(databrick_task_id=39):
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
    databrick_id: int

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
        databrick_id=row["DatabrickID"],
    )
