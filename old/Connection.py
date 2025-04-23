# Databricks notebook source
error_message = None
debug = False

try:
    # Import necessary libraries
    from pyspark.sql import SparkSession
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from itertools import chain, combinations
    import seaborn as sns
    import calendar
    from statsmodels.tsa.arima.model import ARIMA
    from datetime import datetime
    from pyspark.dbutils import DBUtils
    import re
    from sklearn.metrics import mean_squared_error, r2_score
    import pyodbc
    import py4j

    import pandas as pd
    import numpy as np
    from pyspark.sql.types import *
    from pyspark.sql.functions import from_json, col, explode, collect_list

except Exception as e:
    error_message = str(e)
    # print(f"There is an issue during library instalation : {error_message}")

# Initialize Spark session (Databricks environment should have this pre-configured)
spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()

dbutils = DBUtils(spark)
# Retrieve parameter passed to the notebook
# Get Forecasters input from Databricks task id
databrick_task_id = dbutils.widgets.get("DatabrickTaskID")

# print(databrick_task_id)


debug = False

# COMMAND ----------


# Read dataset from SQL Server
server_name = "jdbc:sqlserver://fortrack-maz-sdb-san-qa-01.database.windows.net"
database_name = "FortrackDB"
url = server_name + ";" + "databaseName=" + database_name + ";"
table_DBT = "dbo.DataBrickTasks"
table_UFM = "dbo.UserForecastMethod"

table_actual = "dbo.ActualData"
table_version = "dbo.DimVersion"
table_forecast = "dw.ForecastActive"
performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"

write_url = "jdbc:sqlserver://fortrack-maz-sdb-san-qa-01.database.windows.net;databaseName=FortrackDB"
write_properties = {
    "user": "fortrackSQL",
    "password": "vuxpapyvu@2024",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Define the name of the target table

user = "fortrackSQL"
password = "vuxpapyvu@2024"

# COMMAND ----------

# Get Forecasters input from Databricks task id
# Define a SQL query to retrieve various forecasting details associated with a specific Databricks task.
# This includes forecast methods, customer details, regional dataset, and task execution status.

# ufm.Name

query = f"""
SELECT TOP 1
    ufm.StartDate,
    ufm.EndDate,
    ufm.Parameters,
    ufm.Region,
    ufm.Status,
    ufm.ForecastMethodID,
    ufm.UserForecastMethodID,
    ufm.JSONCustomer as CustomerJSON,
    ufm.varJSON,    
    dfm.Method,
    dbt.DatabrickID
FROM 
    [dbo].[DataBrickTasks] AS dbt
INNER JOIN 
    [dbo].[UserForecastMethod] AS ufm ON dbt.UserForecastMethodID = ufm.UserForecastMethodID
INNER JOIN 
    [dbo].[DimForecastMethod] AS dfm ON ufm.ForecastMethodID = dfm.ForecastMethodID

WHERE  dbt.DatabrickID={databrick_task_id}
ORDER BY
    dbt.CreationDate

"""
# print(query)


try:

    # Read dataset using Spark SQL by setting up the database connection and executing the SQL query.

    df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query) \
        .option("user", user) \
        .option("password", password) \
        .load()

    # Extract specific fields from the DataFrame, convert them to a Pandas DataFrame, and store in variables.

    Forecast_Method_Name = df.select("Method").toPandas().iloc[0]['Method']
    Model_Parmeters = df.select("Parameters").toPandas().iloc[0]['Parameters']
    UFMID = df.select("UserForecastMethodID").toPandas().iloc[0]['UserForecastMethodID']
    # CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']

    StartDate = df.select("StartDate").toPandas().iloc[0]['StartDate']
    EndDate = df.select("EndDate").toPandas().iloc[0]['EndDate']
    DatabrickID = df.select("DatabrickID").toPandas().iloc[0]['DatabrickID']
    Hyper_Parameters = df.select("Parameters").toPandas().iloc[0]['Parameters']

    if debug:
        display(df)

except py4j.protocol.Py4JJavaError as e:
    error_message = str(e)
    if "Unable to connect" in error_message or "Connection refused" in error_message:
        # print(f"Connection-related error occurred: {e}")
        raise ConnectionError(e)
        # print(f"Connection-related error occurred: {e}")
    elif "Login failed for user" in error_message:
        # print(f": {e}")
        raise ValueError(f"Authentication failed, the entered credentials may not be valid : {error_message}")
    elif "ClassNotFoundException" in error_message:
        # print(f"Error: JDBC Driver not found. Please ensure the driver is added to the classpath. : {e}")
        raise ModuleNotFoundError("JDBC Driver missing.")
    elif "SQLTimeoutException" in str(e):
        # print("Error: Connection or query timeout. Please check the query performance or network latency.")
        raise TimeoutError("SQL query or connection timeout occurred.")
    # dbutils.notebook.exit(f"Notebook execution stopped: {e}")

except IndexError as e:
    # print(f"Index out of range which means the query condition is not met. Please check input parameters in UFM table {e}")
    raise IndexError("Index out of range which means the query condition is not met. Please check input parameters ")

except Exception as e:
    error_message = str(e)
    if 'Login failed for user' in error_message:
        # print(f"Entered credentials are not valid : {error_message}")
        raise ValueError(f"Authentication failed, the entered credentials may not be valid : {error_message}")
    else:
        # print(f"Unexpected error occurred: {error_message}")
        raise Exception(f"Unexpected error occurred: {error_message}")

if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}")

# COMMAND ----------