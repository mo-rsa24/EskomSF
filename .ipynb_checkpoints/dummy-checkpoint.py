from itertools import chain, combinations
from autoarima import automated_forecasts_for_all_types

import re
import os

# SQL Server Details
server_name = "jdbc:sqlserver://fortrack-maz-sdb-san-qa-01.database.windows.net"
database_name = "FortrackDB"
url = f"{server_name};databaseName={database_name};"

# SQL Server Tables
table_DBT = "dbo.DataBrickTasks"
table_UFM = "dbo.UserForecastMethod"
table_actual = "dbo.ActualData"
table_version = "dbo.DimVersion"
table_forecast = "dw.ForecastActive"
performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"

# Authentication (Use environment variables for security)
user = os.getenv("DB_USER", "fortrackSQL")
password = os.getenv("DB_PASSWORD", "vuxpapyvu@2024")

# JDBC Driver
jdbc_driver = "com.microsoft.sqlserver.jdbc.SQLServerDriver"

import pyodbc
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=fortrack-maz-sdb-san-qa-01.database.windows.net;"
    "DATABASE=FortrackDB;"
    "UID=fortrackSQL;"
    "PWD=vuxpapyvu@2024;"
)
conn = pyodbc.connect(conn_str)

# Create a cursor
cursor = conn.cursor()

# Test the connection
cursor.execute("SELECT TOP 5 * FROM dbo.DataBrickTasks")
rows = cursor.fetchall()

# Print the results
for row in rows:
    print(row)

# Close the connection
conn.close()

databrick_task_id=39
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

import pandas as pd

conn = pyodbc.connect(conn_str)
df = pd.read_sql(query, conn)
df.head()

Forecast_Method_Name = df.iloc[0]['Method']
Model_Parmeters=df.iloc[0]['Parameters']
UFMID=df.iloc[0]['UserForecastMethodID']
# CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']

StartDate=df.iloc[0]['StartDate']
EndDate=df.iloc[0]['EndDate']
DatabrickID=df.iloc[0]['DatabrickID']
Hyper_Parameters=df.iloc[0]['Parameters']

import json

# Check if column exists
if "CustomerJSON" in df.columns:
    customer_ids = []

    # Iterate through each row
    for row in df["CustomerJSON"].dropna():
        try:
            json_list = json.loads(row)
            print("JSON List: ", json_list.items())
            for entry in json_list:
                print("Entry: ", json_list.get(entry))
                customer_ids.extend(json_list.get(entry))
        except json.JSONDecodeError:
            continue  # skip invalid JSON

    # Remove any None values and duplicates
    customer_ids = [cid for cid in customer_ids if cid]
    customer_ids = list(set(customer_ids))

    if customer_ids:
        multiple_customer_ids_str = ",".join([f"'{cid}'" for cid in customer_ids])
    else:
        multiple_customer_ids_str = ''

    print("Comma-separated Customer IDs:")
    print(multiple_customer_ids_str)
else:
    print("Column 'CustomerJSON' does not exist in the dataframe.")

if "varJSON" in df.columns:
    variable_ids = []

    for row in df["varJSON"].dropna():
        try:
            parsed = json.loads(row)
            print("Parsed: ", parsed)
            vars_list = parsed.get("VariableID", [])
            if vars_list:
                variable_ids.extend(vars_list)
        except json.JSONDecodeError:
            continue  # skip invalid JSON rows

    # Clean up: remove duplicates and empty entries
    variable_ids = [v for v in variable_ids if v]
    variable_ids = list(set(variable_ids))

    if variable_ids:
        all_variables = ",".join(variable_ids)
    else:
        all_variables = ""

    print("All VariableIDs as comma-separated string:")
    print(all_variables)

# Ensure all_variables is a list
if isinstance(all_variables, str):
    all_variables = all_variables.split(',')


all_prediction_columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption","Block1Consumption", "Block2Consumption","Block3Consumption", "Block4Consumption",     "NonTOUConsumption"]

all_combination = chain.from_iterable(combinations(all_prediction_columns, r) for r in range(1,len(all_prediction_columns)+1))
columns_mapping = {}
for comb in all_combination:
    comb_set = frozenset(comb)
    comb_value = ['ReportingMonth', 'CustomerID'] + list(comb)
    columns_mapping[comb_set] = comb_value


# Step 4: Print the result for testing
print(f"✅ Total combinations: {len(columns_mapping)}\n")

# Print 5 random combinations
import random
sample_keys = random.sample(list(columns_mapping.keys()), 5)

print("🔍 Sample mappings:")
for k in sample_keys:
    print(f"{set(k)} -> {columns_mapping[k]}")
    print()

all_variables_set = set(all_variables)
# print("All selected variables :", str(all_variables_set))

# Find the matching key in the columns_mapping
matching_key = None
# #print ("all_variables_set :", all_variables_set )
for key in columns_mapping.keys():
    # #print(key)
    # #print ("current key from columns mapping :", key )
    if key.issuperset(all_variables_set):
        matching_key = key
        break

# Select the appropriate columns based on the matching key
if matching_key:
    # #print(matching_key)
    selected_columns = columns_mapping[matching_key]

    # print(f" All columns based on selectedcolumns{selected_columns}")
else:
    print("No matching columns found in AllVariables")

df = pd.read_csv("PredictiveInputDataARIMA_.csv")
# --- Optional sorting ---
df = df.sort_values(by=["PodID", "ReportingMonth"])

# --- Type conversion ---
df['CustomerID'] = df['CustomerID'].astype(str)
df['ReportingMonth'] = pd.to_datetime(
    df['ReportingMonth'], format='%Y-%m-%d'
).dt.to_period('M').dt.to_timestamp()

# --- Display top rows ---
pandas_df = df.copy()
customer_forecasts = {}
multiple_customer_ids_list = pandas_df['CustomerID'].unique()
if len(multiple_customer_ids_list)>0:
    # Output the comma-separated IDs
    print(f"Future consumption will be predicted for customers:{multiple_customer_ids_list}")
else:
    # Output the comma-separated IDs
    print(f"No consumption dataset available for selected Customer IDs: {multiple_customer_ids_list} or customer ids selection is not successful")

# COMMAND ----------

if len(pandas_df) == 0:
    print("No dataset found for these customers")

# Find the most recent reporting month in the dataset, which will be used to determine the starting point for forecasting.
Actuals_last_date = pandas_df['ReportingMonth'].max()
forecast_dates = pd.date_range(start=StartDate, end=EndDate, freq='MS')[0:]
unique_customers = pandas_df['CustomerID'].unique()
unique_PodIDs = pandas_df['PodID'].unique()

order, seasonal_order = re.findall(r'\(.*?\)',Hyper_Parameters)   #['(1,1,1),(1,1,1,2)']

# Remove parentheses and split the parameters by commas to extract individual elements.
order = order.strip('()')
order_parameters = order.split(',')

seasonal_order = seasonal_order.strip('()')
seasonal_order_parameters = seasonal_order.split(',')

# Assign the values to the variables
p = int(order_parameters[0])
d = int(order_parameters[1])
q = int(order_parameters[2])

# Convert extracted string parameters into integers and assign them to corresponding variables.

s_p = int(seasonal_order_parameters[0])
s_d = int(seasonal_order_parameters[1])
s_q = int(seasonal_order_parameters[2])
s_m = int(seasonal_order_parameters[3])

# print(f"(Order_Paramaters {p,d,q}  Seasonal_Paramaters {s_p,s_d,s_q,s_m}")
n_periods = len(forecast_dates)
automated_forecasts_for_all_types(pandas_df, selected_columns,forecast_dates)