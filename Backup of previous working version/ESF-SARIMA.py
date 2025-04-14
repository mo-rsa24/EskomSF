# Databricks notebook source
# Import necessary libraries
%pip install pmdarima
from pmdarima import auto_arima
# %pip install statsmodels

from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
# from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from pyspark.dbutils import DBUtils
import re
import pyodbc
import py4j
import pandas as pd
import numpy as np
from pyspark.sql.types import ArrayType, StructType, StructField, StringType
from pyspark.sql.functions import from_json, col, explode,collect_list
from sklearn.metrics import mean_squared_error, r2_score
 
# from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from pyspark.dbutils import DBUtils
import re

import pyodbc

import pandas as pd
import numpy as np

from pyspark.sql.types import ArrayType, StructType, StructField, StringType 
from pyspark.sql.functions import from_json, col, explode,collect_list

# Initialize Spark session (Databricks environment should have this pre-configured)
spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()

dbutils = DBUtils(spark)
# Retrieve parameter passed to the notebook
# Get Forecasters input from Databricks task id
databrick_task_id = dbutils.widgets.get("DatabrickTaskID")

print(databrick_task_id)


debug = True


# COMMAND ----------


# Read data from SQL Server
server_name = "jdbc:sqlserver://fortrack-maz-sdb-san-qa-01.database.windows.net"
database_name = "FortrackDB"
# url = server_name + ";" + "databaseName=" + database_name + ";"
url = server_name + ";" + "database=" + database_name + ";"
table_DBT = "dbo.DataBrickTasks"
table_UFM = "dbo.UserForecastMethod"

table_actual = "dbo.ActualData" 
table_version = "dbo.DimVersion"
table_forecast = "dw.ForecastActive"

table_pod_aggrgated = "test.PodEnergyUsageRaw03";
performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"

table_DimPod = "dbo.DIMPOD"



# Define the properties for the database connection
write_url = "jdbc:sqlserver://fortrack-maz-sdb-san-qa-01.database.windows.net;databaseName=FortrackDB"
write_properties = {
    "user": "arul",
    "password": "aari@Singds.8734",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

user = "arul"
password = "aari@Singds.8734"



# COMMAND ----------

# Get Forecasters input from Databricks task id
# Define a SQL query to retrieve various forecasting details associated with a specific Databricks task.
# This includes forecast methods, customer details, regional data, and task execution status.

error_message=None

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

WHERE  dbt.DatabrickID={databrick_task_id} and ufm.ForecastMethodID = 2 and dbt.ExecutionStatus='In Progress' 

ORDER BY
    dbt.CreationDate

"""
print(query)

# WHERE ExecutionStatus IN ('In Progress')

# Read data using Spark SQL by setting up the database connection and executing the SQL query.

# if debug:
#     print(write_url)
try:

    df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query) \
        .option("user", user) \
        .option("password", password) \
        .load()


    print(f"Connecting to server {server_name}")
    Forecast_Method_Name = df.select("Method").toPandas().iloc[0]['Method']
    Model_Parmeters=df.select("Parameters").toPandas().iloc[0]['Parameters']
    UFMID=df.select("UserForecastMethodID").toPandas().iloc[0]['UserForecastMethodID']
    # CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']

    StartDate=df.select("StartDate").toPandas().iloc[0]['StartDate']
    EndDate=df.select("EndDate").toPandas().iloc[0]['EndDate']
    DatabrickID=df.select("DatabrickID").toPandas().iloc[0]['DatabrickID']
    Hyper_Parameters=df.select("Parameters").toPandas().iloc[0]['Parameters']


    print(f"The current job is running for Forecast_Method_Name {Forecast_Method_Name},with Forecast Method Name {UFMID}, DatabrickID {DatabrickID}")
    if debug:
        display(df)

except py4j.protocol.Py4JJavaError as e:
        error_message = str(e)
        if "Unable to connect" in error_message or "Connection refused" in error_message:
            print(f"Connection-related error occurred: {e}")
        elif "Login failed for user" in error_message:
             print(f"Entered credentials are not valid : {e}")
        elif "ClassNotFoundException" in error_message:
             print(f"Error: JDBC Driver not found. Please ensure the driver is added to the classpath. : {e}")    
        elif "SQLTimeoutException" in str(e):
            print("Error: Connection or query timeout. Please check the query performance or network latency.")
        # dbutils.notebook.exit(f"Notebook execution stopped: {e}")
        
except Exception as e:
        error_message = str(e)
        if "single positional indexer is out-of-bounds" in error_message:
            print(f"Connection to the server is successfull. But an empty row is returned, as the Query condition is not met: {e}")
        else:
            print(f"Unexpected error occurred: {e}")


else:
    print("The submitted job details are read.")

# COMMAND ----------

if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}")


# COMMAND ----------


# json_schema = ArrayType(StructType([
#     StructField("CustomerID", StringType(), True)
# ]))


# # df_cust.show()

# if "CustomerJSON" in df.columns:
#     json_schema = ArrayType(StructType([
#         StructField("CustomerID", StringType(), True)
#     ]))

#     # Parse the JSON string into a column of arrays of structs
#     df_cust = df.withColumn("ParsedJSON", from_json("CustomerJSON", json_schema))\
#                 .select(explode("ParsedJSON").alias("CustomerDetails"))\
#                 .select(col("CustomerDetails.CustomerID"))



#     # Collect the IDs into a list
#     multiple_customer_ids_list = df_cust.agg(collect_list("CustomerID")).first()[0]

#     # Convert the list to a comma-separated string


#  # Convert the list to a comma-separated string
#     if multiple_customer_ids_list:
#         multiple_customer_ids_list = ",".join([f"'{customer_id}'" for customer_id in multiple_customer_ids_list])

#     else:
#         multiple_customer_ids_list = ''

#     # Output the comma-separated IDs
#     print("Comma-separated Customer IDs:")
#     print(multiple_customer_ids_list)

# else:
#     print("Column 'CustomerJSON' does not exist in the dataframe.")


# COMMAND ----------

try:
    print(df.columns)
    if "varJSON" in df.columns:
        print("True")
        VarJsonSchema=  StructType([
                                        StructField("VariableID",ArrayType(StringType()),True)
                                    ])

                                
        df_var=df.withColumn("ParsedVarJson",from_json("varJSON",VarJsonSchema))\
            .select(explode(col("ParsedVarJson.VariableID")).alias("VariableID"))
        all_variables=df_var.agg(collect_list("VariableID")).first()[0] 
        all_variables=','.join(all_variables)
    # if not all_variables:
    #     raise ValueError("No variables selected to predict or selected variables are not listed in correct format in job table.")
    # if  all_variables:
    #     raise ValueError("No variables selected to predict or selected variables are not listed in correct format in job table.")    

    # Ensure all_variables is a list
    if isinstance(all_variables, str):
        all_variables = all_variables.split(',')
except AnalysisException as e:
    print("Schema mismtach observed in the VarJSON column",e)

except Exception as e:

        print(f"Unexpected error occurred: {e}")





# COMMAND ----------

if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}")        

# COMMAND ----------


# Create DataFrame


# Sample AllVariables DataFrame
# all_variables = ["Peak"]

# Define the required columns mapping
columns_mapping = {
    frozenset(["PeakConsumption", "StandardConsumption", "OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption", "StandardConsumption", "PeakConsumption"],
    frozenset(["PeakConsumption", "StandardConsumption"]): ["ReportingMonth", "CustomerID", "StandardConsumption", "PeakConsumption"],
    frozenset(["PeakConsumption", "OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption", "PeakConsumption"],
    frozenset(["StandardConsumption", "OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption", "StandardConsumption"],
    frozenset(["PeakConsumption"]): ["ReportingMonth", "CustomerID", "PeakConsumption"],
    frozenset(["StandardConsumption"]): ["ReportingMonth", "CustomerID", "StandardConsumption"],
    frozenset(["OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption"]
}

# Convert AllVariables to a set for easy comparison
all_variables_set = set(all_variables)
print(all_variables_set)

# Find the matching key in the columns_mapping
matching_key = None
for key in columns_mapping.keys():
    # print(key)
    if key.issubset(all_variables_set):
        matching_key = key
        break

# Select the appropriate columns based on the matching key
if matching_key:
    selected_columns = columns_mapping[matching_key]

    print(selected_columns)
else:
    print("No matching columns found in AllVariables")

# COMMAND ----------

print(selected_columns[2:])

# COMMAND ----------


# Construct a SQL query to select all records from the actuals table for a specific customer.
query_act_cons = f"""
                     ( select * from dbo.PredictiveInputData({UFMID}))
                    """
 
print(query_act_cons)


try:
    # Load the data
    act_df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query_act_cons) \
        .option("user", user) \
        .option("password", password) \
        .load()

    # Check if the DataFrame is empty
    if act_df.rdd.isEmpty():
        raise ValueError("No historical data available for the selected customers.")
    
    # Proceed with further processing if data is available
    act_df = act_df.orderBy("PodID", "ReportingMonth")
    pandas_df = act_df.toPandas()

except ValueError as ve:
    # Handle the error for empty data
    print(f"Error: {ve}")
    dbutils.notebook.exit(f"Notebook execution stopped: {ve}")  # Stops notebook execution in Databricks



except Exception as e:
    # Handle other generic exceptions
    print(f"An unexpected error occurred: {e}")
    raise

# pandas_df = df.select(*selected_columns).toPandas()



pandas_df['CustomerID'] = pandas_df['CustomerID'].astype(str)


pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'],format ='%Y-%m-%d').dt.to_period('M').dt.to_timestamp()



print(query_act_cons)

# COMMAND ----------

print(pandas_df.head())

multiple_customer_ids_list = pandas_df['CustomerID'].unique()
if len(multiple_customer_ids_list)>0:
    # Output the comma-separated IDs
    print(f"Future consumption will be predicted for customers:{multiple_customer_ids_list}")
else:
    # Output the comma-separated IDs
    print(f"No consumption data available for selected Customer IDs: {multiple_customer_ids_list} or customer ids selection is not successful")

# COMMAND ----------

if len(pandas_df) == 0:
    print("No data found for these customers")

# COMMAND ----------

# Find the most recent reporting month in the data, which will be used to determine the starting point for forecasting.
Actuals_last_date = pandas_df['ReportingMonth'].max() 


# Generate a date range from the start date to the end date with a monthly frequency, starting on the first of each month.
# This range represents the forecast period.

forecast_dates = pd.date_range(start=StartDate, end=EndDate, freq='MS')[0:]
print(forecast_dates,f"Actual_last_date {Actuals_last_date}",f"No of month to predict for {len(forecast_dates)}")

# COMMAND ----------


# Calculate n_periods as the number of months between the last historical date
# and the last date you want to forecast
n_periods = len(forecast_dates)

print(f"Number of periods to forecast: {n_periods}")


# COMMAND ----------

# validate if data is only processed for intended customer

unique_customers = pandas_df['CustomerID'].unique()
unique_PodIDs = pandas_df['PodID'].unique()

print(f"The customers {unique_customers} and PODs{unique_PodIDs} you predicting the consumption for ")


# COMMAND ----------

# Initialize default parameter sets
parameter_set1 = (0, 0, 0)  # Default value for ARIMA order
parameter_set2 = (0, 0, 0, 0)  # Default value for SARIMA seasonal_order

if Forecast_Method_Name == 'SARIMA':
# Extract parameters for SARIMA

# This regular expression finds tuples within parentheses.
# Splitting the string to isolate order and seasonal_order
    order, seasonal_order= re.findall(r'\(.*?\)',Hyper_Parameters)   #['(1,1,1),(1,1,1,2)']
    print(f"(Order_Paramaters {order}  Seasonal_Paramaters {seasonal_order}")  

# Remove parentheses and split the parameters by commas to extract individual elements.
order = order.strip('()')
order_parameters = order.split(',')

seasonal_order=seasonal_order.strip('()')
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

print(f"(Order_Paramaters {p,d,q}  Seasonal_Paramaters {s_p,s_d,s_q,s_m}")  


# COMMAND ----------

print(Forecast_Method_Name, n_periods,len(forecast_dates))

# COMMAND ----------



# COMMAND ----------


plot_columns = selected_columns[2:]   
# Define the name of the target table


def automated_forecasts_for_all_types(data,  selected_columns,n_periods=n_periods):

    if selected_columns is None:
        selected_columns = ["OffPeakConsumption", "StandardConsumption", "PeakConsumption"]

    all_forecasts = []

    # Determine which consumption types are in the selected columns
    consumption_types = ["OffPeakConsumption", "StandardConsumption", "PeakConsumption"]
    cons_types = [col for col in selected_columns if col in consumption_types]
    # print("cons_types: " + str(cons_types))

    # Iterate through each unique customer ID in the dataset.
    for customer_id in data['CustomerID'].unique():
        customer_forecasts = {}
        
        # Ensure data is sorted by ReportingMonth
        customer_data = data[data['CustomerID'] == customer_id].sort_values('PodID')
        unique_podel_ids = customer_data["PodID"].unique()

        print(f"Processing PODEL_ID: {unique_podel_ids}")

        for podel_id in unique_podel_ids:
            print(f"Processing PODEL_ID: {podel_id}")

            podel_df = customer_data[customer_data["PodID"] == podel_id].sort_values('ReportingMonth')
            print(podel_df.shape)
            future_predictions = []
            
            rmse_results = {}
            performance_data = {
                        'ModelName': 'SARIMA',
                        'CustomerID': str(customer_id),
                        'PodID': str(podel_id),
                        'DataBrickID': int(DatabrickID),   
                        'UserForecastMethodID': int(UFMID)
                }
            performance_full_data = pd.DataFrame({'ModelName':['SARIMA']})

            performance_full_data["CustomerID"] = str(customer_id)
            performance_full_data['PodID']      = str(podel_id)
            performance_full_data['DataBrickID']= int(DatabrickID) 
            performance_full_data['UserForecastMethodID']= int(UFMID)
            RMSE_sum = 0
            R2_sum = 0  
            forecast_df_data = {
                                 'ReportingMonth': forecast_dates,
                                 'CustomerID': [customer_id] * n_periods,
                                 'PodID': [podel_id] * n_periods
                                }                                 
            # Loop through each type of consumption to forecast individually.        
            for cons_type in cons_types:

                # Prepare the time series  by setting ReportingMonth as the index
                series = podel_df.set_index('ReportingMonth')[cons_type]
                series = pd.to_numeric(series, errors='coerce').fillna(0)
                log_series = np.log(series)

                forecast=None

                if Forecast_Method_Name == 'SARIMA':
                    # Fit SARIMA model
                    # model = SARIMAX(log_series, order=(1,2,1), seasonal_order=(1,2,1,3))               
                    #model = SARIMAX(series, order=(q,d,p), seasonal_order=(s_p,s_d,s_q,s_m))
                    model = auto_arima(log_series, 
                                       seasonal=True, 
                                       start_p=0, start_q=0,
                                       max_p=3, max_q=3,
                                       start_P=0, start_Q=0,
                                       max_P=3, max_Q=3,                                       
                                       m=12,   
                                       trace=True,
                                       error_action='ignore',
                                       suppress_warnings=True,
                                       stepwise=True,
                                       seasonal_test='ocsb',
                                       d=1, D=1)
                    # model = SARIMAX(log_series,
                    #                 order=(p, d, q),       # p, d, q
                    #                 seasonal_order=(s_p, s_d, s_q, s_m)
                    #                 )  # P, D, Q, s

                    model_fit = model.fit(log_series)
                    log_forecast = model_fit.predict(n_periods=n_periods)


                    forecast = np.exp(log_forecast)

                    print("Forecast done with SARIMA model")
                    
                    # actual_values = series.tail(-n_periods)
                    actual_values = series[-n_periods:]
                    print(f"len of actuals :{len(actual_values)} n_periods :{n_periods}") 
                    # if len(actual_values) == n_periods:

                    def mean_baseline(train_series):
                        mean_value = train_series.mean()
                        return np.full( len(actual_values), mean_value)
                    
                    baseline_predictions = mean_baseline(actual_values)
                    baseline_rmse = np.sqrt(mean_squared_error(actual_values, baseline_predictions))
                    baseline_r2   = r2_score(actual_values, baseline_predictions)

                    sarima_rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))
                    print(f"RMSE of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {sarima_rmse}")    
                    print(f"RMSE of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_rmse}")    
                    print(" ")
                    sarmia_r2 = r2_score(actual_values, forecast[:len(actual_values)])
                    print(f"R2 of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {sarmia_r2}")    
                    print(f"R2 of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_r2}")    
                                      


                    # if forecast is not None:
                    #  forecast[forecast < 0] = forecast*-1



                    customer_forecasts[cons_type] = forecast
                        # Construct a DataFrame to store the forecast results


                                    # Prepare data for performance metrics insertion
                    print("cons_type"+str(cons_type))
                    performance_data[f"RMSE_{cons_type}"] = sarima_rmse
                    performance_data[f"R2_{cons_type}"] = sarmia_r2

                    
                # if forecast is not None:
                # # Assuming actual_values and forecast are already defined
                #     sarima_rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))
                #     sarmia_r2 = r2_score(actual_values, forecast[:len(actual_values)])
                    display(performance_data)



                performance_full_data[f"RMSE_{cons_type}"] = performance_data[f"RMSE_{cons_type}"]
                performance_full_data[f"R2_{cons_type}"] = performance_data[f"R2_{cons_type}"]   



            for cons_type in cons_types:
                    customer_forecasts[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)      
                               
            for selected_cons_type in cons_types:
                                    # print("no of cons_type "+str(len(selected_cons_type)))
                print( cons_types)
                RMSE_sum +=  performance_full_data[f"RMSE_{selected_cons_type}"]
                R2_sum   +=  performance_full_data[f"R2_{selected_cons_type}"]
                rmse_avg = RMSE_sum/len(cons_types)
                r2_avg = R2_sum/len(cons_types)


            performance_full_data['RMSE_Avg'] = rmse_avg
            performance_full_data['R2_Avg'] = r2_avg   


            display(performance_full_data)              
            performance_spark_df = spark.createDataFrame(performance_full_data)
                # Write the performance metrics to the SQL table
            performance_spark_df.write.jdbc(url=write_url, table=performance_metrics_table, mode="append", properties=write_properties)

   


            for cons_type in cons_types:
                    forecast_df_data[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)

            forecast_df = pd.DataFrame(forecast_df_data)        
                # Append each customer's forecast DataFrame to a list.
            all_forecasts.append(forecast_df)
 

                    
        # Concatenate all individual forecast DataFrames into one.
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)


            # Round the consumption columns if they are present
        for cons_type in cons_types:
                    forecast_combined_df[cons_type] = forecast_combined_df[cons_type].fillna(0).round(2)

                # Return the combined forecast DataFrame.
        forecast_combined_spark_df = spark.createDataFrame(forecast_combined_df)

    #         # Define the properties for the database connection and write predicted results to DB
        forecast_combined_df['UserForecastMethodID'] = UFMID

    #         # Write the DataFrame to the SQL table
        forecast_combined_spark_df.write.jdbc(url=write_url, table=target_table_name, mode="append", properties=write_properties)                
        return forecast_combined_df



#     # Execute the forecasting function with the loaded data.
forecast_combined_df = automated_forecasts_for_all_types(pandas_df,selected_columns)


    # print(forecast_combined_df.describe())




# COMMAND ----------

# MAGIC %md
# MAGIC ==============================           Projection     ====================================