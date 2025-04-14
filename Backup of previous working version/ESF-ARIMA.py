# Databricks notebook source
# Import necessary libraries
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import calendar
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from pyspark.dbutils import DBUtils
import re
from sklearn.metrics import mean_squared_error, r2_score
import pyodbc

import pandas as pd
import numpy as np 
from pyspark.sql.types import *
from pyspark.sql.functions import from_json, col, explode,collect_list


# Initialize Spark session (Databricks environment should have this pre-configured)
spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()

dbutils = DBUtils(spark)
# Retrieve parameter passed to the notebook 
# Get Forecasters input from Databricks task id
databrick_task_id = dbutils.widgets.get("DatabrickTaskID")

print(databrick_task_id)


debug = False


# COMMAND ----------


# Read data from SQL Server
server_name = "jdbc:sqlserver://esk-maz-sdb-san-dev-01.database.windows.net"
database_name = "ESK-MAZ-SDB-SAN-DEV-01"
url = server_name + ";" + "databaseName=" + database_name + ";"
table_DBT = "dbo.DataBrickTasks"
table_UFM = "dbo.UserForecastMethod"

table_actual = "dbo.ActualData"
table_version = "dbo.DimVersion"
table_forecast = "dw.ForecastActive"
performance_metrics_table = "dbo.StatisticalPerformanceMetrics"
target_table_name = "dbo.ForecastFact"

write_url = "jdbc:sqlserver://esk-maz-sdb-san-dev-01.database.windows.net;databaseName=ESK-MAZ-SDB-SAN-DEV-01"
write_properties = {
    "user": "arul",
    "password": "aari@Singds.8734",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

# Define the name of the target table

user = "arul"
password = "aari@Singds.8734"

 

# COMMAND ----------

# Get Forecasters input from Databricks task id
# Define a SQL query to retrieve various forecasting details associated with a specific Databricks task.
# This includes forecast methods, customer details, regional data, and task execution status.

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

WHERE  dbt.DatabrickID={databrick_task_id} and ufm.ForecastMethodID = 1 and  dbt.ExecutionStatus='In Progress' 

ORDER BY
    dbt.CreationDate

"""
print(query)



# Read data using Spark SQL by setting up the database connection and executing the SQL query.

df = spark.read \
    .format("jdbc") \
    .option("url", url) \
    .option("query", query) \
    .option("user", user) \
    .option("password", password) \
    .load()


# Extract specific fields from the DataFrame, convert them to a Pandas DataFrame, and store in variables.

Forecast_Method_Name = df.select("Method").toPandas().iloc[0]['Method']
Model_Parmeters=df.select("Parameters").toPandas().iloc[0]['Parameters']
UFMID=df.select("UserForecastMethodID").toPandas().iloc[0]['UserForecastMethodID']
# CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']

StartDate=df.select("StartDate").toPandas().iloc[0]['StartDate']
EndDate=df.select("EndDate").toPandas().iloc[0]['EndDate']
DatabrickID=df.select("DatabrickID").toPandas().iloc[0]['DatabrickID']
Hyper_Parameters=df.select("Parameters").toPandas().iloc[0]['Parameters']




# COMMAND ----------

print(df)

# COMMAND ----------

print(Forecast_Method_Name,UFMID,DatabrickID,Hyper_Parameters)

# COMMAND ----------

print(UFMID)

# COMMAND ----------


json_schema = ArrayType(StructType([
    StructField("CustomerID", StringType(), True)
]))


# df_cust.show()

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
#         multiple_customer_ids_list = ','.join(multiple_customer_ids_list)
#     else:
#         multiple_customer_ids_list = ''

#     # Output the comma-separated IDs
#     print("Comma-separated Customer IDs:")
#     print(multiple_customer_ids_list)

# else:
#     print("Column 'CustomerJSON' does not exist in the dataframe.")


# COMMAND ----------

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

print(all_variables)

# Ensure all_variables is a list
if isinstance(all_variables, str):
    all_variables = all_variables.split(',')

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


# Construct a SQL query to select all records from the actuals table for a specific customer.
query_act_cons = f"""
                     ( select * from dbo.PredictiveInputData({UFMID}))
                    """
 
print(query_act_cons)

act_df = spark.read \
    .format("jdbc") \
    .option("url", url) \
    .option("query", query_act_cons) \
    .option("user", user) \
    .option("password", password) \
    .load()

pandas_df = act_df.orderBy("PodID", "ReportingMonth").toPandas()

# Convert the Spark DataFrame to a Pandas DataFrame to use with time series analysis in Python.
print(pandas_df.head())

# pandas_df = df.select(*selected_columns).toPandas()
# pandas_df['CustomerID'] = pandas_df['CustomerID'].astype(str)


#pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'])
# pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp()



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
print(unique_customers)


# COMMAND ----------

# Initialize default parameter sets
parameter_set1 = (0, 0, 0)  # Default value for ARIMA order
parameter_set2 = (0, 0, 0, 0)  # Default value for SARIMA seasonal_order


# Check if the forecast method is ARIMA to extract and set the specific parameters for ARIMA from the string.

if Forecast_Method_Name == 'ARIMA':
    # Extract parameters for ARIMA
    # order_str = re.findall(r'\(.?\)',Hyper_Parameters)   #[')', ')']
    # order_str = re.findall(r'\(*?\)',Hyper_Parameters)   #[]
    # order_str = re.findall(r'\(.*\)',Hyper_Parameters)   #['(1,1,1)(1,1,1,2)']


      # This regular expression finds tuples within parentheses.
    order= re.findall(r'\(.*?\)',Hyper_Parameters)   #['(1,1,1),(1,1,1,2)']
    print(f"(Order_Paramaters {order}")  


    # order = order.strip('()')
    # order_parameters = order.split(',')


    # Remove parentheses and split the parameters by commas to extract individual elements.

    order = order[0].strip('()')  # Access the first element and convert it to a string
    order_parameters = order.split(',')

    # Convert extracted string parameters into integers and assign them to corresponding variables. 
 
    # p = int(order_parameters[0])
    # d = int(order_parameters[1])
    # q = int(order_parameters[2])

    p = 2
    d = 1
    q = 0


print(f"(Order_Paramaters {p,d,q}")  


# COMMAND ----------

print(Forecast_Method_Name, n_periods,len(forecast_dates))

# COMMAND ----------

plot_columns = selected_columns[2:]   

def plot_forecast_vs_historical(historical_df, forecast_df, features):
    """
    Plots historical vs forecasted values for specified features.
    
    Parameters:
    - historical_df: DataFrame containing historical data
    - forecast_df: DataFrame containing forecasted data
    - features: List of feature names to plot
    """
    # Convert ReportingMonth to datetime at end of month
    historical_df['ReportingMonth'] = pd.to_datetime(historical_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp('M')
    forecast_df['ReportingMonth'] = pd.to_datetime(forecast_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp('M')

    print(historical_df['ReportingMonth'])
    # Recalculate last_historical_date after conversion
    last_historical_date = historical_df['ReportingMonth'].max()

    if debug:
        print(f"Last Historical Date: {last_historical_date}")
        print(f"Forecast Dates Start: {forecast_df['ReportingMonth'].min()}")

    # Ensure consumption columns are numeric and handle NaNs
    for feature in features:
        historical_df[feature] = pd.to_numeric(historical_df[feature], errors='coerce').fillna(0)
        forecast_df[feature] = pd.to_numeric(forecast_df[feature], errors='coerce').fillna(0)

    # Filter forecast data to only include periods after the last historical date
    forecast_df = forecast_df[forecast_df['ReportingMonth'] > last_historical_date]

    # Group data by 'ReportingMonth'
    historical_grouped = historical_df.groupby('ReportingMonth')[features].mean().reset_index()
    forecast_grouped = forecast_df.groupby('ReportingMonth')[features].mean().reset_index()


    # Add check for empty dataframes
    if historical_grouped.empty or forecast_grouped.empty:
        print("No data available for plotting.")
        return

    # Set plot size
    num_features = len(features)
    fig, axs = plt.subplots(num_features, 1, figsize=(16, 5 * num_features), sharex=True)
    if num_features == 1:
        axs = [axs]

    for idx, feature in enumerate(features):
        ax = axs[idx]

        # Plot Historical Data (up to the last available month)
        ax.plot(historical_grouped['ReportingMonth'], historical_grouped[feature],
                label='Historical', color='blue', linewidth=2)

        # Plot Forecast Data (starting from the first forecasted month)
        ax.plot(forecast_grouped['ReportingMonth'], forecast_grouped[feature],
                label='Forecast', color='orange', linewidth=2)

        # Add vertical line for Forecast Start
        ax.axvline(x=last_historical_date, color='red', linestyle='--', label='Forecast Start')

        # Add grid, labels, and legend
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(feature, fontsize=14)
        ax.set_ylabel(feature, fontsize=12)
        ax.legend(fontsize=12)
        ax.tick_params(axis='x', rotation=30)

        # Improve date formatting on x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # every 3 months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Set common x-label
    axs[-1].set_xlabel('Date', fontsize=12)

    plt.tight_layout()
    plt.show()

def automated_forecasts_for_all_types(data,  selected_columns,n_periods=n_periods):
    if selected_columns is None:
        selected_columns = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]

    all_forecasts = []

    # Determine which consumption types are in the selected columns
    consumption_types = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]
    cons_types = [col for col in selected_columns if col in consumption_types]


    print(plot_columns)  
    # Iterate through each unique customer ID in the dataset.
    for customer_id in data['CustomerID'].unique():
        customer_forecasts = {}
        forecast_combined_df =[]
        # Ensure data is sorted by ReportingMonth
        customer_df = data[data['CustomerID'] == customer_id].sort_values('PodID')
        unique_podel_ids = customer_df["PodID"].unique()

        print(f"Processing PODEL_ID: {unique_podel_ids}")
        for podel_id in unique_podel_ids:
                print(f"Processing PODEL_ID: {podel_id}")

                podel_df = customer_df[customer_df["PodID"] == podel_id].sort_values('ReportingMonth')
                future_predictions = []
                all_forecasts = []
                rmse_results = {}
                performance_data = {
                        'ModelName': 'ARIMA',
                        'CustomerID': str(customer_id),
                        'PodID': str(podel_id),
                        'DataBrickID': int(DatabrickID),   
                        'UserForecastMethodID': int(UFMID)
                }
                performance_full_data = pd.DataFrame({'ModelName':['ARIMA']})

                performance_full_data["CustomerID"] = str(customer_id)
                performance_full_data['PodID']      = str(podel_id)
                performance_full_data['DataBrickID']= int(DatabrickID) 
                performance_full_data['UserForecastMethodID']= int(UFMID)
                RMSE_sum = 0
                R2_sum = 0
         # Loop through each type of consumption to forecast individually.
                for cons_type in cons_types:

                    # Prepare the time series  by setting ReportingMonth as the index
                    series = podel_df.set_index('ReportingMonth')[cons_type].astype(float)
                    series = pd.to_numeric(series, errors='coerce').fillna(0)
                    # print(series.isna().sum())
                    log_series = np.log(series)
                    forecast=None


                    # Fit ARIMA model
                    model = ARIMA(log_series, order=(p,d,q))
                    model_fit = model.fit()
                    log_forecast = model_fit.forecast(steps=n_periods)
                    forecast    = np.exp(log_forecast)
                    print("Forecast done with ARIMA model")

                    actual_values = series[-n_periods:]
                    # display(actual_values)
                    print(f"len of actuals :{len(actual_values)} n_periods :{n_periods}") 
                    # if len(actual_values) == n_periods:

                    def mean_baseline(train_series):
                        mean_value = train_series.mean()
                        return np.full( len(actual_values), mean_value)                     
                    baseline_predictions = mean_baseline(actual_values)
                    baseline_rmse = np.sqrt(mean_squared_error(actual_values, baseline_predictions))
                    baseline_r2   = r2_score(actual_values, baseline_predictions)


                    # # Ensure no NaN values before calculating RMSE
                    actual_values_filled = actual_values.fillna(0)
                    forecast_filled = np.nan_to_num(forecast[:len(actual_values)])  # Replace NaNs with 0



 
                    arima_rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))
                    print(f"RMSE of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {arima_rmse}")    
                    print(f"RMSE of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_rmse}")    
                    print(" ")
                    armia_r2 = r2_score(actual_values, forecast[:len(actual_values)])
                    print(f"R2 of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {armia_r2}")    
                    print(f"R2 of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_r2}")     
                                    
 
 

                    # if forecast is not None:
                    #  forecast[forecast < 0] = forecast*-1


                    # Store forecasts for each type of consumption in a dictionary.
                    customer_forecasts[cons_type] = forecast

        # Construct a DataFrame to store the forecast results.
        

                        # Construct a DataFrame to store the forecast results
                    forecast_df_data = {
                                                'ReportingMonth': forecast_dates,
                                                'CustomerID': [customer_id] * n_periods,
                                                'PodID': [podel_id] * n_periods
                                            }
                    print("cons_type"+str(cons_type))
                    performance_data[f"RMSE_{cons_type}"] = arima_rmse
                    performance_data[f"R2_{cons_type}"] = armia_r2

   
 


                    


                    performance_full_data[f"RMSE_{cons_type}"] = performance_data[f"RMSE_{cons_type}"]
                    performance_full_data[f"R2_{cons_type}"] = performance_data[f"R2_{cons_type}"]    




                               
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
    return forecast_combined_df



#     # Execute the forecasting function with the loaded data.
forecast_combined_df = automated_forecasts_for_all_types(pandas_df,selected_columns)
forecast_combined_df['UserForecastMethodID'] = UFMID

print(forecast_combined_df.describe())
forecast_combined_spark_df = spark.createDataFrame(forecast_combined_df)

#         # Define the properties for the database connection and write predicted results to DB


#         # Write the DataFrame to the SQL table
forecast_combined_spark_df.write.jdbc(url=write_url, table=target_table_name, mode="append", properties=write_properties)


    # Plot forecast vs historical data




# COMMAND ----------



# COMMAND ----------

# import pandas as pd
# from statsmodels.tsa.arima.model import ARIMA
# import matplotlib.pyplot as plt

# # Example time series data
# data = {'Time': pd.date_range(start='2020-01-01', periods=10, freq='M'),
#         'Value': [112, 118, 132, 129, 121, 135, 148, 145, 150, 156]}
# df = pd.DataFrame(data)

# # Plot the time series
# plt.plot(df['Time'], df['Value'])
# plt.title('Time Series Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()

# series = df['Value']
# # Define the ARIMA model
# model = ARIMA(series, order=(10, 1, 2))

# # Fit the ARIMA model
# model_fit = model.fit()

# # Summary of the model
# print(model_fit.summary())
# # Forecast the next 5 steps
# forecast = model_fit.forecast(steps=5)
# print("Forecasted values:", forecast)

# # Plot the original series and the forecast
# plt.plot(df['Time'], df['Value'], label='Original')
# plt.plot(pd.date_range(start='2020-11-01', periods=5, freq='M'), forecast, label='Forecast', color='red')
# plt.title('Time Series Forecast')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# # Extract the coefficients
# ar_params = model_fit.arparams
# print("AR coefficients:", ar_params)

# # Get the AR coefficients
# ar = ar_params[:2]

# # Get the differenced series
# diff_series = series.diff().dropna()

# # Manually calculate the impact of the autoregressive terms for the last few observations
# manual_predictions = []
# for i in range(2, len(diff_series)):
#     prediction = ar[0] * diff_series.iloc[i-1] + ar[1] * diff_series.iloc[i-2]
#     manual_predictions.append(prediction)

# # Print the manual predictions
# print("Manual predictions based on AR(2) terms:", manual_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ==============================           Projection     ====================================