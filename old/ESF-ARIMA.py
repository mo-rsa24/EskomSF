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
    import  py4j

    import pandas as pd
    import numpy as np 
    from pyspark.sql.types import *
    from pyspark.sql.functions import from_json, col, explode,collect_list

except Exception as e:
    error_message = str(e)
    #print(f"There is an issue during library instalation : {error_message}")
    

# Initialize Spark session (Databricks environment should have this pre-configured)
spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()

dbutils = DBUtils(spark)
# Retrieve parameter passed to the notebook 
# Get Forecasters input from Databricks task id
databrick_task_id = dbutils.widgets.get("DatabrickTaskID")

#print(databrick_task_id)


debug = False


# COMMAND ----------


# Read data from SQL Server
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

WHERE  dbt.DatabrickID={databrick_task_id}
ORDER BY
    dbt.CreationDate

"""
#print(query)


try:

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


  
    if debug:
        display(df)        

except py4j.protocol.Py4JJavaError as e:
        error_message = str(e)
        if "Unable to connect" in error_message or "Connection refused" in error_message:
            #print(f"Connection-related error occurred: {e}")
            raise ConnectionError(e)
            #print(f"Connection-related error occurred: {e}")
        elif "Login failed for user" in error_message:
             #print(f": {e}")
             raise ValueError(f"Authentication failed, the entered credentials may not be valid : {error_message}")
        elif "ClassNotFoundException" in error_message:
             #print(f"Error: JDBC Driver not found. Please ensure the driver is added to the classpath. : {e}") 
             raise ModuleNotFoundError("JDBC Driver missing.")   
        elif "SQLTimeoutException" in str(e):
            #print("Error: Connection or query timeout. Please check the query performance or network latency.")
            raise TimeoutError("SQL query or connection timeout occurred.")
        # dbutils.notebook.exit(f"Notebook execution stopped: {e}")
        
except IndexError as e:
    #print(f"Index out of range which means the query condition is not met. Please check input parameters in UFM table {e}")
    raise IndexError("Index out of range which means the query condition is not met. Please check input parameters ")

except Exception as e:
    error_message = str(e)
    if 'Login failed for user' in error_message:
        #print(f"Entered credentials are not valid : {error_message}")
        raise ValueError(f"Authentication failed, the entered credentials may not be valid : {error_message}")
    else :
        #print(f"Unexpected error occurred: {error_message}")
        raise Exception(f"Unexpected error occurred: {error_message}")



if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}")



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
#     #print("Comma-separated Customer IDs:")
#     #print(multiple_customer_ids_list)

# else:
#     #print("Column 'CustomerJSON' does not exist in the dataframe.")


# COMMAND ----------

try:
    #print(df.columns)
    if "varJSON" in df.columns:
        #print("True")
        VarJsonSchema=  StructType([
                                        StructField("VariableID",ArrayType(StringType()),True)
                                    ])

                                
        df_var=df.withColumn("ParsedVarJson",from_json("varJSON",VarJsonSchema))\
            .select(explode(col("ParsedVarJson.VariableID")).alias("VariableID"))
        all_variables=df_var.agg(collect_list("VariableID")).first()[0] 
        all_variables=','.join(all_variables)

    #print(all_variables)

except ValueError as e:
    error_message = str(e)
    raise ValueError("Column 'varJSON' does not exist in the dataframe. or Schema of VarJSON is different")

except Exception as e:
        error_message = str(e)
        #print(f"Unexpected error occurred: {e}")

if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}") 

# Ensure all_variables is a list
if isinstance(all_variables, str):
    all_variables = all_variables.split(',')

# COMMAND ----------


# Create DataFrame


# Sample AllVariables DataFrame
# all_variables = ["Peak"]

# Define the required columns mapping

all_prediction_columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption","Block1Consumption", "Block2Consumption","Block3Consumption", "Block4Consumption",     "NonTOUConsumption"]

all_combination = chain.from_iterable(combinations(all_prediction_columns, r) for r in range(1,len(all_prediction_columns)+1))
columns_mapping = {}
for comb in all_combination:
    comb_set = frozenset(comb)
    comb_value = ['ReportingMonth', 'CustomerID'] + list(comb)
    columns_mapping[comb_set] = comb_value

# for key,value in columns_mapping.items():
#     #print(f" column_mapping key : {key} values : {value}")

# Convert AllVariables to a set for easy comparison
all_variables_set = set(all_variables)
#print("All selected variables :", str(all_variables_set))

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

    #print(f" All columns based on selectedcolumns{selected_columns}")
else:
    print("No matching columns found in AllVariables")

# COMMAND ----------


# Construct a SQL query to select all records from the actuals table for a specific customer.
query_act_cons = f"""
                     ( select * from dbo.PredictiveInputData({UFMID}))
                    """
 
#print(query_act_cons)
try:
    act_df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query_act_cons) \
        .option("user", user) \
        .option("password", password) \
        .load()

    pandas_df = act_df.orderBy("PodID", "ReportingMonth").toPandas()

    # Convert the Spark DataFrame to a Pandas DataFrame to use with time series analysis in Python.
    #print(pandas_df.head())
    if act_df.rdd.isEmpty():
        raise ValueError("No historical data available for the selected customers.")
        
    act_df = act_df.orderBy("PodID", "ReportingMonth")
    pandas_df = act_df.toPandas()
    
except ValueError as ve:
    # Handle the error for empty data
    #print(f"Error: {ve}")
    dbutils.notebook.exit(f"Notebook execution stopped: {ve}")  # Stops notebook execution in Databricks


except Exception as e:
    # Handle other generic exceptions
    #print(f"An unexpected error occurred: {e}")
    raise    
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
#print(forecast_dates,f"Actual_last_date {Actuals_last_date}",f"No of month to predict for {len(forecast_dates)}")

# COMMAND ----------


# Calculate n_periods as the number of months between the last historical date
# and the last date you want to forecast
n_periods = len(forecast_dates)

#print(f"Number of periods to forecast: {n_periods}")


# COMMAND ----------

# validate if data is only processed for intended customer
unique_customers = pandas_df['CustomerID'].unique()
#print(unique_customers)


# COMMAND ----------

# Initialize default parameter sets
parameter_set1 = (0, 0, 0)  # Default value for ARIMA order
parameter_set2 = (0, 0, 0, 0)  # Default value for SARIMA seasonal_order


# Check if the forecast method is ARIMA to extract and set the specific parameters for ARIMA from the string.
#print(Forecast_Method_Name)
if Forecast_Method_Name == 'ARIMA':
    # Extract parameters for ARIMA
    # order_str = re.findall(r'\(.?\)',Hyper_Parameters)   #[')', ')']
    # order_str = re.findall(r'\(*?\)',Hyper_Parameters)   #[]
    # order_str = re.findall(r'\(.*\)',Hyper_Parameters)   #['(1,1,1)(1,1,1,2)']


      # This regular expression finds tuples within parentheses.
    order= re.findall(r'\(.*?\)',Hyper_Parameters)   #['(1,1,1),(1,1,1,2)']
    #print(f"(Order_Paramaters {order}")  


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


#print(f"(Order_Paramaters {p,d,q}")  


# COMMAND ----------

#print(Forecast_Method_Name, n_periods,len(forecast_dates))

# COMMAND ----------

plot_columns = selected_columns[2:]   

def automated_forecasts_for_all_types(data,  selected_columns,n_periods=n_periods):
    if selected_columns is None:
        selected_columns = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]

    all_forecasts = []

    # Determine which consumption types are in the selected columns
    consumption_types = [
        "PeakConsumption", "StandardConsumption", "OffPeakConsumption",
        "Block1Consumption", "Block2Consumption", "Block3Consumption",
        "Block4Consumption", "NonTOUConsumption"
    ]
    


    

    # Determine which consumption types are in the selected columns
    

    cons_types = [col for col in selected_columns if col in consumption_types]
    #print(plot_columns)

    # Iterate through each unique customer ID in the dataset
    for customer_id in data['CustomerID'].unique():
        customer_forecasts = {}
        all_forecasts = []
        forecast_combined_df = []
        # Ensure data is sorted by ReportingMonth
        customer_df = data[data['CustomerID'] == customer_id].sort_values('PodID')
        if customer_df.empty:
            #print(f"No data found for CustomerID: {customer_id}")
            continue
        unique_podel_ids = customer_df["PodID"].unique()

        #print(f"Processing PODEL_ID: {unique_podel_ids}")
        for podel_id in unique_podel_ids:
            #print(f"Processing PODEL_ID: {podel_id}")

            podel_df = customer_df[customer_df["PodID"] == podel_id].sort_values('ReportingMonth')
            if podel_df.empty:
                #print(f"No data found for PODEL_ID: {podel_id}")
                continue
            performance_data = {
                'ModelName': 'ARIMA',
                'CustomerID': str(customer_id),
                'PodID': str(podel_id),
                'DataBrickID': int(DatabrickID),
                'UserForecastMethodID': int(UFMID)
            }

            performance_full_data = pd.DataFrame({'ModelName': ['ARIMA']})
            performance_full_data["CustomerID"] = str(customer_id)
            performance_full_data['PodID'] = str(podel_id)
            performance_full_data['DataBrickID'] = int(DatabrickID)
            performance_full_data['UserForecastMethodID'] = int(UFMID)
            RMSE_sum = 0
            R2_sum = 0
            rmse_avg = 0
            R2_avg = 0
            # Construct forecast DataFrame
            forecast_df_data = {
                'ReportingMonth': forecast_dates,
                'CustomerID': [customer_id] * n_periods,
                'PodID': [podel_id] * n_periods
            }

            # Loop through each type of consumption to forecast individually
            for cons_type in cons_types:
                #print(f"Processing Consumption Type: {cons_type}")
                try:
                    
                    if podel_df[cons_type].isnull().all():
                        #print(f"No data found for consumption type: {cons_type}")
                        continue
                    # Prepare the time series by setting ReportingMonth as the index
                    series = podel_df.set_index('ReportingMonth')[cons_type].astype(float)
                    series = pd.to_numeric(series, errors='coerce').fillna(0)
                    log_series = np.log(series)



                    # Fit ARIMA model
                    model = ARIMA(log_series, order=(p, d, q))
                    model_fit = model.fit()
                    log_forecast = model_fit.forecast(steps=n_periods)
                    forecast = np.exp(log_forecast)
                    #print("Forecast done with ARIMA model")

                    actual_values = series[-n_periods:]
                    baseline_predictions = np.full(len(actual_values), actual_values.mean())

                    baseline_rmse = np.sqrt(mean_squared_error(actual_values, baseline_predictions))
                    baseline_r2 = r2_score(actual_values, baseline_predictions)

                    # Ensure no NaN values before calculating RMSE
                    forecast_filled = np.nan_to_num(forecast[:len(actual_values)])
                    arima_rmse = np.sqrt(mean_squared_error(actual_values, forecast_filled))
                    arima_r2 = r2_score(actual_values, forecast_filled)

                    #print(f"RMSE of ARIMA for {cons_type} (PodID {podel_id}): {arima_rmse}")
                    #print(f"R2 of ARIMA for {cons_type} (PodID {podel_id}): {arima_r2}")

                    # Store results
                    performance_data[f"RMSE_{cons_type}"] = arima_rmse
                    performance_data[f"R2_{cons_type}"] = arima_r2
                    customer_forecasts[cons_type] = forecast



                except Exception as e:
                    #print(f"Error processing {cons_type} for PodID {podel_id}: {e}")
                    # performance_data[f"RMSE_{cons_type}"] = None
                    # performance_data[f"R2_{cons_type}"] = None
                    continue
                performance_full_data[f"RMSE_{cons_type}"] = performance_data[f"RMSE_{cons_type}"]
                performance_full_data[f"R2_{cons_type}"] = performance_data[f"R2_{cons_type}"]   

            for cons_type in cons_types:
                    customer_forecasts[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods) # Peak Consumption (col) -> pd.Series(numbers)

            # Calculate average RMSE and R2
            for selected_cons_type in cons_types:
                    RMSE_sum += performance_data.get(f"RMSE_{selected_cons_type}", 0) or 0
                    R2_sum += performance_data.get(f"R2_{selected_cons_type}", 0) or 0

            rmse_avg = RMSE_sum / len(cons_types)
            r2_avg = R2_sum / len(cons_types)
            performance_full_data['RMSE_Avg'] = rmse_avg
            performance_full_data['R2_Avg'] = r2_avg

            #display(performance_full_data)
            performance_spark_df = spark.createDataFrame(performance_full_data)
            performance_spark_df.write.jdbc(
                url=write_url,
                table=performance_metrics_table,
                mode="append",
                properties=write_properties
            )


            # #print(len(forecast_dates))
            # #print(len([customer_id] * n_periods))
            # #print(len([podel_id] * n_periods))
            for cons_type in cons_types:
                forecast_df_data[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)

            forecast_df = pd.DataFrame(forecast_df_data, index= range(n_periods))
            all_forecasts.append(forecast_df)

        # Combine forecasts
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)

        # Round the consumption columns
        for cons_type in cons_types:
            forecast_combined_df[cons_type] = forecast_combined_df[cons_type].fillna(0).round(2)
        forecast_combined_df['UserForecastMethodID'] = UFMID


        forecast_combined_spark_df = spark.createDataFrame(forecast_combined_df)

        #         # Define the properties for the database connection and write predicted results to DB


        #         # Write the DataFrame to the SQL table
        forecast_combined_spark_df.write.jdbc(url=write_url, table=target_table_name, mode="append", properties=write_properties)

    return forecast_combined_df



#     # Execute the forecasting function with the loaded data.
forecast_combined_df = automated_forecasts_for_all_types(pandas_df,selected_columns)


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
# #print(model_fit.summary())
# # Forecast the next 5 steps
# forecast = model_fit.forecast(steps=5)
# #print("Forecasted values:", forecast)

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
# #print("AR coefficients:", ar_params)

# # Get the AR coefficients
# ar = ar_params[:2]

# # Get the differenced series
# diff_series = series.diff().dropna()

# # Manually calculate the impact of the autoregressive terms for the last few observations
# manual_predictions = []
# for i in range(2, len(diff_series)):
#     prediction = ar[0] * diff_series.iloc[i-1] + ar[1] * diff_series.iloc[i-2]
#     manual_predictions.append(prediction)

# # #print the manual predictions
# #print("Manual predictions based on AR(2) terms:", manual_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ==============================           Projection     ====================================