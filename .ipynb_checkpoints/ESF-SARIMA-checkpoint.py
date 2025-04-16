# Databricks notebook source
error_message = None

debug = True

try:
# Import necessary libraries
    %pip install pmdarima
    from pmdarima import auto_arima
    # %pip install statsmodels
    from itertools import chain, combinations
    from pyspark.sql import SparkSession
    # from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from datetime import datetime
    from pyspark.dbutils import DBUtils
    import py4j
    import re
    import pyodbc
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
except Exception as e:
    error_message = str(e)
    #print(f"There is an issue during library instalation : {error_message}")

# Initialize Spark session (Databricks environment should have this pre-configured)
spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()

dbutils = DBUtils(spark)
# Retrieve parameter passed to the notebook
# Get Forecasters input from Databricks task id
databrick_task_id = dbutils.widgets.get("DatabrickTaskID")

##print(databrick_task_id)



# COMMAND ----------


# Read dataset from SQL Server
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
    "user": "fortrackSQL",
    "password": "vuxpapyvu@2024",
    "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

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
##print(query)

# WHERE ExecutionStatus IN ('In Progress')
try:
# Read dataset using Spark SQL by setting up the database connection and executing the SQL query.
    df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query) \
        .option("user", user) \
        .option("password", password) \
        .load()

    # Assuming you need to convert the Spark DataFrame to a Pandas DataFrame
    # If the resulting DataFrame from your query matches what you want to convert to Pandas, you can do so directly
    # Extract specific fields from the DataFrame, convert them to a Pandas DataFrame, and store in variables.

    Forecast_Method_Name = df.select("Method").toPandas().iloc[0]['Method']
    Model_Parmeters=df.select("Parameters").toPandas().iloc[0]['Parameters']
    UFMID=df.select("UserForecastMethodID").toPandas().iloc[0]['UserForecastMethodID']
    # CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']

    StartDate=df.select("StartDate").toPandas().iloc[0]['StartDate']
    EndDate=df.select("EndDate").toPandas().iloc[0]['EndDate']
    DatabrickID=df.select("DatabrickID").toPandas().iloc[0]['DatabrickID']
    Hyper_Parameters=df.select("Parameters").toPandas().iloc[0]['Parameters']


    # #print(f" Hyper parameters values are {n_estimators, max_depth, learning_rate, subsample, colsample_bytree}")
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

if "CustomerJSON" in df.columns:
    json_schema = ArrayType(StructType([
        StructField("CustomerID", StringType(), True)
    ]))

    # Parse the JSON string into a column of arrays of structs
    df_cust = df.withColumn("ParsedJSON", from_json("CustomerJSON", json_schema))\
                .select(explode("ParsedJSON").alias("CustomerDetails"))\
                .select(col("CustomerDetails.CustomerID"))



    # Collect the IDs into a list
    multiple_customer_ids_list = df_cust.agg(collect_list("CustomerID")).first()[0]

    # Convert the list to a comma-separated string


 # Convert the list to a comma-separated string
    if multiple_customer_ids_list:
        multiple_customer_ids_list = ",".join([f"'{customer_id}'" for customer_id in multiple_customer_ids_list])

    else:
        multiple_customer_ids_list = ''

    # Output the comma-separated IDs
    ##print("Comma-separated Customer IDs:")
    ##print(multiple_customer_ids_list)

else:
    print("Column 'CustomerJSON' does not exist in the dataframe.")


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
    #print(e)
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

#print(selected_columns[2:])

# COMMAND ----------



# COMMAND ----------


# Construct a SQL query to select all records from the actuals table for a specific customer.
query_act_cons = f"""select * from dbo.PredictiveInputData({UFMID})"""
 
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
    # pandas_df = df.select(*selected_columns).toPandas()



    pandas_df['CustomerID'] = pandas_df['CustomerID'].astype(str)


    pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'],format ='%Y-%m-%d').dt.to_period('M').dt.to_timestamp()

    if act_df.rdd.isEmpty():
        raise ValueError("No historical dataset available for the selected customers.")

    #print(query_act_cons)
    
except ValueError as ve:
    # Handle the error for empty dataset
    #print(f"Error: {ve}")
    dbutils.notebook.exit(f"Notebook execution stopped: {ve}")  # Stops notebook execution in Databricks


except Exception as e:
    # Handle other generic exceptions
    print(f"An unexpected error occurred: {e}")
    raise e
    dbutils.notebook.exit(f"Notebook execution stopped: {e}")    
# pandas_df = df.select(*selected_columns).toPandas()
# pandas_df['CustomerID'] = pandas_df['CustomerID'].astype(str)


#pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'])
# pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp()


# COMMAND ----------

#print(pandas_df.head())

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

# COMMAND ----------

# Find the most recent reporting month in the dataset, which will be used to determine the starting point for forecasting.
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

# validate if dataset is only processed for intended customer

unique_customers = pandas_df['CustomerID'].unique()
unique_PodIDs = pandas_df['PodID'].unique()

#print(f"The customers {unique_customers} and PODs{unique_PodIDs} you predicting the consumption for ")


# COMMAND ----------

# Initialize default parameter sets
parameter_set1 = (0, 0, 0)  # Default value for ARIMA order
parameter_set2 = (0, 0, 0, 0)  # Default value for SARIMA seasonal_order

if Forecast_Method_Name == 'SARIMA':
# Extract parameters for SARIMA

# This regular expression finds tuples within parentheses.
# Splitting the string to isolate order and seasonal_order
    order, seasonal_order = re.findall(r'\(.*?\)',Hyper_Parameters)   #['(1,1,1),(1,1,1,2)']
    #print(f"(Order_Paramaters {order}  Seasonal_Paramaters {seasonal_order}")  

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

#print(f"(Order_Paramaters {p,d,q}  Seasonal_Paramaters {s_p,s_d,s_q,s_m}")  


# COMMAND ----------

#print(Forecast_Method_Name, n_periods,len(forecast_dates))

# COMMAND ----------



# COMMAND ----------


plot_columns = selected_columns[2:]   
# Define the name of the target table


def automated_forecasts_for_all_types(data,  selected_columns,n_periods=n_periods):

    if selected_columns is None:
        selected_columns = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]

    all_forecasts = []

    # Determine which consumption types are in the selected columns
    consumption_types = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption","Block1Consumption", "Block2Consumption","Block3Consumption", "Block4Consumption",     "NonTOUConsumption"]
   
    cons_types = [col for col in selected_columns if col in consumption_types]
    # #print("cons_types: " + str(cons_types))

    # Iterate through each unique customer ID in the dataset.
    for customer_id in data['CustomerID'].unique():
        customer_forecasts = {}
        
        # Ensure dataset is sorted by ReportingMonth
        customer_data = data[data['CustomerID'] == customer_id].sort_values('PodID')
        if customer_data.empty:
            #print(f"No dataset found for CustomerID: {customer_id}")
            continue
        unique_podel_ids = customer_data["PodID"].unique()

        #print(f"Processing PODEL_ID: {unique_podel_ids}")

        for podel_id in unique_podel_ids:
            #print(f"Processing PODEL_ID: {podel_id}")

            podel_df = customer_data[customer_data["PodID"] == podel_id].sort_values('ReportingMonth')
            # summary = podel_df[["OffPeakConsumption", "StandardConsumption", "PeakConsumption"]].agg(['sum'])
            # #print("Summary with total:")
            # #print(summary)
            if podel_df.empty:
                #print(f"No dataset found for PODEL_ID: {podel_id}")
                continue
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
            rmse_avg = 0
            R2_avg = 0            
            forecast_df_data = {
                                 'ReportingMonth': forecast_dates,
                                 'CustomerID': [customer_id] * n_periods,
                                 'PodID': [podel_id] * n_periods
                                }                                 
            # Loop through each type of consumption to forecast individually.        
            for cons_type in cons_types:
                #print(f"Processing Consumption Type: {cons_type}")
                try:

                    if podel_df[cons_type].isnull().all():
                        #print(f"No dataset found for consumption type: {cons_type}")
                        continue  
                 
                    # Prepare the time series  by setting ReportingMonth as the index
                    series = podel_df.set_index('ReportingMonth')[cons_type]
                    series = pd.to_numeric(series, errors='coerce').fillna(0)
                    log_series = np.log(series)
                    
                    # #print(log_series)
                    #print(f"cons_type: {cons_type}")


                    forecast=None


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
                                        d=1, D=0)
                     # model = SARIMAX(log_series,
                     #                 order=(p, d, q),       # p, d, q
                     #                 seasonal_order=(s_p, s_d, s_q, s_m)
                     #                 )  # P, D, Q, s

                    model_fit = model.fit(log_series)
                    log_forecast = model_fit.predict(n_periods=n_periods)


                    forecast = np.exp(log_forecast)

                    #print("Forecast done with SARIMA model",forecast)
                    
                    # actual_values = series.tail(-n_periods)
                    actual_values = series[-n_periods:]
                    #print(f"len of actuals :{len(actual_values)} n_periods :{n_periods}") 
                    # if len(actual_values) == n_periods:

                    def mean_baseline(train_series):
                        mean_value = train_series.mean()
                        return np.full( len(actual_values), mean_value)
                    
                    baseline_predictions = mean_baseline(actual_values)
                    baseline_rmse = np.sqrt(mean_squared_error(actual_values, baseline_predictions))
                    baseline_r2   = r2_score(actual_values, baseline_predictions)

                    sarima_rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))
                    sarmia_r2 = r2_score(actual_values, forecast[:len(actual_values)])
                    if debug:
                        print(f"RMSE of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {sarima_rmse}")    
                        print(f"RMSE of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_rmse}")    
                        #print(" ")
                        
                        #print(f"R2 of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {sarmia_r2}")    
                        #print(f"R2 of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_r2}")    
                                        


                    # if forecast is not None:
                    #  forecast[forecast < 0] = forecast*-1



                    customer_forecasts[cons_type] = forecast
                        # Construct a DataFrame to store the forecast results


                                    # Prepare dataset for performance metrics insertion
                    #print("cons_type"+str(cons_type))
                    performance_data[f"RMSE_{cons_type}"] = sarima_rmse
                    performance_data[f"R2_{cons_type}"] = sarmia_r2
                except Exception as e:
                    #print(f"Error processing {cons_type} for PodID {podel_id}: {e}")
                    continue
                # if forecast is not None:
                # # Assuming actual_values and forecast are already defined
                #     sarima_rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))
                #     sarmia_r2 = r2_score(actual_values, forecast[:len(actual_values)])




                performance_full_data[f"RMSE_{cons_type}"] = performance_data[f"RMSE_{cons_type}"]
                performance_full_data[f"R2_{cons_type}"] = performance_data[f"R2_{cons_type}"]   



            for cons_type in cons_types:
                    customer_forecasts[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)      
                               
            for selected_cons_type in cons_types:
                RMSE_sum += performance_data.get(f"RMSE_{selected_cons_type}", 0) or 0
                R2_sum += performance_data.get(f"R2_{selected_cons_type}", 0) or 0

            rmse_avg = RMSE_sum/len(cons_types)
            r2_avg = R2_sum/len(cons_types)
            performance_full_data['RMSE_Avg'] = rmse_avg
            performance_full_data['R2_Avg'] = r2_avg   


            # display(performance_full_data)              
            performance_spark_df = spark.createDataFrame(performance_full_data)
                # Write the performance metrics to the SQL table
            performance_spark_df.write.jdbc(url=write_url, table=performance_metrics_table, mode="append", properties=write_properties)

   


            for cons_type in cons_types:
                    forecast_df_data[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)

            forecast_df = pd.DataFrame(forecast_df_data, index= range(n_periods))    
                # Append each customer's forecast DataFrame to a list.
            all_forecasts.append(forecast_df)
            #print(forecast_df.head())

                    
        # Concatenate all individual forecast DataFrames into one.
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
        #display(forecast_combined_df)

            # Round the consumption columns if they are present
        for cons_type in cons_types:
                    forecast_combined_df[cons_type] = forecast_combined_df[cons_type].fillna(0).round(2)

                # Return the combined forecast DataFrame.


    #         # Define the properties for the database connection and write predicted results to DB
        forecast_combined_df['UserForecastMethodID'] = UFMID
        forecast_combined_spark_df = spark.createDataFrame(forecast_combined_df)
    #         # Write the DataFrame to the SQL table
        forecast_combined_spark_df.write.jdbc(url=write_url, table=target_table_name, mode="append", properties=write_properties)                
    # return forecast_combined_df



#     # Execute the forecasting function with the loaded dataset.
forecast_combined_df = automated_forecasts_for_all_types(pandas_df,selected_columns)


    # #print(forecast_combined_df.describe())




# COMMAND ----------

# MAGIC %md
# MAGIC ==============================           Projection     ====================================