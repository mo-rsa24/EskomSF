# Databricks notebook source
error_message = None
try:
    #%restart_python
    %pip install seaborn
    %pip install miceforest
    spark.sparkContext.setLogLevel("ERROR")

    # Import necessary libraries
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from pyspark.sql.functions import col 
    from pyspark.dbutils import DBUtils
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType,LongType

    from datetime import datetime
    import pyodbc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import calendar
    import py4j
    from itertools import chain, combinations

    import seaborn as sns
    # %pip install sweetviz
    # import sweetviz as sv

    import miceforest as mf
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import accuracy_score,r2_score,make_scorer,mean_absolute_error,mean_squared_error
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score



    from pyspark.sql.types import ArrayType, StructType, StructField, StringType 
    from pyspark.sql.functions import from_json, col, explode,collect_list,trim,regexp_replace
except Exception as e:
    error_message = str(e)
    #print(f"There is an issue during library instalation : {error_message}")

dbutils = DBUtils(spark)
debug = True

# Retrieve parameter passed to the notebook 
# Get Forecasters input from Databricks task id
databrick_task_id = dbutils.widgets.get("DatabrickTaskID")

#print(databrick_task_id)


# Initialize Spark session (Databricks environment should have this pre-configured)
spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()

# Read data from SQL Server


# Read data from SQL Server
server_name = "jdbc:sqlserver://fortrack-maz-sdb-san-qa-01.database.windows.net"
database_name = "FortrackDB"
url = server_name + ";" + "databaseName=" + database_name + ";"
table_DBT = "dbo.DataBrickTasks"
table_UFM = "dbo.UserForecastMethod"

table_actual = "dbo.ActualData"
table_version = "dbo.DimVersion"
table_forecast = "dw.ForecastActive"
table_performance_Metrics = "dbo.StatisticalPerformanceMetrics"
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

debug = False

 

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
# In Progress


# Read data using Spark SQL by setting up the database connection and executing the SQL query.


# if debug:
#     #print(write_url)
try:
    df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query) \
        .option("user", user) \
        .option("password", password) \
        .load()


    Forecast_Method_Name = df.select("Method").toPandas().iloc[0]['Method']
    Model_Parmeters=df.select("Parameters").toPandas().iloc[0]['Parameters']
    UFMID=df.select("UserForecastMethodID").toPandas().iloc[0]['UserForecastMethodID']
    # CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']

    StartDate=df.select("StartDate").toPandas().iloc[0]['StartDate']
    EndDate=df.select("EndDate").toPandas().iloc[0]['EndDate']
    DatabrickID=df.select("DatabrickID").toPandas().iloc[0]['DatabrickID']
    Hyper_Parameters=df.select("Parameters").toPandas().iloc[0]['Parameters']
    
    #print(f"The current job is running for Forecast_Method_Name {Forecast_Method_Name},with Forecast Method Name {UFMID}, DatabrickID {DatabrickID}")
    # if debug:
    #     display(df)

    #n_estimators = int(Hyper_Parameters.strip("()"))    

    #if debug:
        #print(f" Hyper parameters values are {n_estimators}")
 

 

# n_estimators = int(Hyper_parameter_values[0].replace('(', ''))
# max_depth = int(Hyper_parameter_values[1])
# learning_rate = float(Hyper_parameter_values[2])
# subsample = float(Hyper_parameter_values[3])
# # colsample_bytree =int(Hyper_parameter_values[4].replace('(', ''))
# colsample_bytree =float(Hyper_parameter_values[4].replace(')', ''))

# min_child_weight = int(float(Hyper_parameter_values[5].replace(')', '')))
# min_child_weight = int(float(Hyper_parameter_values[5].replace(')', '')))


# #print(f" Hyper parameters values are {n_estimators, max_depth, learning_rate, subsample, colsample_bytree}")

 
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
    #print(f"Index out of range which means the query condition is not met. Please check input parameters in UFM table  {e}")
    raise IndexError("Index out of range which means the query condition is not met. Please check input parameters ")

except Exception as e:
    error_message = str(e)
    # if 'Login failed for user' in error_message:
    #     #print(f"Entered credentials are not valid : {error_message}")
    # else :
    #     #print(f"Unexpected error occurred: {error_message}")


if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}")


json_schema = ArrayType(StructType([
    StructField("CustomerID", StringType(), True)
]))




# if debug:
#     #print(df.columns)
try:    
    if "varJSON" in df.columns:

        VarJsonSchema =  StructType([
                                        StructField("VariableID", ArrayType(StringType()), True)
                                    ])

                                



        df_var=df.withColumn("ParsedVarJson",from_json("varJSON",VarJsonSchema))\
            .select(explode(col("ParsedVarJson.VariableID")).alias("SelectedVarList"))
            #  .select(col("SelectedVarList.VariableID")).alias("VariableID")
        all_variables=df_var.agg(collect_list("SelectedVarList")).first()[0] 
        all_variables=','.join(all_variables)
        

    # Ensure all_variables is a list
    if isinstance(all_variables, str):
        all_variables = all_variables.split(',')


        
except ValueError as e:
    error_message = str(e)
    #print(e)

except Exception as e:
        error_message = str(e)
        #print(f"Unexpected error occurred: {e}")

if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}") 

    

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
#if debug:
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
    #if debug:
        #print(f" All columns based on selectedcolumns{selected_columns}")
#else:
    #print("No matching columns found in AllVariables")

try:
     if not selected_columns:
        raise ValueError("No consumption columns found in the selected columns or name of consumption columns mismatch")
except ValueError as e:
    error_message = str(e)

if error_message is not None:
    dbutils.notebook.exit(f"Notebook execution stopped: {error_message}") 

# Extract specific fields from the DataFrame, convert them to a Pandas DataFrame, and store in variables.


# COMMAND ----------


# Construct a SQL query to select all records from the actuals table for a specific customer.
query_act_cons = f"""
                     ( select * from dbo.PredictiveInputData({UFMID}))
                    """
 
if debug:
        print(query_act_cons)
try:
    act_df = spark.read \
        .format("jdbc") \
        .option("url", url) \
        .option("query", query_act_cons) \
        .option("user", user) \
        .option("password", password) \
        .load()
    pandas_df = act_df.toPandas()


    if act_df.rdd.isEmpty():
        raise ValueError("No historical data available for the selected customers.")

except ValueError as ve:
    # Handle the error for empty data
    #print(f"Error: {ve}")
    dbutils.notebook.exit(f"Notebook execution stopped: {ve}")  # Stops notebook execution in Databricks


except Exception as e:
    # Handle other generic exceptions
    #print(f"An unexpected error occurred: {e}")
    raise    

#print(f"Actual consumption data used from {pandas_df['ReportingMonth'].min()} to {pandas_df['ReportingMonth'].max()}")
multiple_customer_ids_list = pandas_df['CustomerID'].unique()
if len(multiple_customer_ids_list)>0:
    # Output the comma-separated IDs
    print(f"Future consumption will be predicted for customers:{multiple_customer_ids_list}")
else:
    # Output the comma-separated IDs
    print(f"No consumption data available for selected Customer IDs: {multiple_customer_ids_list} or customer ids selection is not successful")

# Convert Spark DataFrame to Pandas DataFrame for SARIMA
pandas_df['CustomerID'] = pandas_df['CustomerID'].astype(str)
pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'],format ='%Y-%m-%d').dt.to_period('M').dt.to_timestamp()
# Find the most recent reporting month in the data, which will be used to determine the starting point for forecasting.
Actuals_last_date = pandas_df['ReportingMonth'].max() 
# Generate a date range from the start date to the end date with a monthly frequency, starting on the first of each month.
# This range represents the forecast period.
forecast_dates = pd.date_range(start=StartDate, end=EndDate, freq='MS')[0:]
#print(forecast_dates,f"Actual_last_date {Actuals_last_date}",f"No of month to predict for {len(forecast_dates)}")
# Calculate n_periods as the number of months between the last historical date
# and the last date you want to forecast
n_periods = len(forecast_dates)
if debug:
    print(f"Number of periods to forecast: {n_periods}")
unique_customers = pandas_df['CustomerID'].unique()
unique_PodID = pandas_df[pandas_df['PodID'].notnull()]['PodID'].unique()
if debug:
    print(f"Unique Customers are {unique_customers}")
    print(f"Unique PODs are {unique_PodID}")
# Convert 'ReportingMonth' to datetime and extract features
#df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'])
pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'])
pandas_df['Month'] = pandas_df['ReportingMonth'].dt.month
pandas_df['Year'] = pandas_df['ReportingMonth'].dt.year
if debug:
    display(pandas_df[['ReportingMonth','Month','Year']])
    #print(pandas_df.head())


# COMMAND ----------

# Scaling the features
scaler = StandardScaler()
sim_imputer = SimpleImputer()

consumption_types = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption","Block1Consumption", "Block2Consumption","Block3Consumption", "Block4Consumption",     "NonTOUConsumption"]
cons_types = [col for col in selected_columns if col in consumption_types]

def create_lag_features(df, lag_columns, lags):
    for col in lag_columns:
            for lag in range(1,lags+1):
                df[f"{col}_lag{lag}"]=df[col].shift(lag)
    return df


# Create lag features for forecast data
def create_forecast_lag_features(df, original_df, lag_columns, lags, step):
    # Update the lag features for the forecast DataFrame using either historical or predicted data
    for col in lag_columns:
        for lag in range(1, lags + 1):
            if step == 0:
                # Use historical data to initialize lags
                df.loc[step, f"{col}_lag{lag}"] = original_df[col].iloc[-lag]
            else:
                # Use previous predictions to update lags
                df.loc[step, f"{col}_lag{lag}"] = df[col].iloc[step - lag]
    if debug:
        print(f"Forecast lag features created for step: {step}")
    return df


# Plot forecast vs historical data
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

    #print(historical_df['ReportingMonth'])
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

 

# Main forecasting loop
for customer_id in unique_customers:
    if debug:
        print(f"Processing CustomerID: {customer_id}")

    customer_df = pandas_df[pandas_df["CustomerID"] == customer_id]
    unique_podel_ids = customer_df["PodID"].unique()

    for podel_id in unique_podel_ids:
        if debug:
            print(f"Processing PODEL_ID: {podel_id}")

        podel_df = customer_df[customer_df["PodID"] == podel_id]
        
        forecast_df_all_cons = pd.DataFrame(forecast_dates, columns=["ReportingMonth"])
        forecast_df_all_cons["CustomerID"] = customer_id
        forecast_df_all_cons["PodID"] = podel_id
        forecast_df_all_cons["CustomerID"] = forecast_df_all_cons["CustomerID"].astype(int)
        forecast_df_all_cons["PodID"] = forecast_df_all_cons["PodID"].astype(int)
        forecast_df_all_cons["UserForecastMethodID"] = UFMID    


        # forecast_df_all_cons={}
        performance_data = { 'ModelName': 'Random Forest',
                            'CustomerID': str(customer_id),
                            'PodID': str(podel_id),
                            'DataBrickID': int(DatabrickID),   
                            'UserForecastMethodID': int(UFMID)
                            }        
                        # Prepare forecast dataframe

        # Prepare forecast dataframe
        rmse_results = {}
        r2_results = {}

        # Create lag features for the historical data
        lag_columns = selected_columns[2:]   
        # #print(lag_columns)  

        podel_df = create_lag_features(podel_df, lag_columns, lags=3)
      
        # Fill NaN values with 0 or an appropriate imputation method
        for col in [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]:
            podel_df[col] = pd.to_numeric(podel_df[col], errors='coerce')
        # podel_df = podel_df.fillna(0)


        feature_columns = ["Month", "Year"] + [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]
        for cons_type in cons_types:
            # Prepare feature and target matrices
            X = podel_df[feature_columns].values
            Y = podel_df[cons_type].values
            # forecast_df =[]
            # Prepare feature and target matrices


            # #print(customer_df )
            # forecast_df = pd.DataFrame(index=[0, 1, 2])
            if X.shape[0] <= 5:
                    continue
            else:   
                try:

                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

                    X_imputer = SimpleImputer(strategy='mean')
                    Y_imputer = SimpleImputer(strategy='mean')
                    
                    # Impute missing values for X_train and X_test
                    X_train_imputed = X_imputer.fit_transform(X_train)
                    X_test_imputed = X_imputer.transform(X_test)

                    # Scale the training features
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_test_scaled = scaler.transform(X_test_imputed)

                    # Impute missing values in Y_train and Y_test
                    Y_train_imputed = Y_imputer.fit_transform(Y_train.reshape(-1,1))
                    Y_test_imputed = Y_imputer.transform(Y_test.reshape(-1,1))


                    # Define the parameter grid for XGBRegressor
                    # param_grid = {
                    #                 'n_estimators': [50, 100, 200],  # Number of trees
                    #                 'max_depth': [10, 20, 30],       # Maximum depth of the trees
                    #                 'min_samples_split': [2, 5, 10], # Minimum number of samples required to split a node
                    #                 'min_samples_leaf': [1, 2, 4],   # Minimum number of samples required at each leaf node
                    #                 'max_features': ['auto', 'sqrt'],# Number of features to consider when looking for the best split
                    #                 'bootstrap': [True, False]  
                    #             }

                    # Create a custom scoring function (RMSE)
                    scoring = make_scorer(mean_squared_error, squared=False)


                    model = RandomForestRegressor(n_estimators=5, random_state=42)


                        # Use GridSearchCV for hyperparameter tuning with cross-validation
                    # grid_search = GridSearchCV(
                    #                             estimator=model,
                    #                             param_grid=param_grid,
                    #                             scoring=scoring,
                    #                             cv=5,  # 5-fold cross-validation
                    #                             verbose=3,
                    #                             n_jobs=-1
                    #                         )                


                    # Perform grid search with cross-validation
                    model.fit(X_train_scaled, Y_train_imputed)
                    # model.fit(X_train_scaled, Y_train_imputed)


                    # Get the best model from the search
                    # best_model = grid_search.best_estimator_
                    # #print(f"best_model parameter values are: {best_model}")

                    # # Evaluate model performance on test data
                    # Y_pred_test = model.predict(X_test_scaled)

                    Y_pred_test = model.predict(X_test_scaled)
                except IndexError as e:
                    print(f"IndexError occurred,there is a mismatch in the no of elements : {e}")
                except ValueError as e:
                    print(f"ValueError occurred: {e}")
                except KeyError as e:
                    print(f"KeyError occurred, certain referenced column not found : {e}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                rmse = np.sqrt(mean_squared_error(Y_test_imputed, Y_pred_test))
                r2 = r2_score(Y_test_imputed, Y_pred_test)




                model.fit(X_train_scaled, Y_train)

                forecast_df = pd.DataFrame(forecast_dates, columns=["ReportingMonth"])
                forecast_df["CustomerID"] = customer_id
                forecast_df["PodID"] = podel_id
                forecast_df["Month"] = forecast_df["ReportingMonth"].dt.month
                forecast_df["Year"] = forecast_df["ReportingMonth"].dt.year
                forecast_df["CustomerID"] = forecast_df["CustomerID"].astype(int)
                forecast_df["PodID"] = forecast_df["PodID"].astype(int)
                forecast_df["UserForecastMethodID"] = UFMID                


 
                
                # Initialize lag columns in the forecast DataFrame
                for col in lag_columns:
                    forecast_df[col] = np.nan  # Initialize the consumption columns
                    for lag in range(1, 4):
                        forecast_df[f"{col}_lag{lag}"] = np.nan

                # Main loop for predictions across each forecasted period
                for pred_cur_mth in range(n_periods):
                    # #print(f"Prediction step {pred_cur_mth}")

                    # Update lag features for current step
                    if pred_cur_mth == 0:
                        # First step: Use historical data to initialize lags
                        forecast_df = create_forecast_lag_features(forecast_df, podel_df, lag_columns, lags=3, step=pred_cur_mth)
                        
                    else:
                        # Subsequent steps: Use previously predicted data to update lags
                        forecast_df = create_forecast_lag_features(forecast_df, forecast_df, lag_columns, lags=3, step=pred_cur_mth)

                    # Convert forecast_df columns to numeric types
                    for col in [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]:
                        forecast_df[col] = pd.to_numeric(forecast_df[col], errors='coerce')

                    # Select the current row of features for prediction
                    X_forecast = forecast_df.loc[pred_cur_mth, feature_columns].values.reshape(1, -1)


                    # Impute missing values in X_forecast
                    X_forecast_imputed = X_imputer.transform(X_forecast)

                    # Scale the features before making predictions
                    X_forecast_scaled = scaler.transform(X_forecast_imputed)
        
                    # Make prediction
                    prediction = model.predict(X_forecast_scaled)

                    # Store the prediction


                    # Update the forecast DataFrame with the predicted values
                    for idx, col in enumerate(selected_columns[2:]):
                        # if idx < prediction.shape[1]:
                        #     forecast_df.loc[pred_cur_mth, col] = prediction[0, idx]
                        forecast_df.loc[pred_cur_mth, cons_type] = prediction
                    # Update lag features with the newly predicted values for future steps
                    for lag in range(1, 4):
                        next_step = pred_cur_mth + lag
                        if next_step < len(forecast_df):
                            # for col in selected_columns[2:]:
                                forecast_df.loc[next_step, f"{cons_type}_lag{lag}"] = forecast_df.loc[pred_cur_mth, cons_type]

              # Drop 'Month' and 'Year' columns and rename the other columns dynamically
                
            lag_columns_to_drop = [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]

        
            # Combine the specific columns to drop with the dynamic lag columns
            columns_to_drop = ["Month", "Year"] + lag_columns_to_drop

            # Drop the columns from the DataFrame
            # forecast_df = forecast_df.drop(columns=columns_to_drop, axis=1)

                # display(forecast_df)
            forecast_df_all_cons[cons_type]=forecast_df[cons_type]
            performance_data[f"RMSE_{cons_type}"] = rmse
            performance_data[f"R2_{cons_type}"] = r2


            historical_df = customer_df[customer_df["PodID"] == podel_id].copy()

                # # Now plot historical vs forecasted values for the features of interest
            # plot_forecast_vs_historical(historical_df, forecast_df,  [cons_type])  
        # display(forecast_df_all_cons)

            # forecast_df.drop(columns='Month', axis=1)
        forecast_combined_spark_df = spark.createDataFrame(forecast_df_all_cons)
        forecast_combined_spark_df = forecast_combined_spark_df.withColumn("CustomerID", forecast_combined_spark_df["CustomerID"].cast("bigint"))
        forecast_combined_spark_df = forecast_combined_spark_df.withColumn("PodID", forecast_combined_spark_df["PodID"].cast("bigint"))
        forecast_combined_spark_df = forecast_combined_spark_df.withColumn("UserForecastMethodID",forecast_combined_spark_df["UserForecastMethodID"].cast("bigint"))

                                    # Write the DataFrame to the SQL table
        forecast_combined_spark_df.write.jdbc(
                                        url=write_url,
                                        table=target_table_name,
                                        mode="append",
                                        properties=write_properties
                                )
            # #print(forecast_df.columns)




              

        



        RMSE_sum = 0
        R2_sum = 0
        rmse_avg = 0  # Initialize rmse_avg
        r2_avg = 0    # Initialize r2_avg
                        
        for cons_type in cons_types:
                if f"RMSE_{cons_type}" in performance_data and f"R2_{cons_type}" in performance_data:
                    RMSE_sum +=  performance_data[f"RMSE_{cons_type}"]
                    R2_sum   +=  performance_data[f"R2_{cons_type}"]
                    rmse_avg = RMSE_sum/len(cons_types)
                    r2_avg = R2_sum/len(cons_types)
        if len(cons_types) > 0:
            rmse_avg = RMSE_sum/len(cons_types)
            r2_avg = R2_sum/len(cons_types)

        performance_data['RMSE_Avg'] = rmse_avg
        performance_data['R2_Avg'] = r2_avg
        performance_df = pd.DataFrame([performance_data])
        performance_spark_df = spark.createDataFrame(performance_df)

                                    # Write the performance metrics to the SQL table
        performance_spark_df.write.jdbc(url=write_url, table=table_performance_Metrics, mode="append", properties=write_properties) 
    

    