# Databricks notebook source
# MAGIC %pip install xg
# MAGIC
# MAGIC import numpy as np
# MAGIC import matplotlib.pyplot as plt
# MAGIC import matplotlib.dates as mdates
# MAGIC import seaborn as sns
# MAGIC import calendar
# MAGIC # Import necessary libraries
# MAGIC from pyspark.sql import SparkSession
# MAGIC from datetime import datetime
# MAGIC from xg import XGBRegressor
# MAGIC from sklearn.model_selection import train_test_split
# MAGIC from sklearn.preprocessing import LabelEncoder
# MAGIC from sklearn.preprocessing import StandardScaler
# MAGIC from sklearn.multioutput import MultiOutputRegressor
# MAGIC from sklearn.impute import KNNImputer, SimpleImputer
# MAGIC from sklearn.metrics import mean_squared_error,r2_score,make_scorer
# MAGIC from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
# MAGIC
# MAGIC import pyodbc
# MAGIC import pandas as pd
# MAGIC from pyspark.dbutils import DBUtils
# MAGIC
# MAGIC
# MAGIC
# MAGIC from pyspark.sql.types import ArrayType, StructType, StructField, StringType 
# MAGIC from pyspark.sql.functions import from_json, col, explode,collect_list
# MAGIC
# MAGIC dbutils = DBUtils(spark)
# MAGIC
# MAGIC
# MAGIC # Retrieve parameter passed to the notebook 
# MAGIC # Get Forecasters input from Databricks task id
# MAGIC databrick_task_id = dbutils.widgets.get("DatabrickTaskID")
# MAGIC
# MAGIC debug = False
# MAGIC print(databrick_task_id)
# MAGIC
# MAGIC
# MAGIC # Initialize Spark session (Databricks environment should have this pre-configured)
# MAGIC spark = SparkSession.builder.appName("Energy Consumption Prediction").getOrCreate()
# MAGIC
# MAGIC # Read dataset from SQL Server
# MAGIC server_name = "jdbc:sqlserver://esk-maz-sdb-san-dev-01.database.windows.net"
# MAGIC database_name = "ESK-MAZ-SDB-SAN-DEV-01"
# MAGIC url = server_name + ";" + "databaseName=" + database_name + ";"
# MAGIC table_DBT = "dbo.DataBrickTasks"
# MAGIC table_UFM = "dbo.UserForecastMethod"
# MAGIC table_DimPod = "dbo.DIMPOD"
# MAGIC
# MAGIC table_actual = "dbo.ActualData"
# MAGIC table_version = "dbo.DimVersion"
# MAGIC table_forecast = "dw.ForecastActive"
# MAGIC table_pod_aggrgated = "test.PodEnergyUsageRaw03";
# MAGIC table_performance_Metrics = "dbo.StatisticalPerformanceMetrics"
# MAGIC
# MAGIC # Define the name of the target table
# MAGIC target_table_name = "dbo.ForecastFact"
# MAGIC
# MAGIC user = "arul"
# MAGIC password = "aari@Singds.8734"
# MAGIC
# MAGIC # Get Forecasters input from Databricks task id
# MAGIC # Define a SQL query to retrieve various forecasting details associated with a specific Databricks task.
# MAGIC # This includes forecast methods, customer details, regional dataset, and task execution status.
# MAGIC
# MAGIC # Define the properties for the database connection
# MAGIC write_url = "jdbc:sqlserver://esk-maz-sdb-san-dev-01.database.windows.net;databaseName=ESK-MAZ-SDB-SAN-DEV-01"
# MAGIC write_properties = {
# MAGIC     "user": "arul",
# MAGIC     "password": "aari@Singds.8734",
# MAGIC     "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
# MAGIC }
# MAGIC
# MAGIC
# MAGIC # ufm.Name
# MAGIC query = f"""
# MAGIC SELECT TOP 1
# MAGIC     ufm.StartDate,
# MAGIC     ufm.EndDate,
# MAGIC     ufm.Parameters,
# MAGIC     ufm.Region,
# MAGIC     ufm.Status,
# MAGIC     ufm.ForecastMethodID,
# MAGIC     ufm.UserForecastMethodID,
# MAGIC     ufm.JSONCustomer as CustomerJSON,
# MAGIC
# MAGIC     ufm.varJSON,    
# MAGIC     dfm.Method,
# MAGIC     dbt.DatabrickID
# MAGIC FROM 
# MAGIC     [dbo].[DataBrickTasks] AS dbt
# MAGIC INNER JOIN 
# MAGIC     [dbo].[UserForecastMethod] AS ufm ON dbt.UserForecastMethodID = ufm.UserForecastMethodID
# MAGIC INNER JOIN 
# MAGIC     [dbo].[DimForecastMethod] AS dfm ON ufm.ForecastMethodID = dfm.ForecastMethodID
# MAGIC WHERE  dbt.DatabrickID={databrick_task_id} and ufm.ForecastMethodID=6 and ExecutionStatus IN ('In Progress')
# MAGIC
# MAGIC
# MAGIC ORDER BY
# MAGIC     dbt.CreationDate 
# MAGIC
# MAGIC """
# MAGIC
# MAGIC
# MAGIC # Failed
# MAGIC # Read dataset using Spark SQL by setting up the database connection and executing the SQL query.
# MAGIC
# MAGIC df = spark.read \
# MAGIC     .format("jdbc") \
# MAGIC     .option("url", url) \
# MAGIC     .option("query", query) \
# MAGIC     .option("user", user) \
# MAGIC     .option("password", password) \
# MAGIC     .load()
# MAGIC
# MAGIC
# MAGIC # Extract specific fields from the DataFrame, convert them to a Pandas DataFrame, and store in variables.
# MAGIC
# MAGIC
# MAGIC Forecast_Method_Name = df.select("Method").toPandas().iloc[0]['Method']
# MAGIC Model_Parmeters=df.select("Parameters").toPandas().iloc[0]['Parameters']
# MAGIC UFMID=df.select("UserForecastMethodID").toPandas().iloc[0]['UserForecastMethodID']
# MAGIC # CustomerID=df.select("Customer").toPandas().iloc[0]['Customer']
# MAGIC
# MAGIC StartDate=df.select("StartDate").toPandas().iloc[0]['StartDate']
# MAGIC EndDate=df.select("EndDate").toPandas().iloc[0]['EndDate']
# MAGIC DatabrickID=df.select("DatabrickID").toPandas().iloc[0]['DatabrickID']
# MAGIC Hyper_Parameters=df.select("Parameters").toPandas().iloc[0]['Parameters']
# MAGIC
# MAGIC
# MAGIC print(Forecast_Method_Name,UFMID,DatabrickID)
# MAGIC
# MAGIC
# MAGIC Hyper_parameter_values = Hyper_Parameters.split(',')
# MAGIC
# MAGIC
# MAGIC
# MAGIC n_estimators = int(Hyper_parameter_values[0].replace('(', ''))
# MAGIC max_depth = int(Hyper_parameter_values[1])
# MAGIC learning_rate = float(Hyper_parameter_values[2])
# MAGIC subsample = float(Hyper_parameter_values[3])
# MAGIC # colsample_bytree =int(Hyper_parameter_values[4].replace('(', ''))
# MAGIC colsample_bytree =float(Hyper_parameter_values[4].replace(')', ''))
# MAGIC
# MAGIC # min_child_weight = int(float(Hyper_parameter_values[5].replace(')', '')))
# MAGIC
# MAGIC
# MAGIC print(f" Hyper parameters values are {n_estimators, max_depth, learning_rate, subsample, colsample_bytree}")
# MAGIC
# MAGIC
# MAGIC # json_schema = ArrayType(StructType([
# MAGIC #     StructField("CustomerID", StringType(), True)
# MAGIC # ]))
# MAGIC
# MAGIC
# MAGIC # # df_cust.show()
# MAGIC
# MAGIC # if "CustomerJSON" in df.columns:
# MAGIC #     json_schema = ArrayType(StructType([
# MAGIC #         StructField("CustomerID", StringType(), True)
# MAGIC #     ]))
# MAGIC
# MAGIC #     # Parse the JSON string into a column of arrays of structs
# MAGIC #     df_cust = df.withColumn("ParsedJSON", from_json("CustomerJSON", json_schema))\
# MAGIC #                 .select(explode("ParsedJSON").alias("CustomerDetails"))\
# MAGIC #                 .select(col("CustomerDetails.CustomerID"))
# MAGIC
# MAGIC
# MAGIC
# MAGIC #     # Collect the IDs into a list
# MAGIC #     multiple_customer_ids_list = df_cust.agg(collect_list("CustomerID")).first()[0]
# MAGIC
# MAGIC #     # Convert the list to a comma-separated string
# MAGIC
# MAGIC
# MAGIC #  # Convert the list to a comma-separated string
# MAGIC #     if multiple_customer_ids_list:
# MAGIC #         # multiple_customer_ids_list = ','.join(multiple_customer_ids_list)
# MAGIC #         multiple_customer_ids_list = ",".join([f"'{customer_id}'" for customer_id in multiple_customer_ids_list])
# MAGIC #     else:
# MAGIC #         multiple_customer_ids_list = ''
# MAGIC
# MAGIC #     # Output the comma-separated IDs
# MAGIC #     print("Comma-separated Customer IDs:")
# MAGIC #     print(multiple_customer_ids_list)
# MAGIC
# MAGIC # else:
# MAGIC #     print("Column 'CustomerJSON' does not exist in the dataframe.")
# MAGIC
# MAGIC
# MAGIC print(df.columns)
# MAGIC if "varJSON" in df.columns:
# MAGIC     print("True")
# MAGIC     VarJsonSchema =  StructType([
# MAGIC                                     StructField("VariableID", ArrayType(StringType()), True)
# MAGIC                                 ])
# MAGIC
# MAGIC                             
# MAGIC
# MAGIC
# MAGIC
# MAGIC     df_var=df.withColumn("ParsedVarJson",from_json("varJSON",VarJsonSchema))\
# MAGIC          .select(explode(col("ParsedVarJson.VariableID")).alias("SelectedVarList"))
# MAGIC         #  .select(col("SelectedVarList.VariableID")).alias("VariableID")
# MAGIC     all_variables=df_var.agg(collect_list("SelectedVarList")).first()[0] 
# MAGIC     all_variables=','.join(all_variables)
# MAGIC     
# MAGIC     # df_var.select("SelectedVarList").show(truncate=False)
# MAGIC print(all_variables)
# MAGIC
# MAGIC # Ensure all_variables is a list
# MAGIC if isinstance(all_variables, str):
# MAGIC     all_variables = all_variables.split(',')
# MAGIC
# MAGIC
# MAGIC     
# MAGIC # Create DataFrame
# MAGIC
# MAGIC
# MAGIC # Sample AllVariables DataFrame
# MAGIC # all_variables = ["Peak"]
# MAGIC
# MAGIC # Define the required columns mapping
# MAGIC columns_mapping = {
# MAGIC     frozenset(["PeakConsumption", "StandardConsumption", "OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption", "StandardConsumption", "PeakConsumption"],
# MAGIC     frozenset(["PeakConsumption", "StandardConsumption"]): ["ReportingMonth", "CustomerID", "StandardConsumption", "PeakConsumption"],
# MAGIC     frozenset(["PeakConsumption", "OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption", "PeakConsumption"],
# MAGIC     frozenset(["StandardConsumption", "OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption", "StandardConsumption"],
# MAGIC     frozenset(["PeakConsumption"]): ["ReportingMonth", "CustomerID", "PeakConsumption"],
# MAGIC     frozenset(["StandardConsumption"]): ["ReportingMonth", "CustomerID", "StandardConsumption"],
# MAGIC     frozenset(["OffPeakConsumption"]): ["ReportingMonth", "CustomerID", "OffpeakConsumption"]
# MAGIC }
# MAGIC
# MAGIC # Convert AllVariables to a set for easy comparison
# MAGIC all_variables_set = set(all_variables)
# MAGIC print(all_variables_set)
# MAGIC
# MAGIC # Find the matching key in the columns_mapping
# MAGIC matching_key = None
# MAGIC for key in columns_mapping.keys():
# MAGIC     # print(key)
# MAGIC     if key.issubset(all_variables_set):
# MAGIC         matching_key = key
# MAGIC         break
# MAGIC
# MAGIC # Select the appropriate columns based on the matching key
# MAGIC if matching_key:
# MAGIC     selected_columns = columns_mapping[matching_key]
# MAGIC
# MAGIC     print(f" All columns based on selectedcolumns{selected_columns}")
# MAGIC else:
# MAGIC     print("No matching columns found in AllVariables")

# COMMAND ----------



 

# COMMAND ----------


# Construct a SQL query to select all records from the actuals table for a specific customer.

# query_act_cons = f"(SELECT * FROM {table_actual} WHERE CustomerID IN ({multiple_customer_ids_list})) AS subquery"
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

act_df = act_df.orderBy("PodID", "ReportingMonth")
pandas_df = act_df.toPandas()

# COMMAND ----------

print(pandas_df.head())

# COMMAND ----------

multiple_customer_ids_list = pandas_df['CustomerID'].unique()
if len(multiple_customer_ids_list)>0:
    # Output the comma-separated IDs
    print(f"Future consumption will be predicted for customers:{multiple_customer_ids_list}")
else:
    # Output the comma-separated IDs
    print(f"No consumption dataset available for selected Customer IDs: {multiple_customer_ids_list} or customer ids selection is not successful")


# COMMAND ----------


# Convert Spark DataFrame to Pandas DataFrame for SARIMA

pandas_df['CustomerID'] = pandas_df['CustomerID'].astype(str)


pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'],format ='%Y-%m-%d').dt.to_period('M').dt.to_timestamp()


# Find the most recent reporting month in the dataset, which will be used to determine the starting point for forecasting.
Actuals_last_date = pandas_df['ReportingMonth'].max() 

# Generate a date range from the start date to the end date with a monthly frequency, starting on the first of each month.
# This range represents the forecast period.

forecast_dates = pd.date_range(start=StartDate, end=EndDate, freq='MS')[0:]
print(forecast_dates,f"Actual_last_date {Actuals_last_date}",f"No of month to predict for {len(forecast_dates)}")


# Calculate n_periods as the number of months between the last historical date
# and the last date you want to forecast
n_periods = len(forecast_dates)

print(f"Number of periods to forecast: {n_periods}")


# COMMAND ----------

unique_customers = pandas_df['CustomerID'].unique()
unique_PodID = pandas_df[pandas_df['PodID'].notnull()]['PodID'].unique()
print(f"Unique Customers are {unique_customers}")
print(f"Unique PODs are {unique_PodID}")

# COMMAND ----------

 

# Convert 'ReportingMonth' to datetime and extract features
#df['ReportingMonth'] = pd.to_datetime(df['ReportingMonth'])
pandas_df['ReportingMonth'] = pd.to_datetime(pandas_df['ReportingMonth'])
pandas_df['Month'] = pandas_df['ReportingMonth'].dt.month
pandas_df['Year'] = pandas_df['ReportingMonth'].dt.year

display(pandas_df[['ReportingMonth','Month','Year']])

# COMMAND ----------

# Scaling the features
scaler = StandardScaler()
sim_imputer = SimpleImputer()
consumption_types = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]
cons_types = [col for col in selected_columns if col in consumption_types]

def create_lag_features(df, lag_columns, lags):
    for col in lag_columns:
            for lag in range(1,lags+1):
                df[f"{col}_lag{lag}"]=df[col].shift(lag)
    return df


# Create lag features for forecast dataset
def create_forecast_lag_features(df, original_df, lag_columns, lags, step):
    # Update the lag features for the forecast DataFrame using either historical or predicted dataset
    for col in lag_columns:
        for lag in range(1, lags + 1):
            if step == 0:
                # Use historical dataset to initialize lags
                df.loc[step, f"{col}_lag{lag}"] = original_df[col].iloc[-lag]
            else:
                # Use previous predictions to update lags
                df.loc[step, f"{col}_lag{lag}"] = df[col].iloc[step - lag]
    if debug:
        print(f"Forecast lag features created for step: {step}")
    return df


# Plot forecast vs historical dataset
def plot_forecast_vs_historical(historical_df, forecast_df, features):
    """
    Plots historical vs forecasted values for specified features.
    
    Parameters:
    - historical_df: DataFrame containing historical dataset
    - forecast_df: DataFrame containing forecasted dataset
    - features: List of feature names to plot
    """
    # Convert ReportingMonth to datetime at end of month
    historical_df['ReportingMonth'] = pd.to_datetime(historical_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp('M')
    forecast_df['ReportingMonth'] = pd.to_datetime(forecast_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp('M')

    # Recalculate last_historical_date after conversion
    last_historical_date = historical_df['ReportingMonth'].max()

    if debug:
        print(f"Last Historical Date: {last_historical_date}")
        print(f"Forecast Dates Start: {forecast_df['ReportingMonth'].min()}")

    # Ensure consumption columns are numeric and handle NaNs
    for feature in features:
        historical_df[feature] = pd.to_numeric(historical_df[feature], errors='coerce').fillna(0)
        forecast_df[feature] = pd.to_numeric(forecast_df[feature], errors='coerce').fillna(0)

    # Filter forecast dataset to only include periods after the last historical date
    forecast_df = forecast_df[forecast_df['ReportingMonth'] > last_historical_date]

    # Group dataset by 'ReportingMonth'
    historical_grouped = historical_df.groupby('ReportingMonth')[features].mean().reset_index()
    forecast_grouped = forecast_df.groupby('ReportingMonth')[features].mean().reset_index()


    # Add check for empty dataframes
    if historical_grouped.empty or forecast_grouped.empty:
        print("No dataset available for plotting.")
        return

    # Set plot size
    num_features = len(features)
    fig, axs = plt.subplots(num_features, 1, figsize=(16, 5 * num_features), sharex=True)
    if num_features == 1:
        axs = [axs]

    for idx, feature in enumerate(features):
        ax = axs[idx]
        # print('ax',ax)
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
    print(f"Processing CustomerID: {customer_id}")

    customer_df = pandas_df[pandas_df["CustomerID"] == customer_id]
    unique_podel_ids = customer_df["PodID"].unique()

    for podel_id in unique_podel_ids:
        # if podel_id == "6582636837":
            
            print(f"Processing PODEL_ID: {podel_id}")

            podel_df = customer_df[customer_df["PodID"] == podel_id]
            future_predictions = []
            forecast_df_all_cons = pd.DataFrame(forecast_dates, columns=["ReportingMonth"])
            forecast_df_all_cons["CustomerID"] = customer_id
            forecast_df_all_cons["PodID"] = podel_id
            forecast_df_all_cons["CustomerID"] = forecast_df_all_cons["CustomerID"].astype(int)
            forecast_df_all_cons["PodID"] = forecast_df_all_cons["PodID"].astype(int)
            forecast_df_all_cons["UserForecastMethodID"] = UFMID    
            performance_data = { 'ModelName': Forecast_Method_Name,
                                'CustomerID': str(customer_id),
                                'PodID': str(podel_id),
                                'DataBrickID': int(DatabrickID),   
                                'UserForecastMethodID': int(UFMID)
                                }

            # Create lag features for the historical dataset
            # lag_columns = ['OffpeakConsumption', 'StandardConsumption', 'PeakConsumption']
            lag_columns = selected_columns[2:]
            
            podel_df = create_lag_features(podel_df, lag_columns, lags=3)

            # Fill NaN values with 0 or an appropriate imputation method
            for col in [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]:
                podel_df[col] = pd.to_numeric(podel_df[col], errors='coerce')
            podel_df = podel_df.fillna(0)

            feature_columns = ["Month", "Year"] + [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]
            for cons_type in cons_types:
                # Prepare feature and target matrices
                X = podel_df[feature_columns].values
                Y = podel_df[cons_type].values







                if X.shape[0] > 4:
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
                    param_grid = {
                        'estimator__n_estimators': [10, 50, 100],
                        'estimator__max_depth': [5, 10, 15],
                        'estimator__learning_rate': [0.01, 0.1, 0.3],
                        'estimator__subsample': [0.8, 0.9, 1],
                        'estimator__colsample_bytree': [0.7, 0.8, 1]
                    }

                    # Create a custom scoring function (RMSE)
                    scoring = make_scorer(mean_squared_error, squared=False)

                    # Define the model inside a MultiOutputRegressor
                    model = XGBRegressor(objective="reg:squarederror", enable_categorical=True)

                    # Use GridSearchCV for hyperparameter tuning with cross-validation
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        scoring=scoring,
                        cv=5,  # 5-fold cross-validation
                        verbose=3,
                        n_jobs=-1
                    )                


                    # Perform grid search with cross-validation
                    grid_search.fit(X_train_scaled, Y_train_imputed)
                    # model.fit(X_train_scaled, Y_train_imputed)


                    # Get the best model from the search
                    best_model = grid_search.best_estimator_
                    print(f"best_model parameter values are: {best_model}")

                    # # Evaluate model performance on test dataset
                    # Y_pred_test = model.predict(X_test_scaled)

                    Y_pred_test = best_model.predict(X_test_scaled)

                    rmse = np.sqrt(mean_squared_error(Y_test_imputed, Y_pred_test))
                    r2 = r2_score(Y_test_imputed, Y_pred_test)

                    # if debug:
                    print(f"Test RMSE: {rmse}")
                    print(f"Test R²: {r2}")


                    model.fit(X_train_scaled, Y_train)

                    # Prepare forecast dataframe
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
                        print(f"Prediction step {pred_cur_mth}")

                        # Update lag features for current step
                        if pred_cur_mth == 0:
                            # First step: Use historical dataset to initialize lags
                            forecast_df = create_forecast_lag_features(forecast_df, podel_df, lag_columns, lags=3, step=pred_cur_mth)
                            
                        else:
                            # Subsequent steps: Use previously predicted dataset to update lags
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
                        future_predictions.append(prediction)


                        # Update the forecast DataFrame with the predicted values
                        if len(prediction.shape) == 1:
                            # for idx, col in enumerate(selected_columns[2:]):
                            #     if idx < len(prediction):
                                    forecast_df.loc[pred_cur_mth, cons_type] = prediction
                        else:
                            # for idx, col in enumerate(selected_columns[2:]):
                            #     if idx < prediction.shape[1]:
                                    forecast_df.loc[pred_cur_mth, cons_type] = prediction[0]

                        # Update lag features with the newly predicted values for future steps
                        for lag in range(1, 4):
                            next_step = pred_cur_mth + lag
                            if next_step < len(forecast_df):
                                # for col in selected_columns[2:]:
                                    forecast_df.loc[next_step, f"{cons_type}_lag{lag}"] = forecast_df.loc[pred_cur_mth, cons_type]

                        print( forecast_df.loc[pred_cur_mth, cons_type])



                    # if cons_type == "PeakConsumption":
                    #     display(forecast_df)

                    # print(rename_dict)

                    # print(forecast_df.info())

        
                lag_columns_to_drop = [f"{col}_lag{lag}" for col in lag_columns for lag in range(1, 4)]

            
                # Combine the specific columns to drop with the dynamic lag columns
                columns_to_drop = ["Month", "Year"] + lag_columns_to_drop

                # Drop the columns from the DataFrame
                forecast_df = forecast_df.drop(columns=columns_to_drop, axis=1)

                    # display(forecast_df)
                forecast_df_all_cons[cons_type]=forecast_df[cons_type]
                performance_data[f"RMSE_{cons_type}"] = rmse
                performance_data[f"R2_{cons_type}"] = r2


                historical_df = customer_df[customer_df["PodID"] == podel_id].copy()

                    # # Now plot historical vs forecasted values for the features of interest
                plot_forecast_vs_historical(historical_df, forecast_df,  [cons_type])  
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
                # print(forecast_df.columns)




                

            



            RMSE_sum = 0
            R2_sum = 0
                            
            for cons_type in cons_types:
                    RMSE_sum +=  performance_data[f"RMSE_{cons_type}"]
                    R2_sum   +=  performance_data[f"R2_{cons_type}"]
                    rmse_avg = RMSE_sum/len(cons_types)
                    r2_avg = R2_sum/len(cons_types)


            performance_data['RMSE_Avg'] = rmse_avg
            performance_data['R2_Avg'] = r2_avg
            performance_df = pd.DataFrame([performance_data])
            performance_spark_df = spark.createDataFrame(performance_df)

                                        # Write the performance metrics to the SQL table
            performance_spark_df.write.jdbc(url=write_url, table=table_performance_Metrics, mode="append", properties=write_properties) 
        

        


# COMMAND ----------

# query_performance_metrics = f"""
#                      ( 
#                      select * from dbo.StatisticalPerformanceMetrics  where ModelName = 'XGBoost'  
#                      union all
#                      select * from dbo.StatisticalPerformanceMetrics  where ModelName = 'Random Forest'  
#                      union all
#                      select * from dbo.StatisticalPerformanceMetrics  where ModelName = 'ARIMA' 
#                      union all
#                      select * from dbo.StatisticalPerformanceMetrics  where ModelName = 'SARIMA'  
                                                               
#                      )
#                     """
 
# print(query_performance_metrics)

# performance_spark_df = spark.read \
#     .format("jdbc") \
#     .option("url", url) \
#     .option("query", query_performance_metrics) \
#     .option("user", user) \
#     .option("password", password) \
#     .load()


# performance_pd_df = performance_spark_df.toPandas()
# # Create a DataFrame


# sns.set(style="whitegrid")

# # Create a figure with subplots for RMSE and R2
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# # Plot for RMSE
# sns.barplot(dataset=performance_pd_df, x='PodID', y='RMSE', hue='ModelName', ax=axes[0])
# axes[0].set_title('RMSE by Model at PodID Level')
# axes[0].set_ylabel('RMSE')
# axes[0].set_xlabel('PodID')
# axes[0].tick_params(axis='x', rotation=45)
# axes[0].legend(title='Model Name')

# # Plot for R2
# sns.barplot(dataset=performance_pd_df, x='PodID', y='R2', hue='ModelName', ax=axes[1])
# axes[1].set_title('R² by Model at PodID Level')
# axes[1].set_ylabel('R²')
# axes[1].set_xlabel('PodID')
# axes[1].tick_params(axis='x', rotation=45)
# axes[1].legend(title='Model Name')

# plt.tight_layout()
# plt.show()

# COMMAND ----------


                # Loop over all_variables to create the corresponding predicted columns
                # selected_columns[2:] is assumed to be a list of the variable names used during training
                # for idx, variable in enumerate(selected_columns[2:]):
                #     predicted_col = f"Predicted{variable}"
                #     print(variable, idx, predicted_col)

                #     # Add predictions to forecast_df based on the length of forecast_df
                #     forecast_df[predicted_col] = future_predictions[:, idx]

                # # Print the forecast_df to verify the predictions have been added


                # for cons_type in [
                #     "PredictedOffpeakConsumption",
                #     "PredictedStandardConsumption",
                #     "PredictedPeakConsumption",
                # ]:
                #     if cons_type in forecast_df:
                #         forecast_df[cons_type] = forecast_df[cons_type].apply(lambda x: round(x, 2))

                # forecast_df["UserForecastMethodID"] = UFMID

                # # Create the renaming dictionary dynamically based on all_variables
                # rename_dict = {f"Predicted{var}": var for var in selected_columns[2:]}

                # # print(rename_dict)
                # rename_dict["Customer_ID"] = "CustomerID"
                # forecast_df["PodID"] = forecast_df["PodID"].astype(object)
    
                # # print(forecast_df.info())

                # # Drop 'Month' and 'Year' columns and rename the other columns dynamically
                # forecast_df = forecast_df.drop(["Month", "Year",'OffpeakConsumption_lag1','OffpeakConsumption_lag2', 'OffpeakConsumption_lag3',             'StandardConsumption_lag1','StandardConsumption_lag2', 'StandardConsumption_lag3','PeakConsumption_lag1', 'PeakConsumption_lag2', 'PeakConsumption_lag3'], axis=1).rename(
                #     columns=rename_dict
                # )
                # print(forecast_df.columns)

                # # display(forecast_df)            
                
                # # print("final forecast df", forecast_df)
                # forecast_combined_spark_df = spark.createDataFrame(forecast_df)
                #     # Define the properties for the database connection to write the final predicted results
                # write_url = "jdbc:sqlserver://esk-maz-sdb-san-dev-01.database.windows.net;databaseName=ESK-MAZ-SDB-SAN-DEV-01"
                # write_properties = {
                #     "user": "arul",
                #     "password": "aari@Singds.8734",
                #     "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver"
                # }


                # # Define the name of the target table
                # target_table_name = "dbo.ForecastFact"

                # # Write the DataFrame to the SQL table
                # forecast_combined_spark_df.write.jdbc(url=write_url, table=target_table_name, mode="append", properties=write_properties)

# COMMAND ----------

