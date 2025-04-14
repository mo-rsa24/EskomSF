import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, r2_score

def automated_forecasts_for_all_types(data, selected_columns,forecast_dates, debug=True,  n_periods=12,UFMID=64):
    if selected_columns is None:
        selected_columns = ["OffpeakConsumption", "StandardConsumption", "PeakConsumption"]

    all_forecasts = []

    # Determine which consumption types are in the selected columns
    consumption_types = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption", "Block1Consumption",
                         "Block2Consumption", "Block3Consumption", "Block4Consumption", "NonTOUConsumption"]

    cons_types = [col for col in selected_columns if col in consumption_types]

    # Iterate through each unique customer ID in the dataset.
    for customer_id in data['CustomerID'].unique():
        customer_forecasts = {}
        # Ensure data is sorted by ReportingMonth
        customer_data = data[data['CustomerID'] == customer_id].sort_values('PodID')
        if customer_data.empty:
            continue
        unique_podel_ids = customer_data["PodID"].unique()

        for podel_id in unique_podel_ids:
            podel_df = customer_data[customer_data["PodID"] == podel_id].sort_values('ReportingMonth')
            if podel_df.empty:
                continue
            future_predictions = []

            rmse_results = {}
            performance_data = {
                'ModelName': 'SARIMA',
                'CustomerID': str(customer_id),
                'PodID': str(podel_id),
                'UserForecastMethodID': UFMID
            }
            performance_full_data = pd.DataFrame({'ModelName': ['SARIMA']})

            performance_full_data["CustomerID"] = str(customer_id)
            performance_full_data['PodID'] = str(podel_id)
            performance_full_data['UserForecastMethodID'] = UFMID
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
                try:

                    if podel_df[cons_type].isnull().all():
                        continue

                    # Prepare the time series  by setting ReportingMonth as the index
                    series = podel_df.set_index('ReportingMonth')[cons_type]
                    series = pd.to_numeric(series, errors='coerce').fillna(0)
                    log_series = np.log(series)


                    forecast = None

                    # Fit SARIMA model
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
                    model_fit = model.fit(log_series)
                    log_forecast = model_fit.predict(n_periods=n_periods)

                    forecast = np.exp(log_forecast)
                    actual_values = series[-n_periods:]

                    def mean_baseline(train_series):
                        mean_value = train_series.mean()
                        return np.full(len(actual_values), mean_value)

                    baseline_predictions = mean_baseline(actual_values)
                    baseline_rmse = np.sqrt(mean_squared_error(actual_values, baseline_predictions))
                    baseline_r2 = r2_score(actual_values, baseline_predictions)

                    sarima_rmse = np.sqrt(mean_squared_error(actual_values, forecast[:len(actual_values)]))
                    sarmia_r2 = r2_score(actual_values, forecast[:len(actual_values)])
                    if debug:
                        print(f"RMSE of SARIMA  for consumption type {cons_type} of Podel {podel_id} is {sarima_rmse}")
                        print(
                            f"RMSE of Baseline for consumption type {cons_type} of Podel {podel_id} is {baseline_rmse}")

                    customer_forecasts[cons_type] = forecast
                    performance_data[f"RMSE_{cons_type}"] = sarima_rmse
                    performance_data[f"R2_{cons_type}"] = sarmia_r2
                except Exception as e:
                    print(f"Error processing {cons_type} for PodID {podel_id}: {e}")
                    continue
                performance_full_data[f"RMSE_{cons_type}"] = performance_data[f"RMSE_{cons_type}"]
                performance_full_data[f"R2_{cons_type}"] = performance_data[f"R2_{cons_type}"]

            for cons_type in cons_types:
                customer_forecasts[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)

            for selected_cons_type in cons_types:
                RMSE_sum += performance_data.get(f"RMSE_{selected_cons_type}", 0) or 0
                R2_sum += performance_data.get(f"R2_{selected_cons_type}", 0) or 0

            rmse_avg = RMSE_sum / len(cons_types)
            r2_avg = R2_sum / len(cons_types)
            performance_full_data['RMSE_Avg'] = rmse_avg
            performance_full_data['R2_Avg'] = r2_avg

            for cons_type in cons_types:
                forecast_df_data[cons_type] = customer_forecasts.get(cons_type, [None] * n_periods)
            cleaned_forecast_df_data = {
                k: (v.values if isinstance(v, pd.Series) else v)
                for k, v in forecast_df_data.items()
            }
            forecast_df = pd.DataFrame(cleaned_forecast_df_data, index=range(n_periods))

            # Append each customer's forecast DataFrame to a list.
            all_forecasts.append(forecast_df)

        # Concatenate all individual forecast DataFrames into one.
        forecast_combined_df = pd.concat(all_forecasts, ignore_index=True)
        # Round the consumption columns if they are present
        for cons_type in cons_types:
            forecast_combined_df[cons_type] = forecast_combined_df[cons_type].fillna(0).round(2)

        # Return the combined forecast DataFrame.

        #         # Define the properties for the database connection and write predicted results to DB
        forecast_combined_df['UserForecastMethodID'] = UFMID
        return forecast_combined_df