import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_lag_features(df, lag_columns, lags):
    for col in lag_columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
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
    # if debug:
    # print(f"Forecast lag features created for step: {step}")
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
    historical_df['ReportingMonth'] = pd.to_datetime(historical_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp(
        'M')
    forecast_df['ReportingMonth'] = pd.to_datetime(forecast_df['ReportingMonth']).dt.to_period('M').dt.to_timestamp('M')

    # Recalculate last_historical_date after conversion
    last_historical_date = historical_df['ReportingMonth'].max()

    # if debug:
    # print(f"Last Historical Date: {last_historical_date}")
    # print(f"Forecast Dates Start: {forecast_df['ReportingMonth'].min()}")

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
        # print("No data available for plotting.")
        return

    # Set plot size
    num_features = len(features)
    fig, axs = plt.subplots(num_features, 1, figsize=(16, 5 * num_features), sharex=True)
    if num_features == 1:
        axs = [axs]

    for idx, feature in enumerate(features):
        ax = axs[idx]
        # #print('ax',ax)
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