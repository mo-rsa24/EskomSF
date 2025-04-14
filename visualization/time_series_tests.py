import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def plot_consumption_trends(melted_df):
    # Plot
    plt.figure(figsize=(16, 10))
    sns.lineplot(data=melted_df, x='ReportingMonth', y='kWh', hue='ConsumptionType', style='CustomerID')
    plt.title("Customer-Level Electricity Consumption Trends")
    plt.xlabel("Reporting Month")
    plt.ylabel("Consumption (kWh)")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_consumption_over_time(df, consumption_cols=None):
    # Create subplots for each consumption type
    if consumption_cols is None:
        consumption_cols = [
            'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
            'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
            'Block4Consumption', 'NonTOUConsumption'
        ]
    fig, axes = plt.subplots(len(consumption_cols), 1, figsize=(16, 28), sharex=True)

    # Get top customers for plotting
    top_customers = df['CustomerID'].value_counts().head(4).index.tolist()
    sample_df = df.reset_index()
    sample_df = sample_df[sample_df['CustomerID'].isin(top_customers)]


    for i, col in enumerate(consumption_cols):
        ax = axes[i]
        for cust_id in top_customers:
            cust_df = sample_df[sample_df['CustomerID'] == cust_id]
            ax.plot(cust_df['ReportingMonth'], cust_df[col], marker='o', label=f'Customer {cust_id}')
        ax.set_title(f"{col} Over Time")
        ax.set_ylabel("kWh")
        ax.grid(True)
        ax.legend(loc="upper right")
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.xlabel("Reporting Month")
    plt.tight_layout()
    plt.show()

# 1. ACF and PACF Plots
def acf_and_pacf_plots(ts, single_customer):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(ts, ax=axes[0], lags=24)
    plot_pacf(ts, ax=axes[1], lags=24)
    axes[0].set_title(f"ACF - PeakConsumption ({single_customer})")
    axes[1].set_title(f"PACF - PeakConsumption ({single_customer})")
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix_of_consumption_types(df, consumption_cols=None):
    if consumption_cols is None:
        consumption_cols = [
            'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
            'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
            'Block4Consumption', 'NonTOUConsumption'
        ]
    correlation_matrix = df[consumption_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix of Consumption Types")
    plt.tight_layout()
    plt.show()

def rolling_statistics(ts, single_customer, window_size = 6 ):
    rolling_mean = ts.rolling(window=window_size).mean()
    rolling_std = ts.rolling(window=window_size).std()

    plt.figure(figsize=(12, 6))
    plt.plot(ts, label='Original', color='blue')
    plt.plot(rolling_mean, label=f'{window_size}-Month Rolling Mean', color='orange')
    plt.plot(rolling_std, label=f'{window_size}-Month Rolling Std Dev', color='green')
    plt.title(f"Rolling Statistics - PeakConsumption (Customer {single_customer})")
    plt.xlabel("Reporting Month")
    plt.ylabel("kWh")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()