import warnings
warnings.filterwarnings('ignore')
import re
from db.queries import get_sample_rows, get_user_forecast_data, row_to_config
import os
from etl.etl import *
from dml.dml import *
os.environ["ENV"] = "DEV"

ufm_df = get_user_forecast_data()

all_prediction_columns = ["PeakConsumption", "StandardConsumption", "OffPeakConsumption","Block1Consumption", "Block2Consumption","Block3Consumption", "Block4Consumption",     "NonTOUConsumption"]

metadata = extract_metadata(ufm_df)
customer_ids = parse_json_column(ufm_df, "CustomerJSON")
variable_ids = parse_json_column(ufm_df, "varJSON", key="VariableID")
columns_mapping = generate_combinations(all_prediction_columns)

logging.info(f"Customer IDs: {customer_ids}")
logging.info(f"Variable IDs: {variable_ids}")
logging.info(f"✅ Total column combinations: {len(columns_mapping)}")

print(f"✅ Total combinations: {len(generate_combinations(all_prediction_columns))}\n")

selected_columns = find_matching_combination(columns_mapping, all_prediction_columns)

if selected_columns:
    logging.info(f"Selected Columns: {selected_columns}")
else:
    logging.error("No matching columns found in AllVariables")

df = load_and_prepare_data("PredictiveInputDataARIMA.csv")

row = ufm_df.iloc[0]
config = row_to_config(row)

if df.empty:
    logging.error("🚫 DataFrame is empty. Check input filters or data source.")
else:
    customer_ids, pod_ids = get_unique_list_of_customer_and_pod(df)

    # These variables should come from user input / config
    StartDate = config.start_date
    EndDate = config.end_date
    Hyper_Parameters = config.model_parameters

    forecast_dates = get_forecast_range(StartDate, EndDate)
    arima_order, seasonal_order = extract_sarimax_params(Hyper_Parameters)

    # Extract actuals range
    latest_actual_date = df.index.max()
    logging.info(f"📍 Last actuals month in data: {latest_actual_date.strftime('%Y-%m')}")

import matplotlib.pyplot as plt
import seaborn as sns

# Select a small sample of customers for clearer plots
top_customers = df['CustomerID'].value_counts().head(4).index.tolist()
sample_df = df[df['CustomerID'].isin(top_customers)]
sample_df = df.reset_index()
# Melt the dataframe for easier time series plotting
melted_df = pd.melt(
   sample_df,
    id_vars=['CustomerID', 'ReportingMonth'],
    value_vars=[
        'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
        'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
        'Block4Consumption', 'NonTOUConsumption'
    ],
    var_name='ConsumptionType',
    value_name='kWh'
)
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

# Define consumption columns
consumption_cols = [
    'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
    'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
    'Block4Consumption', 'NonTOUConsumption'
]

# Get top customers for plotting
top_customers = df['CustomerID'].value_counts().head(4).index.tolist()
sample_df = df[df['CustomerID'].isin(top_customers)]
sample_df = df.reset_index()
# Create subplots for each consumption type
fig, axes = plt.subplots(len(consumption_cols), 1, figsize=(16, 28), sharex=True)

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

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Prepare data for a single customer to perform time series diagnostics
df = df.reset_index()
single_customer = df['CustomerID'].value_counts().idxmax()
cust_df = df[df['CustomerID'] == single_customer].sort_values('ReportingMonth')
# Select a single time series (e.g., PeakConsumption)
ts = cust_df.set_index('ReportingMonth')['PeakConsumption']

# 1. ACF and PACF Plots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(ts, ax=axes[0], lags=24)
plot_pacf(ts, ax=axes[1], lags=24)
axes[0].set_title(f"ACF - PeakConsumption ({single_customer})")
axes[1].set_title(f"PACF - PeakConsumption ({single_customer})")
plt.tight_layout()
plt.show()



# 2. Augmented Dickey-Fuller Test for Stationarity
adf_result = adfuller(ts.dropna())
adf_output = {
    'Test Statistic': adf_result[0],
    'p-value': adf_result[1],
    'Lags Used': adf_result[2],
    'Number of Observations': adf_result[3],
    'Critical Values': adf_result[4]
}
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

# 4. Rolling Mean & Variance for PeakConsumption of selected customer
window_size = 6  # 6-month rolling window

ts = cust_df.set_index('ReportingMonth')['PeakConsumption']
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


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 5. Customer Segmentation Based on Consumption Profiles

# Aggregate average consumption per customer
customer_profiles = df.groupby('CustomerID')[consumption_cols].mean()

# Standardize data
scaler = StandardScaler()
scaled_profiles = scaler.fit_transform(customer_profiles)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_profiles)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(scaled_profiles)

# Create DataFrame for visualization
cluster_df = pd.DataFrame({
    'CustomerID': customer_profiles.index,
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'Cluster': clusters
})

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=100)
plt.title("Customer Segmentation Based on Average Consumption")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
