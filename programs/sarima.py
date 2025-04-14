import warnings
warnings.filterwarnings('ignore')
from db.queries import get_user_forecast_data, row_to_config
from db.utilities import env, logger
from etl.etl import *
from modeling.autoarima import *

logger.info(f"Running on {env}")

databrick_task_id = 2

ufm_df = get_user_forecast_data(databrick_task_id = databrick_task_id)
row = ufm_df.iloc[0]
ufm_config = row_to_config(row)

metadata = extract_metadata(ufm_df)
customer_ids = parse_json_column(ufm_df, "CustomerJSON")
variable_ids = parse_json_column(ufm_df, "varJSON", key="VariableID")
columns_mapping = generate_combinations()

logging.info(f"Customer IDs: {customer_ids}")
logging.info(f"Variable IDs: {variable_ids}")
logging.info(f"✅ Total column combinations: {len(columns_mapping)}")

selected_columns = find_matching_combination(columns_mapping)

df = load_and_prepare_data(ufmd=ufm_config.user_forecast_method_id, method= ufm_config.forecast_method_name, environment=env)

if df.empty:
    logging.error("🚫 DataFrame is empty. Check input filters or data source.")
    exit()

customer_ids, pod_ids = get_unique_list_of_customer_and_pod(df)

# These variables should come from user input / config
StartDate = ufm_config.start_date
EndDate = ufm_config.end_date
Hyper_Parameters = ufm_config.model_parameters

forecast_dates = get_forecast_range(StartDate, EndDate)
arima_order, seasonal_order = extract_sarimax_params(Hyper_Parameters)

# Extract actuals range
latest_actual_date = df.index.max()
logging.info(f"📍 Last actuals month in data: {latest_actual_date.strftime('%Y-%m')}")
# Filter the DataFrame for the specific customer.
customer_id = '5181432365'


if df.empty:
    logging.error(f"Customer ID {customer_id} does not exist in the dataset. Skipping forecast.")
    forecast_data = None  # Or handle as needed (e.g., return an empty DataFrame)
else:
    forecast_data = forecast_arima_for_single_customer(
        df,
        customer_id,
        ufm_config,
        selected_columns=['Block1Consumption'],
        consumption_types = ['Block1Consumption'],
        order=arima_order,
        seasonal_order=seasonal_order,
        mode='validation'
    )
    print()