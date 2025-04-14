import warnings

from modeling.XGBoost import train_xgboost_for_single_customer

warnings.filterwarnings('ignore')
from db.queries import get_user_forecast_data, row_to_config
from db.utilities import env, logger
from etl.etl import *
from dml.dml import *

logger.info(f"Running on {env}")

databrick_task_id = 67

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
random_forest_hyperparameters = extract_random_forest_params(Hyper_Parameters)

# Extract actuals range
latest_actual_date = df.index.max()
logging.info(f"📍 Last actuals month in data: {latest_actual_date.strftime('%Y-%m')}")

# Filter the DataFrame for the specific customer.
customer_id= '8460296087'



if df.empty:
    logging.error(f"Customer ID {customer_id} does not exist in the dataset. Skipping forecast.")
    forecast_data = None  # Or handle as needed (e.g., return an empty DataFrame)
else:
    train_xgboost_for_single_customer(df, customer_id, ufm_config,
    lag_features = ['StandardConsumption'],
    selected_columns = ['StandardConsumption'],
    consumption_types = ['StandardConsumption'],
    mode ='train')

    print()