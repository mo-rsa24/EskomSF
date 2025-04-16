# dataset.py

import logging
from typing import Dict, Any, List

import pandas as pd
import os
from data.dml import load_and_prepare_data, get_unique_list_of_customer_and_pod, get_forecast_range
from db.queries import row_to_config, get_user_forecast_data, ForecastConfig
from etl.etl import extract_metadata, parse_json_column, generate_combinations

logger = logging.getLogger(__name__)

class ForecastDataset:
    def __init__(self, databrick_task_id: int):
        self.databrick_task_id = databrick_task_id
        self.ufm_config = self.load_ufm_config()
        self.user_forecast_data: pd.DataFrame()
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.processed_df: pd.DataFrame = pd.DataFrame()
        self.metadata: dict = {}
        self.customer_ids: list = []
        self.variable_ids: list = []
        self.column_combinations: dict = {}
        self.unique_customers: list = []
        self.unique_pod_ids: list = []
        self.forecast_dates: pd.DatetimeIndex = pd.DatetimeIndex([])

    def load_ufm_config(self) -> ForecastConfig:
        """
        Load the user forecast method configuration by retrieving data from the database.

        Returns:
            ForecastConfig: A configuration object created from the first row of the query.

        Raises:
            ValueError: If the data loaded from the database is empty.
        """
        self.user_forecast_data = get_user_forecast_data(self.databrick_task_id)
        if self.user_forecast_data.empty:
            logger.error("🚫 DataFrame is empty while loading UFM config; exiting.")
            raise ValueError("Empty DataFrame loaded from the database for UFM config.")
        ufm_config: ForecastConfig = row_to_config(self.user_forecast_data.iloc[0])
        logger.info(f"✅ Loaded UFM config: {ufm_config}")
        return ufm_config



    def load_data(self) -> None:
        """
        Load and prepare the dataset using the designated dml function. Copies the result
        into both raw_df and processed_df.

        Raises:
            ValueError: If the loaded DataFrame is empty.
        """
        self.raw_df = load_and_prepare_data(self.ufm_config)
        if self.raw_df.empty:
            logger.error("🚫 DataFrame is empty. Check input filters or data source.")
            raise ValueError("Empty DataFrame loaded from source.")
        self.processed_df = self.raw_df.copy()
        logger.info("✅ Data loaded and initial copy created.")

    def extract_metadata(self) -> Dict[str, Any]:
        """
        Extract forecasting metadata from the raw DataFrame.

        Returns:
            Dict[str, Any]: A dictionary containing metadata (e.g., forecast method, parameters, date range).

        Raises:
            ValueError: If raw data has not been loaded.
        """
        if self.raw_df.empty:
            logger.error("🚫 Data not loaded. Call load_data() first.")
            raise ValueError("Data not loaded.")
        self.metadata = extract_metadata(self.raw_df)
        logger.info(f"🔍 Extracted metadata: {self.metadata}")
        return self.metadata

    def parse_identifiers(self) -> (List[str], List[str]):
        """
        Parse JSON columns to extract Customer and Variable IDs.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing lists of customer IDs and variable IDs.

        Raises:
            ValueError: If raw data has not been loaded.
        """
        if self.user_forecast_data.empty:
            logger.error("🚫 Data not loaded. Cannot parse identifiers.")
            raise ValueError("Data not loaded.")
        self.customer_ids = parse_json_column(self.user_forecast_data, "CustomerJSON")
        self.variable_ids = parse_json_column(self.user_forecast_data, "varJSON", key="VariableID")
        logger.info(f"🔍 Parsed Customer IDs: {self.customer_ids} and Variable IDs: {self.variable_ids}")
        return self.customer_ids, self.variable_ids

    def generate_column_combinations(self) -> Dict[Any, List[str]]:
        """
        Generate all non-empty combinations of consumption columns using ETL logic.

        Returns:
            Dict[Any, List[str]]: Mapping of column combination keys to their corresponding list of columns.
        """
        self.column_combinations = generate_combinations()
        logger.info(f"✅ Generated {len(self.column_combinations)} column combinations.")
        return self.column_combinations

    def extract_unique_customers_and_pods(self) -> (List[str], List[str]):
        """
        Extract unique customer and pod IDs from the processed DataFrame.

        Returns:
            Tuple[List[str], List[str]]: Unique customer IDs and pod IDs.

        Raises:
            ValueError: If processed data is not available.
        """
        if self.processed_df.empty:
            logger.error("🚫 Processed data not available. Call load_data() first.")
            raise ValueError("No processed data available.")
        self.unique_customers, self.unique_pod_ids = get_unique_list_of_customer_and_pod(self.processed_df)
        logger.info(f"✅ Unique Customers: {self.unique_customers}, Unique Pods: {self.unique_pod_ids}")
        return self.unique_customers, self.unique_pod_ids

    def define_forecast_range(self) -> pd.DatetimeIndex:
        """
        Define the forecast date range using start_date and end_date from the ForecastConfig.

        Returns:
            pd.DatetimeIndex: The defined forecast dates.
        """
        try:
            self.forecast_dates = get_forecast_range(self.ufm_config)
            logger.info(f"📅 Forecast range defined: {self.forecast_dates[0]} to {self.forecast_dates[-1]}")
        except Exception as e:
            logger.error(f"🚫 Failed to define forecast range: {e}")
            self.forecast_dates = pd.DatetimeIndex([])
        return self.forecast_dates

    def preprocess(self) -> Dict[str, Any]:
        """
        Perform full preprocessing by executing the following steps:
          - Make a copy of raw data into processed data.
          - Extract metadata.
          - Parse JSON columns for customer and variable identifiers.
          - Generate column combinations.
          - Extract unique customer and pod IDs.

        Returns:
            Dict[str, Any]: A dictionary with all the extracted information.
        """
        # Ensure raw data is loaded before preprocessing.
        if self.raw_df.empty:
            logger.error("🚫 Raw data not loaded. Call load_data() first.")
            raise ValueError("Raw data not loaded.")

        try:
            self.metadata = extract_metadata(self.user_forecast_data)
            logger.info(f"✅ Extracted metadata: {self.metadata}")
        except Exception as e:
            logger.error(f"🚫 Failed to extract metadata: {e}")
            self.metadata = {}

        self.customer_ids = parse_json_column(self.raw_df, "CustomerJSON")
        self.variable_ids = parse_json_column(self.raw_df, "varJSON", key="VariableID")
        self.column_combinations = generate_combinations()
        self.unique_customers, self.unique_pod_ids = get_unique_list_of_customer_and_pod(self.raw_df)

        logger.info(f"🔍 Customer IDs: {self.customer_ids}")
        logger.info(f"🔍 Variable IDs: {self.variable_ids}")
        logger.info(f"✅ Generated {len(self.column_combinations)} column combinations.")
        logger.info(f"✅ Unique Customers: {self.unique_customers}")
        logger.info(f"✅ Unique Pods: {self.unique_pod_ids}")

        return {
            "metadata": self.metadata,
            "customer_ids": self.customer_ids,
            "variable_ids": self.variable_ids,
            "column_combinations": self.column_combinations,
            "unique_customers": self.unique_customers,
            "unique_pods": self.unique_pod_ids
        }
