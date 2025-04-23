# main.py
import argparse
import logging
from config_loader import load_config  # your existing config loader module
from data.dataset import ForecastDataset
from programs.pipeline import ForecastPipeline

import pandas as pd

def main():
    # Load forecasting configuration (from config.yaml or database)
    config = load_config("config.yaml") # Forecast config

    parser = argparse.ArgumentParser(description="Forecasting Engine Runner")
    parser.add_argument("--databrick_task_id", type=int, default=1,
                        help="Databrick task ID for forecasting")
    args = parser.parse_args()

    dataset = ForecastDataset(args.databrick_task_id)
    dataset.load_data()
    dataset.parse_identifiers()

    preprocessed_info = dataset.preprocess()

    logging.info(f"Extracted Metadata: {preprocessed_info['metadata']}")

    # Define forecast date range, assuming start_date/end_date come from the config.
    dataset.define_forecast_range()

    pipeline = ForecastPipeline(dataset=dataset,config=config)
    result = pipeline.run()
    logging.info("Forecasting result: %s", result)


if __name__ == '__main__':
    main()
