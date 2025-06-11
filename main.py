import argparse
import logging

from config_loader import load_config  # your existing config loader module
from data.dataset import ForecastDataset
from profiler.profiler_run import run_context
from programs.pipeline import ForecastPipeline

from diagnostics import run_stationarity_tests, plot_stl_decomposition
from data.dml import get_single_time_series_for_single_customer


from visualization.exploration.behaviour import plot_raw_series, facet_consumption_profiles, plot_overlay_years, \
    plot_top_consumers, plot_missingness_heatmap



def main():
    # Load forecasting configuration (from config.yaml or database)
    config = load_config("config.yaml") # Forecast config

    # ✅ Initialize global run context ONCE
    run_context.init(app_name="forecasting-engine")

    parser = argparse.ArgumentParser(description="Forecasting Engine Runner")
    parser.add_argument("--databrick_task_id", type=int, default=1,
                        help="Databrick task ID for forecasting")
    args = parser.parse_args()

    dataset = ForecastDataset(args.databrick_task_id, config.save)
    dataset.load_data()

    preprocessed_info = dataset.preprocess()

    df = dataset.processed_df
    if config.visualize:
        try:
            customer_id = df['CustomerID'].value_counts().idxmax()
            logging.info(f"🎨 Visualizing raw time series for top customer: {customer_id}")
            plot_raw_series(df, customer_id, columns=['PeakConsumption', 'StandardConsumption','OffPeakConsumption'])
            facet_consumption_profiles(df, customer_id, columns=['PeakConsumption', 'StandardConsumption','OffPeakConsumption'])
            plot_overlay_years(df, customer_id, columns=['PeakConsumption', 'StandardConsumption','OffPeakConsumption'])
            plot_top_consumers(df, columns=['PeakConsumption', 'StandardConsumption','OffPeakConsumption'], top_n=5)
            plot_missingness_heatmap(df)
        except Exception as e:
            logging.warning(f"⚠️ Skipping visual diagnostics due to error: {e}")
    if config.debug:
        try:
            # Example integration
            ts, customer_id, cust_df = get_single_time_series_for_single_customer(dataset.raw_df)

            result = run_stationarity_tests(ts, title=f"Customer {customer_id}")
            print("ADF Result:", result['ADF'])
            print("KPSS Result:", result['KPSS'])

            plot_stl_decomposition(ts, period=12, title=f"Customer {customer_id}")
        except Exception as e:
            logging.warning(f"⚠️ Skipping statistical diagnostics due to error: {e}")


    logging.info(f"Extracted Metadata: {preprocessed_info['metadata']}")

    # Define forecast date range, assuming start_date/end_date come from the config.
    dataset.define_forecast_range()

    pipeline = ForecastPipeline(dataset=dataset,config=config)
    model_performance, forecast_combined_df = pipeline.run()
    print('Done.')
    # forecast_combined_df.to_csv("output.csv")
    # logging.info("Forecasting result: %s", model_performance)


if __name__ == '__main__':
    main()
