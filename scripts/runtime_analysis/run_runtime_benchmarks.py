import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_analysis.common import (
    MODEL_ORDER,
    benchmark_single_run,
    build_runtime_census,
    get_group_lookup,
    parse_valid_types_list,
    sample_runtime_pairs,
    save_csv,
)


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime for current per-pod model helpers.")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV input dataset.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODEL_ORDER,
        choices=MODEL_ORDER,
        help="Models to benchmark.",
    )
    parser.add_argument("--samples-per-bucket", type=int, default=20, help="Maximum samples per history/type bucket.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for stratified sampling.")
    parser.add_argument("--forecast-start", default=None, help="Forecast start month in YYYY-MM-DD format.")
    parser.add_argument("--forecast-end", default=None, help="Forecast end month in YYYY-MM-DD format.")
    parser.add_argument("--out-dir", required=True, help="Directory to write runtime outputs.")
    return parser.parse_args(argv)


def _requested_type_counts(valid_type_count: int) -> List[int]:
    return [count for count in (1, 2, 3, 4) if count <= valid_type_count]


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        census_df, df, _ = build_runtime_census(
            csv_path=args.csv_path,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    save_csv(census_df, out_dir / "runtime_census.csv")

    sampled_pairs = sample_runtime_pairs(
        census_df=census_df,
        samples_per_bucket=args.samples_per_bucket,
        random_seed=args.random_seed,
    )
    if sampled_pairs.empty:
        benchmark_df = pd.DataFrame(
            columns=[
                "model",
                "customer_id",
                "pod_id",
                "history_bucket",
                "valid_type_bucket",
                "requested_type_count",
                "available_type_count",
                "valid_types_used",
                "elapsed_sec_total",
                "elapsed_sec_per_type",
                "status",
                "error",
            ]
        )
        save_csv(benchmark_df, out_dir / "runtime_benchmarks.csv")
        logger.warning("No eligible customer-pod pairs found for benchmarking.")
        return 0

    group_lookup = get_group_lookup(df)
    benchmark_rows = []

    for sample_row in sampled_pairs.itertuples(index=False):
        pair_key = (str(sample_row.customer_id), str(sample_row.pod_id))
        pod_df = group_lookup[pair_key]
        valid_types = parse_valid_types_list(sample_row.valid_types_list)
        forecast_start = pd.Timestamp(sample_row.forecast_start)
        forecast_end = pd.Timestamp(sample_row.forecast_end)

        for requested_type_count in _requested_type_counts(int(sample_row.valid_type_count)):
            requested_types = valid_types[:requested_type_count]
            for model_name in args.models:
                result = benchmark_single_run(
                    model_name=model_name,
                    pod_df=pod_df,
                    customer_id=str(sample_row.customer_id),
                    pod_id=str(sample_row.pod_id),
                    valid_types=requested_types,
                    start_date=forecast_start,
                    end_date=forecast_end,
                )
                result.update(
                    {
                        "history_bucket": sample_row.history_bucket,
                        "valid_type_bucket": sample_row.valid_type_bucket,
                        "requested_type_count": requested_type_count,
                        "available_type_count": int(sample_row.valid_type_count),
                        "valid_types_used": "|".join(requested_types),
                    }
                )
                benchmark_rows.append(result)

    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df = benchmark_df.replace({np.inf: np.nan, -np.inf: np.nan})
    out_path = save_csv(benchmark_df, out_dir / "runtime_benchmarks.csv")
    logger.info("Saved runtime benchmarks to %s (%s rows).", out_path, len(benchmark_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
