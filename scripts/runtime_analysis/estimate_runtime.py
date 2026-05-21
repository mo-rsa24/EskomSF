import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_analysis.common import (
    HISTORY_BUCKET_ORDER,
    MODEL_ORDER,
    build_successful_benchmark_summary,
    parse_valid_types_list,
    save_csv,
)


logger = logging.getLogger(__name__)

COMMON_HISTORY_BUCKET = "24-35"
SCENARIOS = [
    ("1pod_1type_short", 1, 1, "12-23"),
    ("1pod_3type_common", 1, 3, "24-35"),
    ("5pod_3type_common", 5, 3, "24-35"),
    ("20pod_3type_common", 20, 3, "24-35"),
    ("1pod_8type_worst", 1, 8, "24-35"),
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate runtime scenarios from census and benchmark data.")
    parser.add_argument("--census", required=True, help="Path to runtime_census.csv.")
    parser.add_argument("--benchmarks", required=True, help="Path to runtime_benchmarks.csv.")
    parser.add_argument("--out-dir", required=True, help="Directory to write runtime_estimates.csv.")
    return parser.parse_args(argv)


def _quantile_lookup(summary_df: pd.DataFrame) -> Dict[Tuple[str, str, int], Dict[str, float]]:
    lookup: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for row in summary_df.itertuples(index=False):
        lookup[(row.model, row.history_bucket, int(row.requested_type_count))] = {
            "p50": float(row.p50_sec),
            "p90": float(row.p90_sec),
            "sample_count": int(row.sample_count),
        }
    return lookup


def _fit_line(x_values: np.ndarray, y_values: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(x_values)) >= 2:
        slope, intercept = np.polyfit(x_values, y_values, deg=1)
        return float(intercept), float(slope)
    if len(x_values) == 1 and x_values[0] > 0:
        return 0.0, float(y_values[0] / x_values[0])
    if len(x_values) > 1:
        ratio = np.median(y_values / np.maximum(x_values, 1))
        return 0.0, float(ratio)
    return 0.0, np.nan


def _build_linear_fits(successful_df: pd.DataFrame) -> Dict[Tuple[str, Optional[str]], Dict[str, float]]:
    fits: Dict[Tuple[str, Optional[str]], Dict[str, float]] = {}

    for model_name, model_df in successful_df.groupby("model", sort=False):
        model_x = model_df["requested_type_count"].to_numpy(dtype=float)
        model_y = model_df["elapsed_sec_total"].to_numpy(dtype=float)
        intercept, slope = _fit_line(model_x, model_y)
        fits[(model_name, None)] = {
            "intercept": intercept,
            "slope": slope,
            "source": "model_level_fit",
        }

        for history_bucket, bucket_df in model_df.groupby("history_bucket", sort=False):
            bucket_x = bucket_df["requested_type_count"].to_numpy(dtype=float)
            bucket_y = bucket_df["elapsed_sec_total"].to_numpy(dtype=float)
            if len(bucket_df) >= 2:
                intercept, slope = _fit_line(bucket_x, bucket_y)
                source = "history_bucket_fit"
            else:
                model_fit = fits[(model_name, None)]
                intercept = model_fit["intercept"]
                slope = model_fit["slope"]
                source = "model_level_fit"
            fits[(model_name, history_bucket)] = {
                "intercept": float(intercept),
                "slope": float(slope),
                "source": source,
            }

    return fits


def _predict_per_pod_runtime(
    model_name: str,
    history_bucket: str,
    requested_type_count: int,
    quantile_name: str,
    quantiles: Dict[Tuple[str, str, int], Dict[str, float]],
    fits: Dict[Tuple[str, Optional[str]], Dict[str, float]],
) -> Tuple[float, str]:
    exact = quantiles.get((model_name, history_bucket, int(requested_type_count)))
    if exact:
        return float(exact[quantile_name]), "empirical_exact"

    fit = fits.get((model_name, history_bucket)) or fits.get((model_name, None))
    if not fit or np.isnan(fit["slope"]):
        return np.nan, "no_successful_benchmarks"

    predicted = fit["intercept"] + fit["slope"] * requested_type_count
    return float(max(predicted, 0.0)), fit["source"]


def _scenario_rows(
    quantiles: Dict[Tuple[str, str, int], Dict[str, float]],
    fits: Dict[Tuple[str, Optional[str]], Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for scenario_name, pod_count, requested_type_count, history_bucket in SCENARIOS:
            for quantile_name in ("p50", "p90"):
                per_pod, source = _predict_per_pod_runtime(
                    model_name=model_name,
                    history_bucket=history_bucket,
                    requested_type_count=requested_type_count,
                    quantile_name=quantile_name,
                    quantiles=quantiles,
                    fits=fits,
                )
                rows.append(
                    {
                        "row_type": "scenario",
                        "scenario_name": scenario_name,
                        "model": model_name,
                        "history_bucket": history_bucket,
                        "requested_type_count": requested_type_count,
                        "pod_count": pod_count,
                        "estimate_kind": quantile_name,
                        "projected_sec": per_pod * pod_count if pd.notna(per_pod) else np.nan,
                        "source": source,
                    }
                )
    return rows


def _dataset_total_rows(
    census_df: pd.DataFrame,
    quantiles: Dict[Tuple[str, str, int], Dict[str, float]],
    fits: Dict[Tuple[str, Optional[str]], Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    eligible = census_df[census_df["valid_type_count"] > 0].copy()
    for model_name in MODEL_ORDER:
        for quantile_name in ("p50", "p90"):
            per_pair_values = []
            sources = set()
            for row in eligible.itertuples(index=False):
                per_pod, source = _predict_per_pod_runtime(
                    model_name=model_name,
                    history_bucket=row.history_bucket,
                    requested_type_count=int(row.valid_type_count),
                    quantile_name=quantile_name,
                    quantiles=quantiles,
                    fits=fits,
                )
                if pd.notna(per_pod):
                    per_pair_values.append(per_pod)
                    sources.add(source)
            rows.append(
                {
                    "row_type": "dataset_total",
                    "scenario_name": "dataset_empirical_mix",
                    "model": model_name,
                    "history_bucket": "mixed",
                    "requested_type_count": np.nan,
                    "pod_count": int(len(eligible)),
                    "estimate_kind": quantile_name,
                    "projected_sec": float(np.sum(per_pair_values)) if per_pair_values else np.nan,
                    "source": ",".join(sorted(sources)) if sources else "no_successful_benchmarks",
                }
            )
    return rows


def _pod_curve_rows(
    quantiles: Dict[Tuple[str, str, int], Dict[str, float]],
    fits: Dict[Tuple[str, Optional[str]], Dict[str, float]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for requested_type_count in (1, 3, 8):
            for pod_count in range(1, 101):
                per_pod, source = _predict_per_pod_runtime(
                    model_name=model_name,
                    history_bucket=COMMON_HISTORY_BUCKET,
                    requested_type_count=requested_type_count,
                    quantile_name="p50",
                    quantiles=quantiles,
                    fits=fits,
                )
                rows.append(
                    {
                        "row_type": "pod_curve",
                        "scenario_name": f"curve_{requested_type_count}type",
                        "model": model_name,
                        "history_bucket": COMMON_HISTORY_BUCKET,
                        "requested_type_count": requested_type_count,
                        "pod_count": pod_count,
                        "estimate_kind": "p50",
                        "projected_sec": per_pod * pod_count if pd.notna(per_pod) else np.nan,
                        "source": source,
                    }
                )
    return rows


def _empirical_rows(summary_df: pd.DataFrame) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in summary_df.itertuples(index=False):
        rows.append(
            {
                "row_type": "empirical_quantile",
                "scenario_name": "",
                "model": row.model,
                "history_bucket": row.history_bucket,
                "requested_type_count": int(row.requested_type_count),
                "pod_count": 1,
                "estimate_kind": "p50",
                "projected_sec": float(row.p50_sec),
                "source": "empirical_exact",
            }
        )
        rows.append(
            {
                "row_type": "empirical_quantile",
                "scenario_name": "",
                "model": row.model,
                "history_bucket": row.history_bucket,
                "requested_type_count": int(row.requested_type_count),
                "pod_count": 1,
                "estimate_kind": "p90",
                "projected_sec": float(row.p90_sec),
                "source": "empirical_exact",
            }
        )
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        census_df = pd.read_csv(args.census, parse_dates=["first_month", "last_month", "forecast_start", "forecast_end"])
        benchmark_df = pd.read_csv(args.benchmarks)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    successful_df = benchmark_df[
        (benchmark_df["status"] == "completed") &
        benchmark_df["elapsed_sec_total"].notna()
    ].copy()
    summary_df = build_successful_benchmark_summary(benchmark_df)
    quantiles = _quantile_lookup(summary_df)
    fits = _build_linear_fits(successful_df) if not successful_df.empty else {}

    rows: List[Dict[str, object]] = []
    rows.extend(_empirical_rows(summary_df))
    rows.extend(_scenario_rows(quantiles, fits))
    rows.extend(_dataset_total_rows(census_df, quantiles, fits))
    rows.extend(_pod_curve_rows(quantiles, fits))

    estimate_df = pd.DataFrame(rows)
    out_path = save_csv(estimate_df, out_dir / "runtime_estimates.csv")
    logger.info("Saved runtime estimates to %s (%s rows).", out_path, len(estimate_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
