import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_analysis.common import MODEL_ORDER


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot runtime scenario charts from benchmark and estimate outputs.")
    parser.add_argument("--benchmarks", required=True, help="Path to runtime_benchmarks.csv.")
    parser.add_argument("--estimates", required=True, help="Path to runtime_estimates.csv.")
    parser.add_argument("--out-dir", required=True, help="Directory to write plot PNGs and copied CSVs.")
    return parser.parse_args(argv)


def _require_plotting():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to create runtime charts.") from exc

    try:
        import seaborn as sns
    except ModuleNotFoundError:  # pragma: no cover - environment dependent
        sns = None

    return plt, sns


def _copy_if_present(path: Path, out_dir: Path) -> None:
    if path.exists():
        target = out_dir / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(args.benchmarks)
    estimate_path = Path(args.estimates)
    try:
        benchmark_df = pd.read_csv(benchmark_path)
        estimate_df = pd.read_csv(estimate_path)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    _copy_if_present(benchmark_path, out_dir)
    _copy_if_present(estimate_path, out_dir)

    census_candidates = [
        benchmark_path.parent / "runtime_census.csv",
        estimate_path.parent / "runtime_census.csv",
    ]
    for candidate in census_candidates:
        if candidate.exists():
            _copy_if_present(candidate, out_dir)
            break

    plt, sns = _require_plotting()

    scenario_df = estimate_df[
        (estimate_df["row_type"] == "scenario") &
        (estimate_df["estimate_kind"] == "p50")
    ].copy()
    if not scenario_df.empty:
        pivot = scenario_df.pivot(index="scenario_name", columns="model", values="projected_sec")
        pivot = pivot.reindex(columns=[name for name in MODEL_ORDER if name in pivot.columns])
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_title("Projected Runtime By Scenario")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Projected runtime (seconds)")
        plt.tight_layout()
        plt.savefig(out_dir / "runtime_bar_by_scenario.png", dpi=150)
        plt.close()

    dist_df = benchmark_df[
        (benchmark_df["status"] == "completed") &
        benchmark_df["elapsed_sec_total"].notna()
    ].copy()
    if not dist_df.empty:
        plt.figure(figsize=(12, 6))
        if sns is not None:
            sns.violinplot(
                data=dist_df,
                x="model",
                y="elapsed_sec_total",
                hue="requested_type_count",
                cut=0,
            )
        else:  # pragma: no cover - only used when seaborn is missing
            positions = []
            labels = []
            series = []
            counter = 1
            for model_name in MODEL_ORDER:
                model_df = dist_df[dist_df["model"] == model_name]
                if model_df.empty:
                    continue
                series.append(model_df["elapsed_sec_total"].to_numpy())
                positions.append(counter)
                labels.append(model_name)
                counter += 1
            plt.boxplot(series, positions=positions)
            plt.xticks(positions, labels)
        plt.title("Observed Per-Pod Runtime Distribution")
        plt.xlabel("Model")
        plt.ylabel("Elapsed time (seconds)")
        plt.tight_layout()
        plt.savefig(out_dir / "runtime_distribution_by_model.png", dpi=150)
        plt.close()

    curve_df = estimate_df[
        (estimate_df["row_type"] == "pod_curve") &
        (estimate_df["estimate_kind"] == "p50")
    ].copy()
    if not curve_df.empty:
        available_models = [name for name in MODEL_ORDER if name in curve_df["model"].unique()]
        fig, axes = plt.subplots(len(available_models), 1, figsize=(12, 4 * max(len(available_models), 1)), squeeze=False)
        for row_idx, model_name in enumerate(available_models):
            ax = axes[row_idx, 0]
            model_df = curve_df[curve_df["model"] == model_name]
            for type_count in (1, 3, 8):
                series = model_df[model_df["requested_type_count"] == type_count]
                if series.empty:
                    continue
                ax.plot(series["pod_count"], series["projected_sec"], label=f"{type_count} types")
            ax.set_title(f"Projected Runtime Vs Pod Count: {model_name}")
            ax.set_xlabel("Pod count")
            ax.set_ylabel("Projected runtime (seconds)")
            ax.legend()
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "runtime_line_pods_vs_runtime.png", dpi=150)
        plt.close()

    logger.info("Saved runtime charts to %s.", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
