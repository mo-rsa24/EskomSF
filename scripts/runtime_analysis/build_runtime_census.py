import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.runtime_analysis.common import build_runtime_census, save_csv


logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a runtime census for per-pod forecasting.")
    parser.add_argument("--csv-path", required=True, help="Path to the CSV input dataset.")
    parser.add_argument("--forecast-start", default=None, help="Forecast start month in YYYY-MM-DD format.")
    parser.add_argument("--forecast-end", default=None, help="Forecast end month in YYYY-MM-DD format.")
    parser.add_argument("--out-dir", required=True, help="Directory to write runtime_census.csv.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        census_df, _, _ = build_runtime_census(
            csv_path=args.csv_path,
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    out_path = save_csv(census_df, out_dir / "runtime_census.csv")
    logger.info("Saved runtime census to %s (%s rows).", out_path, len(census_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
