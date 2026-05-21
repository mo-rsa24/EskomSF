import importlib
import logging
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
CONSUMPTION_COLUMNS = [
    "OffPeakConsumption",
    "PeakConsumption",
    "StandardConsumption",
    "Block1Consumption",
    "Block2Consumption",
    "Block3Consumption",
    "Block4Consumption",
    "NonTOUConsumption",
]
MODEL_ORDER = ["arima", "sarima", "randomforest", "xgboost"]
HISTORY_BUCKET_ORDER = ["<12", "12-23", "24-35", "36-59", "60+"]
VALID_TYPE_BUCKET_ORDER = ["0", "1", "2", "3", "4+"]
POD_BUCKET_ORDER = ["1", "2-5", "6-20", "21+"]


@dataclass(frozen=True)
class RuntimeForecastConfig:
    forecast_method_id: int
    forecast_method_name: str
    model_parameters: str
    region: str
    status: str
    user_forecast_method_id: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    databrick_task_id: int


def ensure_repo_root_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def suppress_non_error_logs():
    previous_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(previous_disable)


def bootstrap_optional_dependencies() -> None:
    ensure_repo_root_on_path()

    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")

        def load_dotenv(*args, **kwargs):
            return False

        dotenv.load_dotenv = load_dotenv
        sys.modules["dotenv"] = dotenv

    if "pyspark" not in sys.modules:
        pyspark = types.ModuleType("pyspark")
        pyspark_sql = types.ModuleType("pyspark.sql")

        class SparkSession:  # pragma: no cover - trivial shim
            pass

        pyspark_sql.SparkSession = SparkSession
        pyspark.sql = pyspark_sql
        sys.modules["pyspark"] = pyspark
        sys.modules["pyspark.sql"] = pyspark_sql

    if "shap" not in sys.modules:
        sys.modules["shap"] = types.ModuleType("shap")

    if "holidays" not in sys.modules:
        holidays = types.ModuleType("holidays")

        class _EmptyHolidayCalendar(dict):  # pragma: no cover - trivial shim
            def keys(self):
                return []

        def SouthAfrica(*args, **kwargs):
            return _EmptyHolidayCalendar()

        holidays.SouthAfrica = SouthAfrica
        sys.modules["holidays"] = holidays

    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")

        class _NoOpAxes:
            def plot(self, *args, **kwargs):
                return []

            def set_title(self, *args, **kwargs):
                return None

            def set_ylabel(self, *args, **kwargs):
                return None

            def set_xlabel(self, *args, **kwargs):
                return None

            def tick_params(self, *args, **kwargs):
                return None

            def grid(self, *args, **kwargs):
                return None

            def legend(self, *args, **kwargs):
                return None

        class _NoOpFigure:
            def delaxes(self, *args, **kwargs):
                return None

        def _subplots(nrows=1, ncols=1, figsize=None, squeeze=True):
            fig = _NoOpFigure()
            if squeeze and nrows == 1 and ncols == 1:
                return fig, _NoOpAxes()
            axes = np.empty((nrows, ncols), dtype=object)
            for row in range(nrows):
                for col in range(ncols):
                    axes[row, col] = _NoOpAxes()
            return fig, axes

        pyplot.figure = lambda *args, **kwargs: _NoOpFigure()
        pyplot.subplots = _subplots
        pyplot.plot = lambda *args, **kwargs: []
        pyplot.show = lambda *args, **kwargs: None
        pyplot.tight_layout = lambda *args, **kwargs: None
        pyplot.title = lambda *args, **kwargs: None
        pyplot.xlabel = lambda *args, **kwargs: None
        pyplot.ylabel = lambda *args, **kwargs: None
        pyplot.legend = lambda *args, **kwargs: None
        pyplot.xticks = lambda *args, **kwargs: None

        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot

    if "xgboost" not in sys.modules:
        xgboost = types.ModuleType("xgboost")

        class XGBRegressor:  # pragma: no cover - trivial shim
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError("xgboost is required to benchmark xgboost.")

        xgboost.XGBRegressor = XGBRegressor
        sys.modules["xgboost"] = xgboost


def load_runtime_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_consumption_types() -> List[str]:
    return list(load_runtime_config().get("consumption_types", CONSUMPTION_COLUMNS))


def normalize_month_start(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def resolve_forecast_window(
    df: pd.DataFrame,
    forecast_start: Optional[str] = None,
    forecast_end: Optional[str] = None,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if forecast_start:
        start = normalize_month_start(pd.Timestamp(forecast_start))
    else:
        start = normalize_month_start(df["ReportingMonth"].max() + pd.DateOffset(months=1))

    if forecast_end:
        end = normalize_month_start(pd.Timestamp(forecast_end))
    else:
        end = normalize_month_start(start + pd.DateOffset(months=11))

    if end < start:
        raise ValueError("forecast_end must be greater than or equal to forecast_start")

    return start, end


def build_forecast_horizon(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="MS")


def load_csv_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    required_columns = {"CustomerID", "PodID", "ReportingMonth"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(missing_columns)}"
        )

    if "UserForecastMethodID" in df.columns and df["UserForecastMethodID"].nunique(dropna=False) == 1:
        df = df.drop(columns=["UserForecastMethodID"])

    df["CustomerID"] = df["CustomerID"].astype(str)
    df["PodID"] = df["PodID"].astype(str)
    df["ReportingMonth"] = pd.to_datetime(df["ReportingMonth"]).dt.to_period("M").dt.to_timestamp()
    df = df.sort_values(["CustomerID", "PodID", "ReportingMonth"]).reset_index(drop=True)
    return df


def to_indexed_pod_frame(group_df: pd.DataFrame) -> pd.DataFrame:
    pod_df = group_df.copy()
    pod_df["ReportingMonth"] = pd.to_datetime(pod_df["ReportingMonth"]).dt.to_period("M").dt.to_timestamp()
    pod_df = pod_df.sort_values("ReportingMonth").set_index("ReportingMonth")
    return pod_df


def history_bucket_from_rows(n_rows: int) -> str:
    if n_rows < 12:
        return "<12"
    if n_rows < 24:
        return "12-23"
    if n_rows < 36:
        return "24-35"
    if n_rows < 60:
        return "36-59"
    return "60+"


def valid_type_bucket_from_count(count: int) -> str:
    if count >= 4:
        return "4+"
    return str(max(count, 0))


def pod_bucket_from_count(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return "21+"


def build_runtime_forecast_config(
    model_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> RuntimeForecastConfig:
    return RuntimeForecastConfig(
        forecast_method_id=0,
        forecast_method_name=model_name,
        model_parameters="",
        region="local",
        status="runtime-analysis",
        user_forecast_method_id=0,
        start_date=start_date,
        end_date=end_date,
        databrick_task_id=0,
    )


def parse_valid_types_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def dependency_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def availability_for_model(model_name: str) -> Tuple[bool, Optional[str]]:
    if model_name == "xgboost" and not dependency_available("xgboost"):
        return False, "xgboost is not installed"
    return True, None


def get_validation_functions():
    bootstrap_optional_dependencies()
    ensure_repo_root_on_path()
    from profiler.errors.validation import invalid_forecast_horizon, invalid_length, invalid_series

    return invalid_series, invalid_length, invalid_forecast_horizon


def get_process_reporting_months():
    bootstrap_optional_dependencies()
    ensure_repo_root_on_path()
    from models.algorithms.utilities import process_reporting_months

    return process_reporting_months


def evaluate_group_runtime_shape(
    customer_id: str,
    pod_id: str,
    pod_df: pd.DataFrame,
    forecast_horizon: pd.DatetimeIndex,
    consumption_types: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    process_reporting_months = get_process_reporting_months()
    invalid_series, invalid_length, invalid_forecast_horizon = get_validation_functions()
    ordered_types = list(consumption_types or get_consumption_types())

    with suppress_non_error_logs():
        processed_df = process_reporting_months(pod_df.copy())
        active_types: List[str] = []
        valid_types: List[str] = []
        for column in ordered_types:
            if column not in processed_df.columns:
                continue

            series = pd.to_numeric(processed_df[column], errors="coerce")
            processed_df[column] = series

            if series.fillna(0).sum() > 0:
                active_types.append(column)

            if invalid_series(pod_id, processed_df[column], column):
                continue
            if invalid_length(processed_df[column], column):
                continue
            if invalid_forecast_horizon(pod_id, processed_df[column], column, forecast_horizon):
                continue
            valid_types.append(column)

    return {
        "active_type_count": len(active_types),
        "active_types": active_types,
        "valid_type_count": len(valid_types),
        "valid_types": valid_types,
    }


def build_runtime_census(
    csv_path: str,
    forecast_start: Optional[str] = None,
    forecast_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    df = load_csv_dataset(csv_path)
    start_date, end_date = resolve_forecast_window(df, forecast_start, forecast_end)
    forecast_horizon = build_forecast_horizon(start_date, end_date)

    pods_per_customer = df.groupby("CustomerID")["PodID"].nunique().to_dict()
    rows: List[Dict[str, object]] = []

    for (customer_id, pod_id), group_df in df.groupby(["CustomerID", "PodID"], sort=False):
        pod_df = to_indexed_pod_frame(group_df)
        shape = evaluate_group_runtime_shape(customer_id, pod_id, pod_df, forecast_horizon)
        n_rows = len(group_df)
        rows.append(
            {
                "customer_id": customer_id,
                "pod_id": pod_id,
                "n_rows": n_rows,
                "first_month": normalize_month_start(group_df["ReportingMonth"].min()),
                "last_month": normalize_month_start(group_df["ReportingMonth"].max()),
                "history_bucket": history_bucket_from_rows(n_rows),
                "active_type_count": shape["active_type_count"],
                "valid_type_count": shape["valid_type_count"],
                "valid_types_list": "|".join(shape["valid_types"]),
                "pods_per_customer": int(pods_per_customer[customer_id]),
                "pod_bucket": pod_bucket_from_count(int(pods_per_customer[customer_id])),
                "valid_type_bucket": valid_type_bucket_from_count(shape["valid_type_count"]),
                "forecast_start": start_date,
                "forecast_end": end_date,
            }
        )

    census_df = pd.DataFrame(rows)
    if not census_df.empty:
        census_df = census_df.sort_values(
            ["customer_id", "pod_id"],
            kind="stable",
        ).reset_index(drop=True)
    return census_df, df, forecast_horizon


def save_csv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def sample_runtime_pairs(
    census_df: pd.DataFrame,
    samples_per_bucket: int,
    random_seed: int,
) -> pd.DataFrame:
    eligible = census_df[census_df["valid_type_count"] > 0].copy()
    if eligible.empty:
        return eligible

    rng = np.random.default_rng(random_seed)
    sampled_frames = []
    for _, group_df in eligible.groupby(["history_bucket", "valid_type_bucket"], sort=False):
        take = min(samples_per_bucket, len(group_df))
        if take == len(group_df):
            sampled_frames.append(group_df.copy())
            continue
        chosen = rng.choice(group_df.index.to_numpy(), size=take, replace=False)
        sampled_frames.append(group_df.loc[chosen].copy())

    sampled = pd.concat(sampled_frames, ignore_index=True)
    sampled = sampled.sort_values(
        ["history_bucket", "valid_type_bucket", "customer_id", "pod_id"],
        kind="stable",
    ).reset_index(drop=True)
    return sampled


def get_group_lookup(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    groups: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, group_df in df.groupby(["CustomerID", "PodID"], sort=False):
        groups[key] = to_indexed_pod_frame(group_df)
    return groups


def _import_tree_benchmark(module_name: str):
    bootstrap_optional_dependencies()
    ensure_repo_root_on_path()
    module = importlib.import_module(module_name)
    return module.forecast_for_podel_id


def _import_arima_benchmark():
    bootstrap_optional_dependencies()
    ensure_repo_root_on_path()
    module = importlib.import_module("models.algorithms.autoarima")
    return module.forecast_for_podel_id


def get_model_benchmark_callable(model_name: str):
    if model_name in {"arima", "sarima"}:
        return _import_arima_benchmark()
    if model_name == "randomforest":
        return _import_tree_benchmark("models.algorithms.tree_algorithms.rf")
    if model_name == "xgboost":
        return _import_tree_benchmark("models.algorithms.tree_algorithms.xgb")
    raise ValueError(f"Unsupported model: {model_name}")


def benchmark_single_run(
    model_name: str,
    pod_df: pd.DataFrame,
    customer_id: str,
    pod_id: str,
    valid_types: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> Dict[str, object]:
    available, dependency_error = availability_for_model(model_name)
    if not available:
        return {
            "model": model_name,
            "customer_id": customer_id,
            "pod_id": pod_id,
            "available_type_count": len(valid_types),
            "elapsed_sec_total": np.nan,
            "elapsed_sec_per_type": np.nan,
            "status": "dependency_missing",
            "error": dependency_error,
        }

    ensure_repo_root_on_path()
    try:
        benchmark_callable = get_model_benchmark_callable(model_name)
    except Exception as exc:
        return {
            "model": model_name,
            "customer_id": customer_id,
            "pod_id": pod_id,
            "available_type_count": len(valid_types),
            "elapsed_sec_total": np.nan,
            "elapsed_sec_per_type": np.nan,
            "status": "import_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    config = build_runtime_forecast_config(model_name, start_date, end_date)

    started = perf_counter()
    try:
        with suppress_non_error_logs():
            if model_name in {"arima", "sarima"}:
                from hyperparameters import get_model_hyperparameters

                order, seasonal_order = get_model_hyperparameters(model_name, config.model_parameters)
                if model_name == "arima":
                    seasonal_order = None
                model_stub = types.SimpleNamespace(config=types.SimpleNamespace(log=False))
                benchmark_callable(
                    pod_df.copy(),
                    order,
                    customer_id,
                    pod_id,
                    list(valid_types),
                    config,
                    model_stub,
                    seasonal_order=seasonal_order,
                )
            else:
                benchmark_callable(
                    pod_df.copy(),
                    customer_id,
                    pod_id,
                    list(valid_types),
                    config,
                )
        elapsed = perf_counter() - started
        per_type = elapsed / max(len(valid_types), 1)
        return {
            "model": model_name,
            "customer_id": customer_id,
            "pod_id": pod_id,
            "available_type_count": len(valid_types),
            "elapsed_sec_total": elapsed,
            "elapsed_sec_per_type": per_type,
            "status": "completed",
            "error": "",
        }
    except Exception as exc:
        elapsed = perf_counter() - started
        return {
            "model": model_name,
            "customer_id": customer_id,
            "pod_id": pod_id,
            "available_type_count": len(valid_types),
            "elapsed_sec_total": elapsed,
            "elapsed_sec_per_type": np.nan,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_successful_benchmark_summary(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    successful = benchmark_df[
        (benchmark_df["status"] == "completed") &
        benchmark_df["elapsed_sec_total"].notna()
    ].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "history_bucket",
                "requested_type_count",
                "sample_count",
                "p50_sec",
                "p90_sec",
            ]
        )

    summary = (
        successful.groupby(["model", "history_bucket", "requested_type_count"], dropna=False)
        .agg(
            sample_count=("elapsed_sec_total", "size"),
            p50_sec=("elapsed_sec_total", lambda s: float(np.quantile(s, 0.5))),
            p90_sec=("elapsed_sec_total", lambda s: float(np.quantile(s, 0.9))),
        )
        .reset_index()
    )
    return summary
