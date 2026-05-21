import tempfile
import unittest
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.runtime_analysis.build_runtime_census import main as census_main
from scripts.runtime_analysis.estimate_runtime import main as estimate_main
from scripts.runtime_analysis.plot_runtime_scenarios import main as plot_main
from scripts.runtime_analysis.run_runtime_benchmarks import main as benchmark_main


def _plotting_available() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def _xgboost_available() -> bool:
    try:
        return importlib.util.find_spec("xgboost") is not None
    except (ImportError, ValueError):
        return False


class RuntimeAnalysisTests(unittest.TestCase):
    def _write_fixture_csv(self, path: Path) -> None:
        dates = pd.date_range("2023-05-01", periods=14, freq="MS")
        rows = []
        for idx, month in enumerate(dates):
            rows.append(
                {
                    "ReportingMonth": month.strftime("%Y-%m-%d"),
                    "CustomerID": "1001",
                    "PodID": "2001",
                    "TariffID": "LANDR123",
                    "OffPeakConsumption": 100 + idx,
                    "PeakConsumption": 50 + 2 * idx,
                    "StandardConsumption": 70 + idx,
                    "Block1Consumption": 0,
                    "Block2Consumption": 0,
                    "Block3Consumption": 0,
                    "Block4Consumption": 0,
                    "NonTOUConsumption": 0,
                }
            )
        for idx, month in enumerate(pd.date_range("2024-01-01", periods=6, freq="MS")):
            rows.append(
                {
                    "ReportingMonth": month.strftime("%Y-%m-%d"),
                    "CustomerID": "1002",
                    "PodID": "2002",
                    "TariffID": "LANDR123",
                    "OffPeakConsumption": 10 + idx,
                    "PeakConsumption": 0,
                    "StandardConsumption": 0,
                    "Block1Consumption": 0,
                    "Block2Consumption": 0,
                    "Block3Consumption": 0,
                    "Block4Consumption": 0,
                    "NonTOUConsumption": 0,
                }
            )
        for idx, month in enumerate(pd.date_range("2023-05-01", periods=14, freq="MS")):
            rows.append(
                {
                    "ReportingMonth": month.strftime("%Y-%m-%d"),
                    "CustomerID": "1003",
                    "PodID": "2003",
                    "TariffID": "LANDR123",
                    "OffPeakConsumption": 0,
                    "PeakConsumption": 5,
                    "StandardConsumption": 0,
                    "Block1Consumption": 0,
                    "Block2Consumption": 0,
                    "Block3Consumption": 0,
                    "Block4Consumption": 0,
                    "NonTOUConsumption": 0,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_build_runtime_census(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "fixture.csv"
            out_dir = tmp_path / "out"
            self._write_fixture_csv(csv_path)

            exit_code = census_main(["--csv-path", str(csv_path), "--out-dir", str(out_dir)])
            self.assertEqual(exit_code, 0)

            census_df = pd.read_csv(out_dir / "runtime_census.csv")
            self.assertEqual(set(["history_bucket", "valid_type_count", "valid_type_bucket", "pod_bucket"]).issubset(census_df.columns), True)

            pair_one = census_df[census_df["pod_id"] == 2001].iloc[0]
            self.assertEqual(pair_one["valid_type_count"], 3)
            self.assertEqual(str(pair_one["valid_type_bucket"]), "3")

            pair_two = census_df[census_df["pod_id"] == 2002].iloc[0]
            self.assertEqual(pair_two["history_bucket"], "<12")

            pair_three = census_df[census_df["pod_id"] == 2003].iloc[0]
            self.assertEqual(pair_three["valid_type_count"], 0)

    def test_benchmark_harness(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "fixture.csv"
            out_dir = tmp_path / "out"
            self._write_fixture_csv(csv_path)

            models = ["arima", "sarima", "randomforest", "xgboost"]
            exit_code = benchmark_main(
                [
                    "--csv-path",
                    str(csv_path),
                    "--models",
                    *models,
                    "--samples-per-bucket",
                    "1",
                    "--random-seed",
                    "1",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            benchmark_df = pd.read_csv(out_dir / "runtime_benchmarks.csv")
            self.assertFalse(benchmark_df.empty)

            available_models = {"arima", "sarima", "randomforest"}
            completed = benchmark_df[benchmark_df["model"].isin(available_models)]
            self.assertTrue((completed["status"] == "completed").any())
            self.assertTrue((completed.loc[completed["status"] == "completed", "elapsed_sec_total"] > 0).all())

            xgb_rows = benchmark_df[benchmark_df["model"] == "xgboost"]
            if _xgboost_available():
                self.assertTrue((xgb_rows["status"] == "completed").any())
            else:
                self.assertTrue((xgb_rows["status"] == "dependency_missing").all())

    def test_estimator_outputs_synthetic_8_type_projection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            census_path = tmp_path / "runtime_census.csv"
            benchmarks_path = tmp_path / "runtime_benchmarks.csv"
            out_dir = tmp_path / "out"

            census_df = pd.DataFrame(
                [
                    {
                        "customer_id": "1001",
                        "pod_id": "2001",
                        "n_rows": 31,
                        "first_month": "2022-01-01",
                        "last_month": "2024-07-01",
                        "history_bucket": "24-35",
                        "active_type_count": 3,
                        "valid_type_count": 3,
                        "valid_types_list": "OffPeakConsumption|PeakConsumption|StandardConsumption",
                        "pods_per_customer": 1,
                        "pod_bucket": "1",
                        "valid_type_bucket": "3",
                        "forecast_start": "2024-08-01",
                        "forecast_end": "2025-07-01",
                    }
                ]
            )
            census_df.to_csv(census_path, index=False)

            benchmark_rows = []
            for model_name in ("arima", "sarima", "randomforest"):
                for type_count, elapsed in ((1, 1.0), (2, 2.1), (3, 3.2), (4, 4.4)):
                    benchmark_rows.append(
                        {
                            "model": model_name,
                            "customer_id": "1001",
                            "pod_id": "2001",
                            "history_bucket": "24-35",
                            "valid_type_bucket": "3",
                            "requested_type_count": type_count,
                            "available_type_count": 3,
                            "valid_types_used": "OffPeakConsumption",
                            "elapsed_sec_total": elapsed,
                            "elapsed_sec_per_type": elapsed / type_count,
                            "status": "completed",
                            "error": "",
                        }
                    )
            pd.DataFrame(benchmark_rows).to_csv(benchmarks_path, index=False)

            exit_code = estimate_main(
                [
                    "--census",
                    str(census_path),
                    "--benchmarks",
                    str(benchmarks_path),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            estimate_df = pd.read_csv(out_dir / "runtime_estimates.csv")
            curve_rows = estimate_df[
                (estimate_df["row_type"] == "pod_curve") &
                (estimate_df["model"] == "arima") &
                (estimate_df["requested_type_count"] == 8) &
                (estimate_df["pod_count"] == 1)
            ]
            self.assertFalse(curve_rows.empty)
            self.assertTrue(np.isfinite(curve_rows["projected_sec"]).all())

            one_pod = estimate_df[
                (estimate_df["row_type"] == "scenario") &
                (estimate_df["scenario_name"] == "1pod_3type_common") &
                (estimate_df["model"] == "arima") &
                (estimate_df["estimate_kind"] == "p50")
            ]["projected_sec"].iloc[0]
            five_pod = estimate_df[
                (estimate_df["row_type"] == "scenario") &
                (estimate_df["scenario_name"] == "5pod_3type_common") &
                (estimate_df["model"] == "arima") &
                (estimate_df["estimate_kind"] == "p50")
            ]["projected_sec"].iloc[0]
            self.assertAlmostEqual(five_pod, one_pod * 5, places=6)

    @unittest.skipUnless(_plotting_available(), "matplotlib is not installed")
    def test_plot_script_outputs_pngs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benchmark_path = tmp_path / "runtime_benchmarks.csv"
            estimate_path = tmp_path / "runtime_estimates.csv"
            census_path = tmp_path / "runtime_census.csv"
            out_dir = tmp_path / "out"

            pd.DataFrame(
                [
                    {
                        "customer_id": "1001",
                        "pod_id": "2001",
                        "n_rows": 31,
                        "first_month": "2022-01-01",
                        "last_month": "2024-07-01",
                        "history_bucket": "24-35",
                        "active_type_count": 3,
                        "valid_type_count": 3,
                        "valid_types_list": "OffPeakConsumption|PeakConsumption|StandardConsumption",
                        "pods_per_customer": 1,
                        "pod_bucket": "1",
                        "valid_type_bucket": "3",
                        "forecast_start": "2024-08-01",
                        "forecast_end": "2025-07-01",
                    }
                ]
            ).to_csv(census_path, index=False)

            pd.DataFrame(
                [
                    {
                        "model": "arima",
                        "customer_id": "1001",
                        "pod_id": "2001",
                        "history_bucket": "24-35",
                        "valid_type_bucket": "3",
                        "requested_type_count": 3,
                        "available_type_count": 3,
                        "valid_types_used": "OffPeakConsumption|PeakConsumption|StandardConsumption",
                        "elapsed_sec_total": 3.0,
                        "elapsed_sec_per_type": 1.0,
                        "status": "completed",
                        "error": "",
                    }
                ]
            ).to_csv(benchmark_path, index=False)

            pd.DataFrame(
                [
                    {
                        "row_type": "scenario",
                        "scenario_name": "1pod_3type_common",
                        "model": "arima",
                        "history_bucket": "24-35",
                        "requested_type_count": 3,
                        "pod_count": 1,
                        "estimate_kind": "p50",
                        "projected_sec": 3.0,
                        "source": "empirical_exact",
                    },
                    {
                        "row_type": "pod_curve",
                        "scenario_name": "curve_3type",
                        "model": "arima",
                        "history_bucket": "24-35",
                        "requested_type_count": 3,
                        "pod_count": 1,
                        "estimate_kind": "p50",
                        "projected_sec": 3.0,
                        "source": "empirical_exact",
                    },
                    {
                        "row_type": "pod_curve",
                        "scenario_name": "curve_3type",
                        "model": "arima",
                        "history_bucket": "24-35",
                        "requested_type_count": 3,
                        "pod_count": 2,
                        "estimate_kind": "p50",
                        "projected_sec": 6.0,
                        "source": "empirical_exact",
                    },
                ]
            ).to_csv(estimate_path, index=False)

            exit_code = plot_main(
                [
                    "--benchmarks",
                    str(benchmark_path),
                    "--estimates",
                    str(estimate_path),
                    "--out-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "runtime_bar_by_scenario.png").exists())
            self.assertTrue((out_dir / "runtime_distribution_by_model.png").exists())
            self.assertTrue((out_dir / "runtime_line_pods_vs_runtime.png").exists())
