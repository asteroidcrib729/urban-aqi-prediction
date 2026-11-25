"""Automated verification tests for the Pakistan Air Quality & Winter Smog project."""

from __future__ import annotations

import ast
import json
import math
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = ROOT / "urban-aqi-prediction.ipynb"
DATASET_PATH = ROOT / "dataset" / "pakistan_air_quality_daily.csv"
MODEL_PATH = ROOT / "model" / "pakistan_aqi_smog_models.pkl"
MAP_PATH = ROOT / "visualizations" / "pakistan_air_quality_map.html"


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the notebook's leak-free city-grouped lag and rolling features."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], format="%Y-%m-%d")
    city_map = {c: i for i, c in enumerate(sorted(work["city"].unique()))}
    work["city_id"] = work["city"].map(city_map)
    work["month"] = work["date"].dt.month
    work["day_of_week"] = work["date"].dt.day_of_week
    work["day_of_year"] = work["date"].dt.day_of_year
    work["temp_range"] = (work["temp_max"] - work["temp_min"]).round(2)
    work["dew_point_spread"] = (work["temp_mean"] - work["dew_point_mean"]).round(2)
    work = work.sort_values(["city_id", "date"]).reset_index(drop=True)

    base_cols = [
        "city_id",
        "latitude",
        "longitude",
        "month",
        "day_of_week",
        "day_of_year",
        "is_winter_smog_season",
        "pm2_5_mean",
        "pm2_5_max",
        "pm10_mean",
        "pm10_max",
        "no2_mean",
        "so2_mean",
        "o3_8h_max",
        "co_8h_max",
        "dust_mean",
        "aod_mean",
        "temp_mean",
        "temp_min",
        "temp_range",
        "dew_point_spread",
        "humidity_mean",
        "wind_speed_mean",
        "pressure_mean",
        "precip_sum",
        "cloud_cover_mean",
    ]
    lag_source_cols = [
        "pm2_5_mean",
        "pm2_5_max",
        "pm10_mean",
        "no2_mean",
        "so2_mean",
        "o3_8h_max",
        "co_8h_max",
        "dust_mean",
        "aod_mean",
        "temp_mean",
        "humidity_mean",
        "wind_speed_mean",
        "pressure_mean",
        "dew_point_spread",
    ]
    rolling_cols = [
        "pm2_5_mean",
        "pm10_mean",
        "no2_mean",
        "so2_mean",
        "aod_mean",
        "temp_mean",
        "humidity_mean",
        "wind_speed_mean",
        "pressure_mean",
        "dew_point_spread",
    ]

    grouped = work.groupby("city_id", group_keys=False)
    day_diff_1 = grouped["date"].diff(1).dt.days
    day_diff_2 = grouped["date"].diff(2).dt.days
    day_diff_3 = grouped["date"].diff(3).dt.days

    lag1 = grouped[lag_source_cols].shift(1).add_suffix("_lag_1")
    lag2 = grouped[lag_source_cols].shift(2).add_suffix("_lag_2")
    lag3 = grouped[lag_source_cols].shift(3).add_suffix("_lag_3")

    roll3 = (
        grouped[rolling_cols]
        .rolling(3)
        .mean()
        .reset_index(level=0, drop=True)
        .add_suffix("_rolling_3")
    )
    roll7 = (
        grouped[rolling_cols]
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
        .add_suffix("_rolling_7")
    )

    result = pd.concat(
        [work[["date", "city", "province"] + base_cols], lag1, lag2, lag3, roll3, roll7],
        axis=1,
    )
    valid_history = (day_diff_1 == 1) & (day_diff_2 == 2) & (day_diff_3 == 3)
    result = result.loc[valid_history].copy()

    next_date = grouped["date"].shift(-1)
    result["target_pm25"] = grouped["pm2_5_mean"].shift(-1)
    result["target_smog_day"] = grouped["is_smog_day"].shift(-1)
    result["target_winter_season"] = grouped["is_winter_smog_season"].shift(-1)
    result = result.loc[(next_date - result["date"]).dt.days == 1].copy()
    result["target_smog_day"] = result["target_smog_day"].astype(int)
    result["target_winter_season"] = result["target_winter_season"].astype(int)
    return result.dropna().reset_index(drop=True)


class TestPakistanAqiProject(unittest.TestCase):
    """End-to-end validation suite for configs, notebook, data, models, and visualizations."""

    def test_project_config_files_exist(self) -> None:
        required_files = [
            ".python-version",
            "pyproject.toml",
            "requirements.txt",
            "pyrefly.toml",
            ".gitignore",
            ".gitattributes",
            "LICENSE",
            "README.md",
            "scripts/fetch_pakistan_data.py",
            ".vscode/settings.json",
            ".vscode/extensions.json",
        ]
        for rel_path in required_files:
            with self.subTest(file=rel_path):
                self.assertTrue((ROOT / rel_path).is_file(), f"Missing {rel_path}")

        self.assertEqual((ROOT / ".python-version").read_text(encoding="utf-8").strip(), "3.14.7")
        license_first_line = (ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(license_first_line, "# MIT License")

        vscode_settings = json.loads(
            (ROOT / ".vscode" / "settings.json").read_text(encoding="utf-8")
        )
        self.assertEqual(vscode_settings["python.languageServer"], "None")
        self.assertTrue(vscode_settings["python.pyrefly.syncNotebooks"])

    def test_notebook_structure_syntax_and_execution(self) -> None:
        nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertEqual(nb["metadata"]["language_info"]["version"], "3.14.7")

        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 7)

        full_source = []
        for idx, cell in enumerate(code_cells):
            src = "".join(cell["source"])
            full_source.append(src)
            ast.parse(src)
            self.assertIsNotNone(
                cell.get("execution_count"), f"Code cell {idx} has not been executed"
            )
            for out in cell.get("outputs", []):
                self.assertNotEqual(
                    out.get("output_type"),
                    "error",
                    f"Code cell {idx} contains error output: {out}",
                )
                if out.get("output_type") == "stream" and out.get("name") == "stderr":
                    stderr_text = "".join(out.get("text", []))
                    self.assertEqual(
                        stderr_text.strip(),
                        "",
                        f"Code cell {idx} produced stderr warning: {stderr_text}",
                    )

        combined = "\n".join(full_source)
        self.assertIn("Esri.WorldTopoMap", combined)
        self.assertIn("folium.Popup(", combined)
        self.assertIn("max_width=280", combined)

    def test_dataset_schema_and_invariants(self) -> None:
        df = pd.read_csv(DATASET_PATH)
        self.assertEqual(df.shape, (47190, 29))
        self.assertEqual(int(df.isna().sum().sum()), 0)
        self.assertEqual(df["city"].nunique(), 30)
        self.assertEqual(df["date"].min(), "2022-08-05")
        self.assertEqual(df["date"].max(), "2026-11-24")

    def test_model_artifact_and_holdout_metrics(self) -> None:
        artifact = joblib.load(MODEL_PATH)
        self.assertIn("pm25_model", artifact)
        self.assertIn("smog_classifier", artifact)
        self.assertIn("feature_columns", artifact)
        self.assertIn("city_map", artifact)
        self.assertEqual(artifact["smog_threshold_ugm3"], 55.5)

        feature_columns = artifact["feature_columns"]
        city_map = artifact["city_map"]
        self.assertEqual(len(feature_columns), 88)
        self.assertEqual(len(city_map), 30)

        raw_df = pd.read_csv(DATASET_PATH)
        feat_df = build_engineered_features(raw_df)
        feat_df["_rank_pct"] = feat_df.groupby("city_id")["date"].rank(
            method="first", pct=True
        )
        test_df = feat_df[feat_df["_rank_pct"] > 0.90].drop(columns=["_rank_pct"])
        self.assertEqual(len(test_df), 4710)

        preds = np.maximum(artifact["pm25_model"].predict(test_df[feature_columns]), 0)
        probs = artifact["smog_classifier"].predict_proba(test_df[feature_columns])[:, 1]
        self.assertFalse(np.isnan(preds).any())
        self.assertFalse(np.isnan(probs).any())

        y_true = test_df["target_pm25"]
        rmse = math.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        roc_auc = roc_auc_score(test_df["target_smog_day"], probs)

        self.assertAlmostEqual(rmse, 13.35, places=2)
        self.assertAlmostEqual(mae, 8.82, places=2)
        self.assertAlmostEqual(r2, 0.7723, places=4)
        self.assertAlmostEqual(roc_auc, 0.9372, places=4)

    def test_visualization_exports_and_map_provider(self) -> None:
        expected_pngs = [
            "eda_overview.png",
            "lahore_pacf.png",
            "model_evaluation.png",
            "winter_smog_city_ranking.png",
            "city_dashboard_overview.png",
        ]
        for filename in expected_pngs:
            img_path = ROOT / "visualizations" / filename
            with self.subTest(image=filename):
                self.assertTrue(img_path.is_file(), f"Missing visualizations/{filename}")
                raw = img_path.read_bytes()
                self.assertGreater(len(raw), 10000)
                self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")

        self.assertTrue(MAP_PATH.is_file(), "Missing visualizations/pakistan_air_quality_map.html")
        html = MAP_PATH.read_text(encoding="utf-8")
        self.assertGreater(len(html), 10000)
        self.assertIn("server.arcgisonline.com", html)
        self.assertNotIn("tile.openstreetmap.org", html)
        self.assertNotIn("basemaps.cartocdn.com", html)
        self.assertIn("Lahore", html)
        self.assertIn("Winter Smog Risk", html)


if __name__ == "__main__":
    unittest.main()
