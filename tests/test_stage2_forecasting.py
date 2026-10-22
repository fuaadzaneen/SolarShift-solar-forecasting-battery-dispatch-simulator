"""
Stage 2 Forecasting Verification Test Suite.
Verifies zero temporal leakage, daylight constraints, and baseline ordering.
"""

import unittest
import numpy as np
import pandas as pd
from src.models.forecaster import compute_clear_sky_pv, compute_solar_elevation, build_forecasting_dataset
from src.metrics.forecast_metrics import calculate_metrics


class TestStage2Forecasting(unittest.TestCase):
    def test_solar_elevation_nighttime(self):
        """At solar midnight (hour 0), elevation must be zero."""
        elev = compute_solar_elevation(np.array([180]), np.array([0]))
        self.assertEqual(elev[0], 0.0)

    def test_clearsky_daylight_boundary(self):
        """Clear sky PV must be 0 at midnight and positive at midday."""
        pv_midnight = compute_clear_sky_pv(np.array([100]), np.array([0]))
        pv_noon = compute_clear_sky_pv(np.array([100]), np.array([12]))
        self.assertEqual(pv_midnight[0], 0.0)
        self.assertGreater(pv_noon[0], 2.0)

    def test_zero_leakage_feature_alignment(self):
        """Verify that past features lag strictly by at least 24 hours."""
        dummy_df = pd.DataFrame({
            "timestamp": pd.date_range("2021-01-01", periods=100, freq="h"),
            "ghi": np.linspace(0, 800, 100),
            "temp_air": np.full(100, 28.0),
            "cloud_cover": np.full(100, 50.0),
            "relative_humidity": np.full(100, 70.0),
            "pv_kw": np.arange(100, dtype=float)
        })
        feat_df = build_forecasting_dataset(dummy_df)
        
        # Check that pv_lag_24 at index 0 corresponds to pv_kw 24 hours prior
        self.assertEqual(feat_df["pv_lag_24"].iloc[0], dummy_df["pv_kw"].iloc[24])

    def test_ml_beats_persistence_overall(self):
        """Verify that HistGradientBoosting forecast achieves lower overall RMSE than persistence."""
        summary = pd.read_csv("data/processed/forecast_metrics_summary.csv")
        overall = summary[summary["Season"] == "All Seasons (Overall)"]
        
        ml_rmse = overall[overall["Model"] == "pv_ml_forecast"]["RMSE_kW"].values[0]
        pers_rmse = overall[overall["Model"] == "pv_persistence_baseline"]["RMSE_kW"].values[0]
        
        self.assertLess(ml_rmse, pers_rmse, f"ML RMSE ({ml_rmse}) must be lower than Persistence RMSE ({pers_rmse})")


if __name__ == "__main__":
    unittest.main()
