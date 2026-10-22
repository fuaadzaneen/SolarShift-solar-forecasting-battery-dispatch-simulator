"""
Stage 1 Physical Model Verification Test Suite.
Verifies NOCT cell temperature, thermal derating bounds, and load profile integrity.
"""

import unittest
import numpy as np
import pandas as pd
from src.models.pv_system import PVSystemModel
from src.models.load_profile import KeralaLoadProfileGenerator


class TestStage1Physics(unittest.TestCase):
    def setUp(self):
        self.pv_model = PVSystemModel(
            p_rated_kw=5.0,
            noct_celsius=45.0,
            temp_coeff_gamma=0.004,
            performance_ratio=0.82,
            inverter_efficiency=0.96
        )
        self.load_gen = KeralaLoadProfileGenerator()

    def test_zero_irradiance_produces_zero_pv(self):
        """At nighttime (GHI = 0), PV generation must be exactly 0.0 kW."""
        ghi = np.array([0.0, 0.0, 0.0])
        temp_air = np.array([24.0, 25.0, 26.0])
        res = self.pv_model.simulate_generation(ghi, temp_air)
        np.testing.assert_array_equal(res["pv_kw"], [0.0, 0.0, 0.0])

    def test_thermal_derating_under_hot_conditions(self):
        """Hot cell temperature (> 25 C) must reduce efficiency derate factor below 1.0."""
        ghi = np.array([1000.0])
        temp_air = np.array([32.0])
        res = self.pv_model.simulate_generation(ghi, temp_air)
        
        # Expected T_cell = 32 + ((45 - 20)/800) * 1000 = 32 + 31.25 = 63.25 C
        self.assertAlmostEqual(res["t_cell"][0], 63.25, delta=0.1)
        # Expected derate factor = 1 - 0.004 * (63.25 - 25) = 1 - 0.153 = 0.847
        self.assertLess(res["temp_derate_factor"][0], 1.0)
        self.assertGreater(res["temp_derate_factor"][0], 0.70)

    def test_load_profile_evening_peak(self):
        """Evening load (19:00 - 21:00) should be substantially higher than 03:00 night sleep load."""
        dates = pd.date_range("2022-04-10 00:00:00", "2022-04-10 23:00:00", freq="h")
        loads = self.load_gen.generate(dates)
        
        sleep_load = loads[dates.hour == 3].values[0]
        evening_peak = loads[dates.hour == 20].values[0]
        
        self.assertGreater(evening_peak, sleep_load * 1.8, "Evening peak should be >= 1.8x sleep load in Kerala.")


if __name__ == "__main__":
    unittest.main()
