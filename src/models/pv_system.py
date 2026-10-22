"""
Physical PV Generation Simulation Module.
Calculates temperature-derated AC power output for a 5 kWp rooftop PV system in Kochi.

Physical Formulation:
---------------------
1. Cell Temperature (NOCT Model):
   T_cell = T_amb + ((NOCT - 20) / 800) * G
   Where:
     - T_amb = Ambient air temperature at 2m (deg C)
     - NOCT = Nominal Operating Cell Temperature (default: 45.0 deg C)
     - G = Global Horizontal Irradiance (W/m^2)

2. Temperature-Derated AC Power Output:
   P_pv = P_rated * (G / 1000) * PR * [1 - gamma * (T_cell - 25)]
   Where:
     - P_rated = Rated capacity (5.0 kW)
     - PR = Performance Ratio accounting for cabling, inverter, dust/soiling losses (0.82)
     - gamma = Temperature coefficient of power (-0.004 / deg C for crystalline silicon)
     - 25 = Standard Test Condition (STC) cell temperature (deg C)

NOTE:
-----
PV generation is simulated from physical meteorological data rather than field-metered telemetry.
"""

import pandas as pd
import numpy as np


class PVSystemModel:
    """
    Rooftop Solar PV Physical Model with NOCT cell temperature and thermal derating.
    """
    def __init__(
        self,
        p_rated_kw: float = 5.0,
        noct_celsius: float = 45.0,
        temp_coeff_gamma: float = 0.004,
        performance_ratio: float = 0.82,
        inverter_efficiency: float = 0.96
    ):
        self.p_rated_kw = p_rated_kw
        self.noct_celsius = noct_celsius
        self.temp_coeff_gamma = temp_coeff_gamma
        self.performance_ratio = performance_ratio
        self.inverter_efficiency = inverter_efficiency

    def calculate_cell_temperature(self, ghi: np.ndarray, temp_air: np.ndarray) -> np.ndarray:
        """
        NOCT cell temperature approximation:
        T_cell = T_amb + ((NOCT - 20) / 800) * GHI
        """
        ghi_clean = np.maximum(0.0, ghi)
        t_cell = temp_air + ((self.noct_celsius - 20.0) / 800.0) * ghi_clean
        return t_cell

    def simulate_generation(self, ghi: np.ndarray, temp_air: np.ndarray) -> dict:
        """
        Simulate hourly PV AC power output (kW).
        
        Returns:
        --------
        dict with:
          - 'pv_kw': Inverter AC power output (kW)
          - 't_cell': Estimated cell temperature (deg C)
          - 'temp_derate_factor': [1 - gamma * (T_cell - 25)]
        """
        ghi_clean = np.maximum(0.0, ghi)
        t_cell = self.calculate_cell_temperature(ghi_clean, temp_air)

        # Thermal loss factor: efficiency drops when T_cell > 25°C
        temp_derate_factor = 1.0 - self.temp_coeff_gamma * (t_cell - 25.0)
        temp_derate_factor = np.clip(temp_derate_factor, 0.70, 1.10)  # Physical sanity bounds

        # DC power before inverter
        p_dc_kw = self.p_rated_kw * (ghi_clean / 1000.0) * self.performance_ratio * temp_derate_factor
        
        # AC power after inverter efficiency and clipping at rated inverter capacity (5 kW)
        p_ac_kw = np.minimum(p_dc_kw * self.inverter_efficiency, self.p_rated_kw)
        p_ac_kw = np.maximum(0.0, p_ac_kw)

        return {
            "pv_kw": p_ac_kw,
            "t_cell": t_cell,
            "temp_derate_factor": temp_derate_factor
        }


def add_pv_generation_to_df(df: pd.DataFrame, pv_model: PVSystemModel = None) -> pd.DataFrame:
    """
    Convenience wrapper to append PV generation columns to a weather DataFrame.
    """
    if pv_model is None:
        pv_model = PVSystemModel()

    res = pv_model.simulate_generation(df["ghi"].values, df["temp_air"].values)
    df["t_cell"] = np.round(res["t_cell"], 2)
    df["temp_derate_factor"] = np.round(res["temp_derate_factor"], 4)
    df["pv_kw"] = np.round(res["pv_kw"], 3)
    return df


if __name__ == "__main__":
    from src.ingest.weather import RAW_DATA_PATH
    import os
    if os.path.exists(RAW_DATA_PATH):
        df_weather = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
        df_with_pv = add_pv_generation_to_df(df_weather)
        print("PV Simulation Sample:")
        print(df_with_pv[["timestamp", "ghi", "temp_air", "t_cell", "pv_kw"]].iloc[10:16])
        print("\nPV Output Summary (kW):")
        print(df_with_pv["pv_kw"].describe())
