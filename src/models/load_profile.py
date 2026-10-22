"""
Modular Kerala Synthetic Load Profile Generator.
Constructs an explainable, hourly electrical demand curve representing
a residential / light-commercial consumer in Kochi, Kerala.

Key Grid Characteristics Modelled:
----------------------------------
1. Pronounced Evening Peak (18:00 - 22:00 IST):
   - Driven by simultaneous domestic lighting, cooking, TV/entertainment, and air conditioning.
   - Typically 2.8 - 4.2 kW.
2. Daytime Profile:
   - Basal refrigeration, fans, home office/light commercial operations (1.0 - 1.8 kW).
3. Weekday vs. Weekend Split:
   - Weekends exhibit shifted daytime occupancy and higher midday domestic usage.
4. Kerala Seasonal Variations:
   - Pre-Monsoon / Summer (Mar - May): Peak AC cooling demand driven by intense tropical heat and humidity.
   - Southwest Monsoon (Jun - Aug): Lower ambient temperature reduces AC demand, but high humidity maintains fan usage; early evening lighting loads increase due to gloomy skies.
   - Post-Monsoon / Winter (Sep - Feb): Moderate cooling and typical residential cycles.

Drop-in Design:
---------------
A consumer can drop in real smart meter CSV data via `load_real_meter_profile()`.
"""

import os
import pandas as pd
import numpy as np


class KeralaLoadProfileGenerator:
    """
    Synthetic load generator parameterized for tropical coastal Kerala.
    """
    def __init__(
        self,
        base_load_kw: float = 0.80,
        peak_load_kw: float = 3.60,
        random_seed: int = 42
    ):
        self.base_load_kw = base_load_kw
        self.peak_load_kw = peak_load_kw
        self.rng = np.random.default_rng(random_seed)

        # Baseline normalized 24-hour diurnal profile (sums or scales around 0 to 1)
        # Hour 0 to 23
        self.hourly_shape = np.array([
            0.25, 0.22, 0.20, 0.20, 0.22, 0.28,  # 00:00 - 05:00: Sleep baseline
            0.45, 0.65, 0.60, 0.50, 0.48, 0.52,  # 06:00 - 11:00: Morning prep & breakfast
            0.55, 0.50, 0.48, 0.45, 0.50, 0.65,  # 12:00 - 17:00: Afternoon baseline / work
            0.90, 1.00, 0.95, 0.85, 0.60, 0.35   # 18:00 - 23:00: Pronounced Evening Peak
        ])

    def get_seasonal_multiplier(self, month: int) -> float:
        """
        Seasonal scale factor reflecting Kerala's climate:
        - March to May (Pre-Monsoon Summer): Highest cooling load (1.20 - 1.30)
        - June to August (SW Monsoon): Rain cooling, less AC (0.85 - 0.95)
        - Sept to Nov (Post-Monsoon): Moderate (0.95 - 1.05)
        - Dec to Feb (Dry Winter): Pleasant (0.90 - 1.00)
        """
        seasonal_factors = {
            1: 0.92, 2: 0.98,
            3: 1.15, 4: 1.28, 5: 1.25,  # Summer Peak
            6: 0.88, 7: 0.85, 8: 0.90,  # Monsoon dip in cooling
            9: 0.95, 10: 1.00, 11: 0.96,
            12: 0.90
        }
        return seasonal_factors.get(month, 1.0)

    def generate(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """
        Generate synthetic hourly load series matching timestamps.
        
        Parameters:
        -----------
        timestamps : pd.DatetimeIndex
        
        Returns:
        --------
        pd.Series : Hourly electrical demand in kW
        """
        n = len(timestamps)
        load_kw = np.zeros(n)

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            month = ts.month
            day_of_week = ts.dayofweek  # 5 = Sat, 6 = Sun

            # Base normalized load for the hour
            norm_val = self.hourly_shape[hour]

            # Weekend shift: 15% higher daytime load (11:00 - 16:00)
            if day_of_week in [5, 6] and 10 <= hour <= 16:
                norm_val *= 1.20

            # Scale between base_load and peak_load
            unscaled = self.base_load_kw + (self.peak_load_kw - self.base_load_kw) * norm_val

            # Apply seasonal temperature/cooling multiplier
            seasonal_mult = self.get_seasonal_multiplier(month)
            load_val = unscaled * seasonal_mult

            # Add stochastic consumer noise (+/- 8% Gaussian)
            noise = self.rng.normal(0, 0.05 * load_val)
            load_kw[i] = max(0.30, load_val + noise)

        return pd.Series(np.round(load_kw, 3), index=timestamps, name="load_kw")


def load_real_meter_profile(
    csv_path: str,
    target_timestamps: pd.DatetimeIndex,
    timestamp_col: str = "timestamp",
    load_col: str = "load_kw"
) -> pd.Series:
    """
    Drop-in module to load real field-metered demand data.
    Aligns and reindexes to target timestamps via interpolation or forward fill.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Meter file not found at: {csv_path}")

    df_meter = pd.read_csv(csv_path)
    df_meter[timestamp_col] = pd.to_datetime(df_meter[timestamp_col])
    df_meter = df_meter.set_index(timestamp_col).sort_index()

    # Reindex to target simulation timestamps
    reindexed = df_meter[load_col].reindex(target_timestamps).interpolate(method="time").bfill().ffill()
    return reindexed


if __name__ == "__main__":
    from src.ingest.weather import RAW_DATA_PATH
    if os.path.exists(RAW_DATA_PATH):
        df_weather = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
        gen = KeralaLoadProfileGenerator()
        loads = gen.generate(pd.DatetimeIndex(df_weather["timestamp"]))
        df_weather["load_kw"] = loads.values
        
        print("Generated Synthetic Load Profile:")
        print(df_weather[["timestamp", "load_kw"]].head(12))
        print("\nEvening Peak Hours (18:00 - 22:00) Sample:")
        print(df_weather[["timestamp", "load_kw"]].iloc[18:23])
        print("\nSummary Statistics of Load Profile (kW):")
        print(df_weather["load_kw"].describe())
