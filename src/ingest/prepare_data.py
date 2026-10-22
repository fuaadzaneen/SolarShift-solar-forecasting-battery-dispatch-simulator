"""
Stage 1 Master Dataset Preparation Script.
Merges historical weather, physical 5kW PV generation, and Kerala synthetic load.
Caches processed dataset at data/processed/kochi_hourly_dataset.csv.
"""

import os
import pandas as pd
from src.ingest.weather import fetch_kochi_weather, RAW_DATA_PATH
from src.models.pv_system import PVSystemModel, add_pv_generation_to_df
from src.models.load_profile import KeralaLoadProfileGenerator

PROCESSED_DATA_PATH = os.path.join("data", "processed", "kochi_hourly_dataset.csv")


def prepare_stage1_dataset() -> pd.DataFrame:
    """
    Executes the entire Stage 1 data pipeline:
    1. Fetch / verify 3-year hourly Kochi weather (Open-Meteo).
    2. Simulate temperature-derated 5 kWp PV output.
    3. Generate calibrated Kerala synthetic load profile.
    4. Save clean processed dataset.
    """
    print("=" * 65)
    print("SOLARSHIFT: EXECUTING STAGE 1 DATA & PHYSICAL MODELING PIPELINE")
    print("=" * 65)

    # 1. Weather
    df = fetch_kochi_weather()

    # 2. PV Modeling
    print("[PHYSICS] Simulating 5 kWp rooftop PV with thermal derating...")
    pv_model = PVSystemModel(
        p_rated_kw=5.0,
        noct_celsius=45.0,
        temp_coeff_gamma=0.004,
        performance_ratio=0.82,
        inverter_efficiency=0.96
    )
    df = add_pv_generation_to_df(df, pv_model=pv_model)

    # 3. Load Profile
    print("[LOAD] Synthesizing Kerala diurnal/seasonal load profile...")
    load_gen = KeralaLoadProfileGenerator(
        base_load_kw=0.80,
        peak_load_kw=3.60,
        random_seed=42
    )
    loads = load_gen.generate(pd.DatetimeIndex(df["timestamp"]))
    df["load_kw"] = loads.values

    # Net load without storage: Net = Load - PV
    df["net_load_no_battery_kw"] = np.round(df["load_kw"] - df["pv_kw"], 3)

    # Save processed dataset
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[STAGE 1 COMPLETE] Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Total Rows: {len(df)} | Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    import numpy as np
    prepare_stage1_dataset()
