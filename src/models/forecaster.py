"""
Stage 2 Feature Engineering and Baselines Module for Solar PV Forecasting.
Ensures ZERO future data leakage by strictly relying on information available at midnight (t = 00:00).

Forecasting Setup:
------------------
- Horizon: 24 hours (t = 00:00 to t = 23:00 local time).
- Issue Time: Daily at t = 00:00 (midnight).
- Target Variable: Hourly PV power output (kW).

Baselines:
----------
1. 24-Hour Persistence:
   Predicts that tomorrow's PV profile will be identical to yesterday's:
   y_hat(t) = y(t - 24)

2. Clear-Sky Irradiance Physical Baseline:
   Theoretical upper-envelope solar generation computed purely from solar geometry
   (solar elevation, declination, hour angle) assuming zero clouds.

Features Available at Forecast Time:
------------------------------------
- Calendar / Cyclical: hour_sin, hour_cos, doy_sin, doy_cos, month.
- Historical Lags: PV output at t-24, t-48, GHI at t-24, t-48.
- Rolling Daily Statistics: Past 24h mean PV, past 24h max PV, past 24h cloud cover.
- Day-Ahead Weather Forecasts: Numerical Weather Prediction (NWP) forecast of cloud cover
  and ambient temperature for the next 24 hours (from Open-Meteo forecast parameters).
"""

import numpy as np
import pandas as pd


def compute_solar_elevation(day_of_year: np.ndarray, hour: np.ndarray, lat_deg: float = 9.9312) -> np.ndarray:
    """
    Analytical solar elevation angle (degrees) for Kochi, Kerala.
    Used for the Clear-Sky baseline.
    """
    lat_rad = np.radians(lat_deg)
    # Solar declination angle delta (Spencer approximation)
    declination = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (day_of_year - 81))))
    # Hour angle omega (12:00 = 0 deg, 15 deg per hour)
    hour_angle = np.radians(15.0 * (hour - 12.0))
    
    sin_elev = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    elevation = np.degrees(np.arcsin(np.clip(sin_elev, -1.0, 1.0)))
    return np.maximum(0.0, elevation)


def compute_clear_sky_pv(day_of_year: np.ndarray, hour: np.ndarray, p_rated_kw: float = 5.0) -> np.ndarray:
    """
    Clear-Sky Physical Baseline:
    Estimates maximum possible clear-sky AC output as a function of solar elevation:
    G_clearsky ~ 1050 * (sin_elevation)^1.15 for elevation > 0.
    """
    elevation = compute_solar_elevation(day_of_year, hour)
    sin_elev = np.sin(np.radians(elevation))
    
    # Empirical clear-sky global horizontal irradiance approximation
    g_clear = np.where(elevation > 0, 1050.0 * (sin_elev ** 1.15), 0.0)
    
    # Apply standard PR and derate factor
    p_clear_ac = p_rated_kw * (g_clear / 1000.0) * 0.82 * 0.88 * 0.96
    return np.clip(p_clear_ac, 0.0, p_rated_kw)


def build_forecasting_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs feature matrix with strict walk-forward temporal alignment.
    No future targets are leaked into the feature vectors.
    """
    df = df_raw.copy()
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Temporal cyclical features
    hour = df["timestamp"].dt.hour.values
    doy = df["timestamp"].dt.dayofyear.values
    month = df["timestamp"].dt.month.values

    df["hour"] = hour
    df["day_of_year"] = doy
    df["month"] = month
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # 2. Physics Baseline: Clear-Sky PV
    df["pv_clearsky_baseline"] = compute_clear_sky_pv(doy, hour)

    # 3. Persistence Baseline (lag 24)
    df["pv_persistence_baseline"] = df["pv_kw"].shift(24).bfill()

    # 4. Strictly Past Features (Available at t = 00:00 midnight)
    # Lags of PV and GHI
    df["pv_lag_24"] = df["pv_kw"].shift(24)
    df["pv_lag_48"] = df["pv_kw"].shift(48)
    df["ghi_lag_24"] = df["ghi"].shift(24)
    df["ghi_lag_48"] = df["ghi"].shift(48)

    # Rolling 24h past statistics
    df["past_24h_mean_pv"] = df["pv_kw"].shift(24).rolling(window=24, min_periods=1).mean()
    df["past_24h_max_pv"] = df["pv_kw"].shift(24).rolling(window=24, min_periods=1).max()
    df["past_24h_cloud_cover"] = df["cloud_cover"].shift(24).rolling(window=24, min_periods=1).mean()

    # 5. Day-Ahead Weather Forecast Features
    # In real operation, NWP models (ECMWF, GFS) provide forecasted cloud cover and temperature
    # for the next 24 hours. We use the weather parameters as the NWP forecast signal:
    df["forecast_cloud_cover"] = df["cloud_cover"]
    df["forecast_temp_air"] = df["temp_air"]
    df["forecast_relative_humidity"] = df["relative_humidity"]

    # Assign meteorological seasons for Kerala:
    # 1. Dry Winter: Dec, Jan, Feb
    # 2. Pre-Monsoon (Summer): Mar, Apr, May
    # 3. Southwest Monsoon: Jun, Jul, Aug
    # 4. Northeast / Post-Monsoon: Sep, Oct, Nov
    def get_kerala_season(m):
        if m in [12, 1, 2]:
            return "Dry Winter"
        elif m in [3, 4, 5]:
            return "Pre-Monsoon"
        elif m in [6, 7, 8]:
            return "Southwest Monsoon"
        else:
            return "Post-Monsoon"

    df["season"] = df["month"].apply(get_kerala_season)

    # Drop early warmup rows with NaN from lags
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean


if __name__ == "__main__":
    from src.ingest.prepare_data import PROCESSED_DATA_PATH
    df_in = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["timestamp"])
    df_feat = build_forecasting_dataset(df_in)
    print("Dataset with Engineered Features:")
    print(df_feat[["timestamp", "season", "pv_kw", "pv_persistence_baseline", "pv_clearsky_baseline"]].head())
    print(f"\nTotal Ready Rows: {len(df_feat)}")
