"""
Weather data ingestion module for Kochi, Kerala.
Pulls historical meteorological data (GHI, 2m Temperature, Cloud Cover)
from the Open-Meteo Historical Weather API and caches locally as CSV.
"""

import os
import requests
import pandas as pd
from datetime import datetime

# Geographic coordinates for Kochi, Kerala
KOCHI_LATITUDE = 9.9312
KOCHI_LONGITUDE = 76.2673
TIMEZONE = "Asia/Kolkata"

DEFAULT_START_DATE = "2021-01-01"
DEFAULT_END_DATE = "2023-12-31"
RAW_DATA_PATH = os.path.join("data", "raw", "kochi_weather_hourly.csv")


def fetch_kochi_weather(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    output_path: str = RAW_DATA_PATH,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo Archive API.
    
    Parameters:
    -----------
    start_date : str (YYYY-MM-DD)
    end_date : str (YYYY-MM-DD)
    output_path : str
    force_download : bool
    
    Returns:
    --------
    pd.DataFrame with columns:
      - timestamp (tz-aware Asia/Kolkata)
      - ghi (W/m^2) [shortwave_radiation]
      - direct_radiation (W/m^2)
      - diffuse_radiation (W/m^2)
      - temp_air (deg C) [temperature_2m]
      - cloud_cover (%) [cloud_cover]
      - relative_humidity (%) [relative_humidity_2m]
      - wind_speed (m/s) [wind_speed_10m]
    """
    if os.path.exists(output_path) and not force_download:
        print(f"[CACHE] Loading cached weather data from: {output_path}")
        df = pd.read_csv(output_path, parse_dates=["timestamp"])
        return df

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": KOCHI_LATITUDE,
        "longitude": KOCHI_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "temperature_2m",
            "cloud_cover",
            "relative_humidity_2m",
            "wind_speed_10m"
        ],
        "timezone": TIMEZONE
    }

    print(f"[API] Fetching weather from Open-Meteo for Kochi ({start_date} to {end_date})...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    hourly = payload.get("hourly", {})
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "ghi": hourly.get("shortwave_radiation", []),
        "direct_radiation": hourly.get("direct_radiation", []),
        "diffuse_radiation": hourly.get("diffuse_radiation", []),
        "temp_air": hourly.get("temperature_2m", []),
        "cloud_cover": hourly.get("cloud_cover", []),
        "relative_humidity": hourly.get("relative_humidity_2m", []),
        "wind_speed": hourly.get("wind_speed_10m", [])
    })

    # Ensure no negative irradiance values from sensor/model noise
    df["ghi"] = df["ghi"].clip(lower=0.0)
    df["direct_radiation"] = df["direct_radiation"].clip(lower=0.0)
    df["diffuse_radiation"] = df["diffuse_radiation"].clip(lower=0.0)
    df["cloud_cover"] = df["cloud_cover"].clip(lower=0.0, upper=100.0)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Saved {len(df)} hourly weather records to {output_path}")
    return df


if __name__ == "__main__":
    df = fetch_kochi_weather()
    print(df.head())
    print("\nSummary Statistics:")
    print(df[["ghi", "temp_air", "cloud_cover"]].describe())
