"""Fetch daily CAMS air quality and ERA5 weather data for 30 major Pakistani cities."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

START_DATE = "2022-08-05"
END_DATE = "2026-11-24"
TIMEZONE = "Asia/Karachi"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://archive-api.open-meteo.com/v1/archive"

AQ_HOURLY_VARS = (
    "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,"
    "ozone,carbon_monoxide,dust,aerosol_optical_depth,us_aqi"
)
WX_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,"
    "precipitation,surface_pressure,cloud_cover,wind_speed_10m"
)

PAKISTAN_CITIES: list[tuple[str, str, float, float]] = [
    ("Abbottabad", "Khyber Pakhtunkhwa", 34.1688, 73.2215),
    ("Bahawalpur", "Punjab", 29.3544, 71.6911),
    ("Dera Ghazi Khan", "Punjab", 30.0489, 70.6455),
    ("Dera Ismail Khan", "Khyber Pakhtunkhwa", 31.8626, 70.9019),
    ("Faisalabad", "Punjab", 31.4504, 73.1350),
    ("Gilgit", "Gilgit-Baltistan", 35.9208, 74.3080),
    ("Gujranwala", "Punjab", 32.1877, 74.1945),
    ("Gujrat", "Punjab", 32.5731, 74.0750),
    ("Gwadar", "Balochistan", 25.1216, 62.3254),
    ("Hyderabad", "Sindh", 25.3960, 68.3578),
    ("Islamabad", "Islamabad Capital Territory", 33.6844, 73.0479),
    ("Jhang", "Punjab", 31.2781, 72.3317),
    ("Karachi", "Sindh", 24.8607, 67.0011),
    ("Kasur", "Punjab", 31.1179, 74.4408),
    ("Lahore", "Punjab", 31.5204, 74.3587),
    ("Larkana", "Sindh", 27.5570, 68.2028),
    ("Mardan", "Khyber Pakhtunkhwa", 34.1986, 72.0404),
    ("Mingora", "Khyber Pakhtunkhwa", 34.7717, 72.3602),
    ("Multan", "Punjab", 30.1575, 71.5249),
    ("Muzaffarabad", "Azad Kashmir", 34.3700, 73.4711),
    ("Nawabshah", "Sindh", 26.2442, 68.4100),
    ("Peshawar", "Khyber Pakhtunkhwa", 34.0151, 71.5249),
    ("Quetta", "Balochistan", 30.1798, 66.9750),
    ("Rahim Yar Khan", "Punjab", 28.4212, 70.2989),
    ("Rawalpindi", "Punjab", 33.5651, 73.0169),
    ("Sahiwal", "Punjab", 30.6682, 73.1114),
    ("Sargodha", "Punjab", 32.0740, 72.6861),
    ("Sheikhupura", "Punjab", 31.7167, 73.9850),
    ("Sialkot", "Punjab", 32.4945, 74.5229),
    ("Sukkur", "Sindh", 27.7052, 68.8574),
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "dataset" / "pakistan_air_quality_daily.csv"


def _get_json_with_retry(
    session: requests.Session, url: str, params: dict[str, str | float]
) -> dict | list[dict]:
    """Perform an HTTP GET with exponential backoff for Open-Meteo rate limits."""
    for attempt in range(6):
        resp = session.get(url, params=params, timeout=60)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait_s = min(60, 5 * (2**attempt))
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Failed after retries: {url}")


def _classify_pm25_category(val: float) -> str:
    """Classify daily mean PM2.5 (µg/m³) using US EPA / Pakistan AQI breakpoints."""
    if val <= 12.0:
        return "Good"
    if val <= 35.4:
        return "Moderate"
    if val <= 55.4:
        return "Unhealthy for Sensitive"
    if val <= 150.4:
        return "Unhealthy"
    if val <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


def _aggregate_city_daily(
    aq_hourly: dict, wx_hourly: dict, city: str, province: str, lat: float, lon: float
) -> pd.DataFrame:
    """Aggregate hourly CAMS air quality + ERA5 weather dicts into daily rows."""
    aq_df = pd.DataFrame(aq_hourly)
    wx_df = pd.DataFrame(wx_hourly)
    merged = aq_df.merge(wx_df, on="time", how="inner")
    merged["dt"] = pd.to_datetime(merged["time"], format="%Y-%m-%dT%H:%M")
    merged = merged.sort_values("dt").reset_index(drop=True)

    merged["o3_8h"] = merged["ozone"].rolling(8, min_periods=1).mean()
    merged["co_8h"] = merged["carbon_monoxide"].rolling(8, min_periods=1).mean()
    merged["date"] = merged["dt"].dt.strftime("%Y-%m-%d")

    daily = (
        merged.groupby("date", as_index=False)
        .agg(
            pm2_5_mean=("pm2_5", "mean"),
            pm2_5_max=("pm2_5", "max"),
            pm10_mean=("pm10", "mean"),
            pm10_max=("pm10", "max"),
            no2_mean=("nitrogen_dioxide", "mean"),
            so2_mean=("sulphur_dioxide", "mean"),
            o3_8h_max=("o3_8h", "max"),
            co_8h_max=("co_8h", "max"),
            dust_mean=("dust", "mean"),
            aod_mean=("aerosol_optical_depth", "mean"),
            us_aqi_max=("us_aqi", "max"),
            hours_available=("pm2_5", "count"),
            temp_mean=("temperature_2m", "mean"),
            temp_min=("temperature_2m", "min"),
            temp_max=("temperature_2m", "max"),
            dew_point_mean=("dew_point_2m", "mean"),
            humidity_mean=("relative_humidity_2m", "mean"),
            wind_speed_mean=("wind_speed_10m", "mean"),
            pressure_mean=("surface_pressure", "mean"),
            precip_sum=("precipitation", "sum"),
            cloud_cover_mean=("cloud_cover", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily.insert(1, "city", city)
    daily.insert(2, "province", province)
    daily.insert(3, "latitude", lat)
    daily.insert(4, "longitude", lon)

    daily["pm25_aqi_category"] = daily["pm2_5_mean"].map(_classify_pm25_category)
    months = pd.to_datetime(daily["date"], format="%Y-%m-%d").dt.month
    daily["is_winter_smog_season"] = months.isin([11, 12, 1, 2]).astype(int)
    daily["is_smog_day"] = (daily["pm2_5_mean"] >= 55.5).astype(int)

    float_cols = daily.select_dtypes(include=[np.floating]).columns
    daily[float_cols] = daily[float_cols].round(2)
    return daily


def fetch_batch_daily(
    session: requests.Session, batch: list[tuple[str, str, float, float]]
) -> list[pd.DataFrame]:
    """Fetch a multi-city batch from Open-Meteo AQ and ERA5 APIs."""
    lats = ",".join(str(item[2]) for item in batch)
    lons = ",".join(str(item[3]) for item in batch)

    aq_raw = _get_json_with_retry(
        session,
        AQ_URL,
        {
            "latitude": lats,
            "longitude": lons,
            "hourly": AQ_HOURLY_VARS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "timezone": TIMEZONE,
        },
    )
    wx_raw = _get_json_with_retry(
        session,
        WX_URL,
        {
            "latitude": lats,
            "longitude": lons,
            "hourly": WX_HOURLY_VARS,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "timezone": TIMEZONE,
        },
    )

    aq_list = aq_raw if isinstance(aq_raw, list) else [aq_raw]
    wx_list = wx_raw if isinstance(wx_raw, list) else [wx_raw]

    out: list[pd.DataFrame] = []
    for (city, province, lat, lon), aq_item, wx_item in zip(batch, aq_list, wx_list):
        city_df = _aggregate_city_daily(
            aq_item["hourly"], wx_item["hourly"], city, province, lat, lon
        )
        out.append(city_df)
    return out


def main() -> None:
    """Download all 30 cities in batches and write dataset/pakistan_air_quality_daily.csv."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    batch_size = 5

    with requests.Session() as session:
        for start in range(0, len(PAKISTAN_CITIES), batch_size):
            batch = PAKISTAN_CITIES[start : start + batch_size]
            batch_frames = fetch_batch_daily(session, batch)
            frames.extend(batch_frames)
            for city_df in batch_frames:
                city = city_df["city"].iloc[0]
                province = city_df["province"].iloc[0]
                print(
                    f"[{len(frames):02d}/{len(PAKISTAN_CITIES)}] {city} ({province}): "
                    f"{len(city_df)} days | mean PM2.5={city_df['pm2_5_mean'].mean():.2f}"
                )
            time.sleep(2.0)

    full_df = pd.concat(frames, ignore_index=True).sort_values(["city", "date"]).reset_index(drop=True)
    if int(full_df.isna().sum().sum()) != 0:
        raise ValueError(f"Unexpected nulls:\n{full_df.isna().sum()}")

    full_df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Saved {OUTPUT_PATH} -> {full_df.shape[0]:,} rows x {full_df.shape[1]} columns "
        f"({full_df['date'].min()} to {full_df['date'].max()})"
    )


if __name__ == "__main__":
    main()
