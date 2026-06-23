from pathlib import Path

import pandas as pd
import requests


API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    "latitude": 21.0245,
    "longitude": 105.8412,
    "start_date": "2015-01-01",
    "end_date": "2025-12-31",
    "daily": ",".join(
        [
            "temperature_2m_mean",  # Nhiet do trung binh o do cao 2 m
            "temperature_2m_max",  # Nhiet do cao nhat o do cao 2 m
            "temperature_2m_min",  # Nhiet do thap nhat o do cao 2 m
            "relative_humidity_2m_mean",  # Do am tuong doi trung binh
            "pressure_msl_mean",  # Ap suat muc nuoc bien trung binh
            "cloud_cover_mean",  # Do che phu may trung binh
            "wind_speed_10m_mean",  # Toc do gio trung binh o do cao 10 m
            "wind_speed_10m_max",  # Toc do gio cao nhat o do cao 10 m
            "precipitation_sum",  # Tong luong giang thuy
            "rain_sum",  # Tong luong mua
        ]
    ),
    "timezone": "Asia/Bangkok",
}

OUTPUT_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "hanoi_weather_2015_2025.csv"
)


def main():
    response = requests.get(API_URL, params=PARAMS, timeout=120)
    response.raise_for_status()

    weather_data = pd.DataFrame(response.json()["daily"])
    weather_data = weather_data.rename(columns={"time": "date"})
    weather_data.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(weather_data)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
