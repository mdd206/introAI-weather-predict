from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "hanoi_weather_2015_2025.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "hanoi_weather_processed.csv"


def preprocess_data():
    weather_data = pd.read_csv(INPUT_FILE)
    weather_data = weather_data.drop(columns=["rain_sum"])

    weather_data["date"] = pd.to_datetime(weather_data["date"])
    weather_data = weather_data.sort_values("date").reset_index(drop=True)

    temperature = weather_data["temperature_2m_mean"]
    humidity = weather_data["relative_humidity_2m_mean"]
    precipitation = weather_data["precipitation_sum"]
    pressure = weather_data["pressure_msl_mean"]

    weather_data["temp_range"] = (
        weather_data["temperature_2m_max"]
        - weather_data["temperature_2m_min"]
    )

    weather_data["temperature_lag1"] = temperature.shift(1)
    weather_data["temperature_lag2"] = temperature.shift(2)
    weather_data["temperature_lag3"] = temperature.shift(3)
    weather_data["temperature_rolling_3days"] = temperature.rolling(3).mean()
    weather_data["temperature_rolling_7days"] = temperature.rolling(7).mean()

    weather_data["humidity_lag1"] = humidity.shift(1)
    weather_data["humidity_rolling_3days"] = humidity.rolling(3).mean()

    weather_data["rain_lag1"] = precipitation.shift(1)
    weather_data["rain_lag2"] = precipitation.shift(2)
    weather_data["rain_rolling_3days"] = precipitation.rolling(3).mean()
    weather_data["rain_rolling_7days"] = precipitation.rolling(7).mean()

    weather_data["pressure_lag1"] = pressure.shift(1)
    weather_data["pressure_change"] = pressure - weather_data["pressure_lag1"]

    weather_data["month"] = weather_data["date"].dt.month
    weather_data["day_of_year"] = weather_data["date"].dt.dayofyear

    weather_data["target_temp"] = temperature.shift(-1)
    weather_data["target_rain"] = (precipitation.shift(-1) > 1.0).astype(int)

    weather_data = weather_data.dropna().reset_index(drop=True)

    return weather_data


def main():
    weather_data = preprocess_data()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    weather_data.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(weather_data)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
