from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "hanoi_weather_processed.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "split"


def split_data():
    weather_data = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    train_data = weather_data[weather_data["date"] < "2022-01-01"]
    validation_data = weather_data[
        (weather_data["date"] >= "2022-01-01")
        & (weather_data["date"] < "2024-01-01")
    ]
    test_data = weather_data[weather_data["date"] >= "2024-01-01"]

    return train_data, validation_data, test_data


def main():
    train_data, validation_data, test_data = split_data()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_data.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print(f"Train: {len(train_data)} rows")
    print(f"Validation: {len(validation_data)} rows")
    print(f"Test: {len(test_data)} rows")


if __name__ == "__main__":
    main()
