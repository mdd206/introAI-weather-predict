from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_DIR / "data" / "split"
RESULTS_DIR = PROJECT_DIR / "results" / "regression" / "temperature" / "lasso_regression"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

TARGET_COLUMN = "target_temp"
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
EPOCHS = 5000


def add_cyclical_features(weather_data):
    weather_data = weather_data.copy()

    weather_data["month_sin"] = np.sin(2 * np.pi * weather_data["month"] / 12)
    weather_data["month_cos"] = np.cos(2 * np.pi * weather_data["month"] / 12)
    weather_data["day_sin"] = np.sin(2 * np.pi * weather_data["day_of_year"] / 365)
    weather_data["day_cos"] = np.cos(2 * np.pi * weather_data["day_of_year"] / 365)

    weather_data = weather_data.drop(columns=["month", "day_of_year"])

    return weather_data


def split_features_target(weather_data):
    ignore_columns = ["date", "target_temp", "target_rain"]
    feature_columns = [
        column for column in weather_data.columns if column not in ignore_columns
    ]

    features = weather_data[feature_columns]
    target = weather_data[TARGET_COLUMN]

    return features, target, feature_columns


def fit_scaler(features):
    mean = features.mean()
    std = features.std(ddof=0).replace(0, 1)

    return mean, std


def transform_features(features, mean, std):
    return ((features - mean) / std).to_numpy(dtype=np.float32)


def to_tensor(features, target, device):
    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(
        target.to_numpy(dtype=np.float32).reshape(-1, 1),
        dtype=torch.float32,
        device=device,
    )

    return x_tensor, y_tensor


def get_learning_rate(x_tensor):
    spectral_norm = torch.linalg.matrix_norm(x_tensor, ord=2)
    lipschitz = 2 * spectral_norm**2 / len(x_tensor)
    learning_rate = min(0.05, float((1 / lipschitz).item()))

    return learning_rate


def train_lasso(x_tensor, y_tensor, alpha):
    weights = torch.zeros(
        (x_tensor.shape[1], 1),
        dtype=torch.float32,
        device=x_tensor.device,
        requires_grad=True,
    )
    bias = torch.zeros(1, dtype=torch.float32, device=x_tensor.device, requires_grad=True)
    learning_rate = get_learning_rate(x_tensor)

    for _ in range(EPOCHS):
        predictions = x_tensor @ weights + bias
        mse_loss = torch.mean((predictions - y_tensor) ** 2)
        mse_loss.backward()

        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad

            weights.copy_(
                torch.sign(weights)
                * torch.clamp(torch.abs(weights) - learning_rate * alpha, min=0)
            )

            weights.grad.zero_()
            bias.grad.zero_()

    return weights.detach(), bias.detach()


def predict(x_tensor, weights, bias):
    return (x_tensor @ weights + bias).squeeze().detach().cpu().numpy()


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    errors = y_pred - y_true

    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    rmse = math.sqrt(mse)
    r2 = 1 - float(np.sum(errors**2) / np.sum((y_true - y_true.mean()) ** 2))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def tune_alpha(train_data, validation_data, device):
    train_data = add_cyclical_features(train_data)
    validation_data = add_cyclical_features(validation_data)

    train_features, train_target, feature_columns = split_features_target(train_data)
    validation_features, validation_target, _ = split_features_target(validation_data)

    mean, std = fit_scaler(train_features)
    train_features = transform_features(train_features, mean, std)
    validation_features = transform_features(validation_features, mean, std)

    x_train, y_train = to_tensor(train_features, train_target, device)
    x_validation, _ = to_tensor(validation_features, validation_target, device)

    tuning_results = []

    for alpha in ALPHAS:
        weights, bias = train_lasso(x_train, y_train, alpha)
        validation_predictions = predict(x_validation, weights, bias)
        metrics = calculate_metrics(validation_target, validation_predictions)

        tuning_results.append(
            {
                "alpha": alpha,
                "validation_mae": metrics["mae"],
                "validation_mse": metrics["mse"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
            }
        )

    tuning_table = pd.DataFrame(tuning_results)
    best_alpha = tuning_table.sort_values("validation_mse").iloc[0]["alpha"]

    return float(best_alpha), tuning_table, feature_columns


def train_final_model(train_data, validation_data, test_data, best_alpha, device):
    train_validation_data = pd.concat([train_data, validation_data], ignore_index=True)

    train_validation_data = add_cyclical_features(train_validation_data)
    test_data = add_cyclical_features(test_data)

    train_features, train_target, feature_columns = split_features_target(
        train_validation_data
    )
    test_features, test_target, _ = split_features_target(test_data)

    mean, std = fit_scaler(train_features)
    train_features = transform_features(train_features, mean, std)
    test_features = transform_features(test_features, mean, std)

    x_train, y_train = to_tensor(train_features, train_target, device)
    x_test, _ = to_tensor(test_features, test_target, device)

    weights, bias = train_lasso(x_train, y_train, best_alpha)
    test_predictions = predict(x_test, weights, bias)
    test_metrics = calculate_metrics(test_target, test_predictions)

    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": weights.squeeze().detach().cpu().numpy(),
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["is_zero"] = coefficients["abs_coefficient"] < 1e-6
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)

    predictions = pd.DataFrame(
        {
            "date": test_data["date"],
            "actual_target_temp": test_target,
            "predicted_target_temp": test_predictions,
            "error": test_predictions - test_target.to_numpy(dtype=np.float32),
        }
    )

    return test_metrics, coefficients, predictions


def save_results(best_alpha, tuning_table, test_metrics, coefficients, predictions, device):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tuning_table.to_csv(RESULTS_DIR / "alpha_tuning.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / "coefficients.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    metrics = {
        "model": "Lasso Regression",
        "target": TARGET_COLUMN,
        "best_alpha": best_alpha,
        "device": str(device),
        "epochs": EPOCHS,
        "mae": test_metrics["mae"],
        "mse": test_metrics["mse"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "zero_coefficients": int(coefficients["is_zero"].sum()),
        "kept_coefficients": int((~coefficients["is_zero"]).sum()),
    }

    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "test_metrics.csv", index=False)

    top_features = coefficients.head(10)
    zero_features = coefficients[coefficients["is_zero"]]["feature"].tolist()
    zero_feature_text = ", ".join(zero_features) if zero_features else "Khong co"

    summary_lines = [
        "Lasso Regression feature analysis",
        f"Device: {device}",
        f"Best alpha: {best_alpha}",
        f"MAE: {test_metrics['mae']:.4f}",
        f"RMSE: {test_metrics['rmse']:.4f}",
        f"R2: {test_metrics['r2']:.4f}",
        f"So feature bi loai bo: {len(zero_features)}",
        f"Feature bi loai bo: {zero_feature_text}",
        "",
        "Top 10 feature quan trong nhat:",
    ]

    for _, row in top_features.iterrows():
        summary_lines.append(
            f"- {row['feature']}: coefficient = {row['coefficient']:.6f}"
        )

    (RESULTS_DIR / "feature_analysis.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Can GPU/CUDA de train model")

    device = torch.device("cuda")

    train_data = pd.read_csv(TRAIN_FILE)
    validation_data = pd.read_csv(VALIDATION_FILE)
    test_data = pd.read_csv(TEST_FILE)

    best_alpha, tuning_table, _ = tune_alpha(train_data, validation_data, device)
    test_metrics, coefficients, predictions = train_final_model(
        train_data,
        validation_data,
        test_data,
        best_alpha,
        device,
    )
    save_results(
        best_alpha,
        tuning_table,
        test_metrics,
        coefficients,
        predictions,
        device,
    )

    print(f"Device: {device} - {torch.cuda.get_device_name(0)}")
    print(f"Best alpha: {best_alpha}")
    print("Test metrics:")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"R2: {test_metrics['r2']:.4f}")
    print(f"Zero coefficients: {int(coefficients['is_zero'].sum())}")
    print(f"Kept coefficients: {int((~coefficients['is_zero']).sum())}")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
