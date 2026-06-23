from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_DIR / "data" / "split"
RESULTS_DIR = (
    PROJECT_DIR / "results" / "classification" / "rain" / "logistic_regression"
)

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

TARGET_COLUMN = "target_rain"
C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
THRESHOLDS = [round(value, 2) for value in np.arange(0.1, 0.91, 0.05)]
EPOCHS = 5000
LEARNING_RATE = 0.01


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


def train_logistic_regression(x_tensor, y_tensor, c_value):
    torch.manual_seed(42)

    model = torch.nn.Linear(x_tensor.shape[1], 1).to(x_tensor.device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    l2_penalty = 1 / c_value

    for _ in range(EPOCHS):
        optimizer.zero_grad()

        logits = model(x_tensor)
        loss = loss_function(logits, y_tensor)
        loss = loss + l2_penalty * torch.sum(model.weight**2)

        loss.backward()
        optimizer.step()

    return model


def predict_probabilities(model, x_tensor):
    logits = model(x_tensor)
    probabilities = torch.sigmoid(logits).squeeze()

    return probabilities.detach().cpu().numpy()


def calculate_roc_auc(y_true, y_probability):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_probability = np.asarray(y_probability, dtype=np.float32)

    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))

    if positive_count == 0 or negative_count == 0:
        return 0.5

    ranks = pd.Series(y_probability).rank(method="average").to_numpy()
    positive_rank_sum = np.sum(ranks[y_true == 1])

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)

    return float(auc)


def calculate_metrics(y_true, y_probability, threshold):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = (np.asarray(y_probability) >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    roc_auc = calculate_roc_auc(y_true, y_probability)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def find_best_threshold(y_true, y_probability):
    threshold_results = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(y_true, y_probability, threshold)
        threshold_results.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    threshold_table = pd.DataFrame(threshold_results)
    best_row = threshold_table.sort_values(
        ["f1", "roc_auc"],
        ascending=False,
    ).iloc[0]

    return float(best_row["threshold"]), threshold_table


def tune_c_value(train_data, validation_data, device):
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

    for c_value in C_VALUES:
        model = train_logistic_regression(x_train, y_train, c_value)
        validation_probabilities = predict_probabilities(model, x_validation)
        best_threshold, _ = find_best_threshold(
            validation_target,
            validation_probabilities,
        )
        metrics = calculate_metrics(
            validation_target,
            validation_probabilities,
            best_threshold,
        )

        tuning_results.append(
            {
                "c_value": c_value,
                "l2_penalty": 1 / c_value,
                "best_threshold": best_threshold,
                "validation_accuracy": metrics["accuracy"],
                "validation_precision": metrics["precision"],
                "validation_recall": metrics["recall"],
                "validation_f1": metrics["f1"],
                "validation_roc_auc": metrics["roc_auc"],
            }
        )

    tuning_table = pd.DataFrame(tuning_results)
    best_row = tuning_table.sort_values(
        ["validation_f1", "validation_roc_auc"],
        ascending=False,
    ).iloc[0]

    return (
        float(best_row["c_value"]),
        float(best_row["best_threshold"]),
        tuning_table,
        feature_columns,
    )


def train_final_model(train_data, validation_data, test_data, best_c, best_threshold, device):
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

    model = train_logistic_regression(x_train, y_train, best_c)
    test_probabilities = predict_probabilities(model, x_test)
    test_metrics = calculate_metrics(test_target, test_probabilities, best_threshold)

    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": model.weight.detach().cpu().numpy().squeeze(),
        }
    )
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["effect"] = np.where(
        coefficients["coefficient"] >= 0,
        "increase_rain_probability",
        "decrease_rain_probability",
    )
    coefficients = coefficients.sort_values("abs_coefficient", ascending=False)

    predictions = pd.DataFrame(
        {
            "date": test_data["date"],
            "actual_target_rain": test_target,
            "predicted_probability": test_probabilities,
            "predicted_target_rain": (test_probabilities >= best_threshold).astype(int),
        }
    )

    return test_metrics, coefficients, predictions


def calculate_baseline_metrics(train_data, test_data):
    majority_class = int(train_data[TARGET_COLUMN].mode()[0])
    test_target = test_data[TARGET_COLUMN]
    baseline_probability = np.full(len(test_data), majority_class, dtype=np.float32)

    metrics = calculate_metrics(test_target, baseline_probability, 0.5)
    metrics["model"] = "Always predict train majority class"
    metrics["majority_class"] = majority_class

    return metrics


def save_results(
    best_c,
    best_threshold,
    tuning_table,
    test_metrics,
    baseline_metrics,
    coefficients,
    predictions,
    device,
):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tuning_table.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    coefficients.to_csv(RESULTS_DIR / "coefficients.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    pd.DataFrame([baseline_metrics]).to_csv(
        RESULTS_DIR / "baseline_metrics.csv",
        index=False,
    )

    metrics = {
        "model": "Logistic Regression",
        "target": TARGET_COLUMN,
        "best_c": best_c,
        "best_threshold": best_threshold,
        "device": str(device),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
    }
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "test_metrics.csv", index=False)

    top_positive_features = coefficients.sort_values(
        "coefficient",
        ascending=False,
    ).head(10)
    top_negative_features = coefficients.sort_values("coefficient").head(10)

    summary_lines = [
        "Logistic Regression feature analysis",
        f"Device: {device}",
        f"Best C: {best_c}",
        f"Best threshold: {best_threshold}",
        f"Accuracy: {test_metrics['accuracy']:.4f}",
        f"Precision: {test_metrics['precision']:.4f}",
        f"Recall: {test_metrics['recall']:.4f}",
        f"F1: {test_metrics['f1']:.4f}",
        f"ROC-AUC: {test_metrics['roc_auc']:.4f}",
        "",
        "He so duong lam tang xac suat mua:",
    ]

    for _, row in top_positive_features.iterrows():
        summary_lines.append(
            f"- {row['feature']}: coefficient = {row['coefficient']:.6f}"
        )

    summary_lines.append("")
    summary_lines.append("He so am lam giam xac suat mua:")

    for _, row in top_negative_features.iterrows():
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

    best_c, best_threshold, tuning_table, _ = tune_c_value(
        train_data,
        validation_data,
        device,
    )
    test_metrics, coefficients, predictions = train_final_model(
        train_data,
        validation_data,
        test_data,
        best_c,
        best_threshold,
        device,
    )
    baseline_metrics = calculate_baseline_metrics(train_data, test_data)

    save_results(
        best_c,
        best_threshold,
        tuning_table,
        test_metrics,
        baseline_metrics,
        coefficients,
        predictions,
        device,
    )

    print(f"Device: {device} - {torch.cuda.get_device_name(0)}")
    print(f"Best C: {best_c}")
    print(f"Best threshold: {best_threshold}")
    print("Test metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Confusion matrix: TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TP={test_metrics['tp']}")
    print(f"Baseline F1: {baseline_metrics['f1']:.4f}")
    print(f"Saved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
