from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          
        self.threshold = threshold      
        self.left = left                
        self.right = right              
        self.value = value              

class PyTorchXGBTree:
    def __init__(self, max_depth=3, reg_lambda=1.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def predict(self, X):
        device = X.device
        preds = torch.zeros(X.shape[0], device=device)
        if self.value is not None:
            preds.fill_(self.value)
            return preds
        
        left_mask = X[:, self.feature] <= self.threshold
        right_mask = ~left_mask

        if torch.any(left_mask):
            preds[left_mask] = self.left.predict(X[left_mask])
        if torch.any(right_mask):
            preds[right_mask] = self.right.predict(X[right_mask])
        return preds

class PyTorchXGBoostClassifier:
    def __init__(self, n_estimators=30, max_depth=3, learning_rate=0.1, reg_lambda=1.0, subsample=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.trees = []

    def predict_proba(self, X):
        device = X.device
        raw = torch.zeros(X.shape[0], device=device)
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X)
        return (1 / (1 + torch.exp(-raw))).cpu().numpy()

    def predict(self, X):
        device = X.device
        raw = torch.zeros(X.shape[0], device=device)
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X)
        return ((1 / (1 + torch.exp(-raw))) > 0.5).to(torch.int32).cpu().numpy()


PROJECT_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_DIR / "data" / "split"
RESULTS_DIR = (
    PROJECT_DIR / "results" / "classification" / "rain" / "xgboost_classifier"
)

TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = RESULTS_DIR / "xgboost_model.pt"
HYPERPARAMS_FILE = RESULTS_DIR / "best_hyperparameters.csv"

TARGET_COLUMN = "target_rain"


def add_cyclical_features(weather_data):
    weather_data = weather_data.copy()
    weather_data["month_sin"] = np.sin(2 * np.pi * weather_data["month"] / 12)
    weather_data["month_cos"] = np.cos(2 * np.pi * weather_data["month"] / 12)
    weather_data["day_sin"] = np.sin(2 * np.pi * weather_data["day_of_year"] / 365)
    weather_data["day_cos"] = np.cos(2 * np.pi * weather_data["day_of_year"] / 365)
    return weather_data.drop(columns=["month", "day_of_year"])

def split_features_target(weather_data, device):
    ignore_columns = ["date", "target_temp", "target_rain"]
    feature_columns = [col for col in weather_data.columns if col not in ignore_columns]
    X_tensor = torch.tensor(weather_data[feature_columns].to_numpy(), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(weather_data[TARGET_COLUMN].to_numpy(), dtype=torch.float32, device=device)
    return X_tensor, y_tensor


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại {MODEL_FILE}. Bạn cần chạy file XGBoostClassifier.py trước!"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- [HỆ THỐNG ĐÁNH GIÁ] KHỞI ĐỘNG TRÊN THIẾT BỊ: {device} ---")

    print("Đang tải mô hình Pre-trained XGBoost...")
    import sys
    sys.modules['__main__'].PyTorchXGBoostClassifier = PyTorchXGBoostClassifier
    sys.modules['__main__'].PyTorchXGBTree = PyTorchXGBTree
    sys.modules['__main__'].Node = Node

    final_model = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    test_data = pd.read_csv(TEST_FILE)
    processed_test = add_cyclical_features(test_data)
    X_test_tensor, y_test_tensor = split_features_target(processed_test, device)
    
    y_test_numpy = y_test_tensor.cpu().numpy().astype(np.int32)

    print("Đang chạy dự báo trên tập Test biệt lập...")
    test_preds = final_model.predict(X_test_tensor)
    test_probs = final_model.predict_proba(X_test_tensor)

    test_acc = accuracy_score(y_test_numpy, test_preds)
    cm = confusion_matrix(y_test_numpy, test_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n==================================================")
    print(f"🏆TEST ACCURACY CUỐI CÙNG: {test_acc:.4f}")
    print("==================================================")
    print("\nClassification Report Chi Tiết:")
    print(classification_report(y_test_numpy, test_preds))
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    best_hyper = pd.read_csv(HYPERPARAMS_FILE).iloc[0]

    predictions_df = pd.DataFrame({
        "date": test_data["date"],
        "actual_target_rain": y_test_numpy,
        "predicted_probability": test_probs,
        "predicted_target_rain": test_preds
    })
    predictions_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    summary_metrics = pd.DataFrame([{
        "model": "Custom_PyTorch_XGBoost_Classifier_From_Scratch",
        "best_max_depth": int(best_hyper["max_depth"]),
        "best_learning_rate": float(best_hyper["learning_rate"]),
        "best_reg_lambda": float(best_hyper["reg_lambda"]),
        "best_subsample": float(best_hyper["subsample"]),
        "test_accuracy": test_acc,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }])
    summary_metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    print(f"\n[SUCCESS] Đã xuất file báo cáo test_metrics.csv và test_predictions.csv vào: {RESULTS_DIR}")


if __name__ == "__main__":
    main()