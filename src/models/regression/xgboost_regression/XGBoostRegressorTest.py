from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# ĐỊNH NGHĨA CẤU TRÚC ĐỂ PYTORCH GIẢI NÉN FILE MÔ HÌNH HỒI QUY (.PT)
# =====================================================================
class PyTorchXGBRegressorTree:
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


class PyTorchXGBoostRegressor:
    def __init__(self, n_estimators=30, max_depth=3, learning_rate=0.1, reg_lambda=1.0, subsample=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.trees = []
        self.base_pred = 0.0

    def predict(self, X):
        device = X.device
        raw = torch.zeros(X.shape[0], device=device) + self.base_pred
        for tree in self.trees:
            raw += self.learning_rate * tree.predict(X)
        return raw.cpu().numpy()


# =====================================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN ĐỌC FILE PRE-TRAINED & DATA TEST
# =====================================================================
PROJECT_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_DIR / "data" / "split"
RESULTS_DIR = (
    PROJECT_DIR / "results" / "regression" / "temperature" / "xgboost_regression"
)

TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = RESULTS_DIR / "xgboost_regressor_model.pt"
HYPERPARAMS_FILE = RESULTS_DIR / "best_hyperparameters.csv"

TARGET_COLUMN = "target_temp"

# =====================================================================
# 2. CÁC HÀM TIỀN XỬ LÝ (ĐỒNG BỘ 100% VỚI FILE TRAIN)
# =====================================================================
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


# =====================================================================
# 3. QUY TRÌNH NẠP MÔ HÌNH PRE-TRAINED & ĐÁNH GIÁ SIÊU TỐC
# =====================================================================
def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file mô hình tại {MODEL_FILE}. Bạn cần chạy file XGBoostRegressor.py trước!"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- [HỆ THỐNG ĐÁNH GIÁ HỒI QUY] KHỞI ĐỘNG TRÊN THIẾT BỊ: {device} ---")

    # BƯỚC 1: NẠP MÔ HÌNH PRE-TRAINED HỒI QUY LÊN BỘ NHỚ
    print("Đang tải mô hình Pre-trained XGBoost Regressor...")
    import sys
    sys.modules['__main__'].PyTorchXGBoostRegressor = PyTorchXGBoostRegressor
    sys.modules['__main__'].PyTorchXGBRegressorTree = PyTorchXGBRegressorTree

    final_model = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    # BƯỚC 2: ĐỌC VÀ TIỀN XỬ LÝ TẬP TEST ĐỘC LẬP
    test_data = pd.read_csv(TEST_FILE)
    processed_test = add_cyclical_features(test_data)
    X_test_tensor, y_test_tensor = split_features_target(processed_test, device)
    
    y_test_numpy = y_test_tensor.cpu().numpy()

    # BƯỚC 3: DỰ BÁO NGAY LẬP TỨC (KHÔNG HUẤN LUYỆN LẠI)
    print("Đang chạy dự báo nhiệt độ trên tập Test biệt lập...")
    test_preds = final_model.predict(X_test_tensor)

    # BƯỚC 4: TÍNH TOÁN CÁC CHỈ SỐ ĐO LƯỜNG ĐỘ LỖI HỒI QUY
    mae = mean_absolute_error(y_test_numpy, test_preds)
    mse = mean_squared_error(y_test_numpy, test_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_numpy, test_preds)

    print("\n==================================================")
    print(f"🏆TEST:")
    print("==================================================")
    print(f"-> Mean Absolute Error (MAE)      : {mae:.4f} °C")
    print(f"-> Root Mean Squared Error (RMSE)  : {rmse:.4f} °C")
    print(f"-> Coefficient of Determination (R²): {r2:.4f}")
    print("==================================================")

    # BƯỚC 5: GHI KẾT QUẢ DỰ BÁO VÀO THƯ MỤC RESULTS/
    best_hyper = pd.read_csv(HYPERPARAMS_FILE).iloc[0]

    predictions_df = pd.DataFrame({
        "date": test_data["date"],
        "actual_target_temp": y_test_numpy,
        "predicted_target_temp": test_preds,
        "absolute_error": np.abs(y_test_numpy - test_preds)
    })
    predictions_df.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    summary_metrics = pd.DataFrame([{
        "model": "Custom_PyTorch_XGBoost_Regressor_From_Scratch",
        "best_max_depth": int(best_hyper["max_depth"]),
        "best_learning_rate": float(best_hyper["learning_rate"]),
        "best_reg_lambda": float(best_hyper["reg_lambda"]),
        "best_subsample": float(best_hyper["subsample"]),
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2
    }])
    summary_metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    print(f"\n[SUCCESS] Đã xuất file báo cáo test_metrics.csv và test_predictions.csv vào: {RESULTS_DIR}")


if __name__ == "__main__":
    main()