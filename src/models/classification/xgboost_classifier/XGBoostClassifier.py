from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

PROJECT_DIR = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_DIR / "data" / "split"
RESULTS_DIR = (
    PROJECT_DIR / "results" / "classification" / "rain" / "xgboost_classifier"
)

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"

TARGET_COLUMN = "target_rain"

GRID_MAX_DEPTH = [3, 4, 5]
GRID_LEARNING_RATE = [0.05, 0.1, 0.2]
GRID_REG_LAMBDA = [0.1, 1.0, 10.0]
GRID_SUBSAMPLE = [0.8, 1.0]

N_ESTIMATORS_TUNING = 15  
N_ESTIMATORS_FINAL = 50   

class PyTorchXGBTree:
    def __init__(self, max_depth=3, reg_lambda=1.0):
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X, g, h, depth=0):
        device = X.device
        if depth == self.max_depth or X.shape[0] < 2:
            self.value = -torch.sum(g) / (torch.sum(h) + self.reg_lambda)
            return self

        best_gain = 0.0
        best_criteria = None
        best_sets = None
        n_samples, n_features = X.shape

        for feat in range(n_features):
            X_column = X[:, feat]
            thresholds = torch.unique(X_column)
            
            if len(thresholds) > 10:
                quantiles = torch.linspace(0, 1, 10, device=device)
                thresholds = torch.quantile(X_column, quantiles)

            for thr in thresholds:
                left_mask = X_column <= thr
                right_mask = X_column > thr

                if not torch.any(left_mask) or not torch.any(right_mask):
                    continue

                G_L, H_L = torch.sum(g[left_mask]), torch.sum(h[left_mask])
                G_R, H_R = torch.sum(g[right_mask]), torch.sum(h[right_mask])
                G_all, H_all = G_L + G_R, H_L + H_R

                gain = 0.5 * (
                    (G_L**2 / (H_L + self.reg_lambda)) + 
                    (G_R**2 / (H_R + self.reg_lambda)) - 
                    (G_all**2 / (H_all + self.reg_lambda))
                )

                if gain > best_gain:
                    best_gain = gain.item()
                    best_criteria = (feat, thr.item())
                    best_sets = (left_mask, right_mask)

        if best_gain == 0.0:
            self.value = -torch.sum(g) / (torch.sum(h) + self.reg_lambda)
            return self

        self.feature, self.threshold = best_criteria
        left_mask, right_mask = best_sets

        self.left = PyTorchXGBTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda).fit(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        self.right = PyTorchXGBTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda).fit(X[right_mask], g[right_mask], h[right_mask], depth + 1)
        return self

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

    def fit(self, X, y):
        device = X.device
        y_pred = torch.zeros(X.shape[0], device=device)
        self.trees = []

        for i in range(self.n_estimators):
            p = 1 / (1 + torch.exp(-y_pred))  
            g = p - y                         
            h = p * (1 - p)                  

            if self.subsample < 1.0:
                indices = torch.randperm(X.shape[0], device=device)[:int(X.shape[0] * self.subsample)]
                X_b, g_b, h_b = X[indices], g[indices], h[indices]
            else:
                X_b, g_b, h_b = X, g, h

            tree = PyTorchXGBTree(max_depth=self.max_depth, reg_lambda=self.reg_lambda)
            tree.fit(X_b, g_b, h_b)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

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

def tune_xgboost(train_data, validation_data, device):
    print("--- [CUSTOM XGBOOST] GIAI ĐOẠN 1: BẮT ĐẦU QUÉT SIÊU THAM SỐ VỚI VALID ---")
    train_data = add_cyclical_features(train_data)
    validation_data = add_cyclical_features(validation_data)

    X_train, y_train = split_features_target(train_data, device)
    X_val, y_val = split_features_target(validation_data, device)
    y_val_cpu = y_val.cpu().numpy()

    best_score = -1
    best_params = {}
    tuning_results = []

    for depth in GRID_MAX_DEPTH:
        for lr in GRID_LEARNING_RATE:
            for reg in GRID_REG_LAMBDA:
                for sub in GRID_SUBSAMPLE:
                    print(f"Đang thử nghiệm: max_depth={depth}, lr={lr}, lambda={reg}, subsample={sub}...")
                    model = PyTorchXGBoostClassifier(
                        n_estimators=N_ESTIMATORS_TUNING, max_depth=depth, learning_rate=lr, reg_lambda=reg, subsample=sub
                    )
                    model.fit(X_train, y_train)
                    val_preds = model.predict(X_val)
                    score = accuracy_score(y_val_cpu, val_preds)
                    print(f"-> Điểm Accuracy trên tập Valid: {score:.4f}")

                    tuning_results.append({
                        "max_depth": depth, "learning_rate": lr, "reg_lambda": reg, "subsample": sub, "validation_accuracy": score
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {"max_depth": depth, "learning_rate": lr, "reg_lambda": reg, "subsample": sub}

    return best_params, pd.DataFrame(tuning_results)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kích hoạt phần cứng: {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Cảnh báo: Chạy trên CPU, quá trình tuning đa tham số có thể mất vài phút...")
        
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_data = pd.read_csv(TRAIN_FILE)
    validation_data = pd.read_csv(VALIDATION_FILE)

    best_params, tuning_table = tune_xgboost(train_data, validation_data, device)
    print(f"\n[KẾT QUẢ TUNING] Bộ tham số tối ưu hoàn chỉnh tìm được: {best_params}")

    print("\n--- GIAI ĐOẠN 2: HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC TRÊN TẬP GỘP (TRAIN + VALID) ---")
    train_validation_data = pd.concat([train_data, validation_data], ignore_index=True)
    train_validation_data = add_cyclical_features(train_validation_data)
    X_train_full, y_train_full = split_features_target(train_validation_data, device)

    final_model = PyTorchXGBoostClassifier(
        n_estimators=N_ESTIMATORS_FINAL,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        reg_lambda=best_params["reg_lambda"],
        subsample=best_params["subsample"]
    )
    final_model.fit(X_train_full, y_train_full)

    tuning_table.to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    
    torch.save(final_model, RESULTS_DIR / "xgboost_model.pt")
    
    pd.DataFrame([best_params]).to_csv(RESULTS_DIR / "best_hyperparameters.csv", index=False)
    print(f"\n[SUCCESS] Đã huấn luyện xong và đóng gói mô hình Pre-trained tại: {RESULTS_DIR / 'xgboost_model.pt'}")


if __name__ == "__main__":
    main()