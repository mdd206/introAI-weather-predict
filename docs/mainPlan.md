# DỰ ÁN MÔN HỌC - DỰ BÁO XÁC SUẤT MƯA TRONG 1, 2, 3 GIỜ TIẾP THEO

## 1. DỮ LIỆU RAW
- Lấy dữ liệu thời tiết Hà Nội từ Open-Meteo theo từng giờ, giai đoạn 2019-2025.
- File code: `src/01_download_raw_data.ipynb`
- File raw: `data/raw/hanoi_weather_2019_2025.csv`
- Các cột chính: `precipitation`, `relative_humidity_2m`, `surface_pressure`, `temperature_2m`, `wind_speed_10m`, `wind_direction_10m`, `cloud_cover`.

## 2. TIỀN XỬ LÝ
- File code: `src/02_preprocess_split_data.ipynb`
- Sắp xếp dữ liệu theo `datetime`, bỏ trùng thời gian.
- Tạo đặc trưng thời gian: `hour`, `month`, `dayofweek`.
- Tạo đặc trưng quá khứ: lag 1h, 2h, 3h cho một số cột thời tiết.
- Tạo rolling lượng mưa: `precipitation_rolling_3h`, `precipitation_rolling_6h`.
- Tạo nhãn sự kiện mưa để huấn luyện mô hình xác suất:
  - `rain_next_1h`: sau 1 giờ có mưa hay không.
  - `rain_next_2h`: sau 2 giờ có mưa hay không.
  - `rain_next_3h`: sau 3 giờ có mưa hay không.
- Ngưỡng có mưa: `precipitation >= 0.1`.

## 3. CHIA DỮ LIỆU
- Không shuffle vì đây là dữ liệu theo thời gian.
- Train: 2019-2023.
- Valid: 2024.
- Test: 2025.
- Valid dùng để kiểm tra/chọn cách làm.
- Test chỉ dùng cho kết quả cuối cùng.
- Sau khi chốt cách làm bằng valid, gộp train + valid để fit lại rồi mới đánh giá cuối trên test.
- Nếu không có bước valid thì thực hiện test cuối trực tiếp theo kế hoạch ban đầu.
- File output:
  - `data/processed/hanoi_weather_processed.csv`
  - `data/processed/train.csv`
  - `data/processed/valid.csv`
  - `data/processed/test.csv`

## 4. MỤC TIÊU MÔ HÌNH
- Mô hình không chỉ dự đoán `0/1`, mà dự đoán xác suất:
  - `P(rain_next_1h = 1)`
  - `P(rain_next_2h = 1)`
  - `P(rain_next_3h = 1)`
- Khi cần cảnh báo mưa, có thể chọn threshold sau, ví dụ xác suất >= 0.5.
- Chỉ số đánh giá nên dùng: `Log Loss`, `Brier Score`, `ROC-AUC`, `PR-AUC`, và có thể xem thêm `Precision`, `Recall`, `F1` theo một threshold cụ thể.

## 5. BASELINE ĐƠN GIẢN
- File code: `src/03_baseline_month_hour.ipynb`
- Chưa dùng model học máy.
- Cách làm: tính xác suất mưa theo `month` và `hour` từ dữ liệu dùng để fit baseline.
- Baseline này không có bước chọn model/tham số nên không cần đánh giá valid.
- Gộp train + valid, tính xác suất mưa theo `month` và `hour`, sau đó đánh giá final trên test.
- Threshold khi đổi xác suất sang nhãn: `0.5`.
- File kết quả:
  - `data/results/baseline_month_hour_metrics.csv`
  - `data/results/baseline_month_hour_test_predictions.csv`

## 6. LOGISTIC REGRESSION
- File code: `src/04_logistic_regression_gpu.ipynb`
- Dùng PyTorch để có thể chạy GPU nếu máy có CUDA.
- Model: `Linear(input_dim, 3)`, cho ra 3 xác suất mưa sau 1h, 2h, 3h.
- Biến chu kỳ được đổi sang sin/cos: `hour`, `month`, `dayofweek`, `wind_direction_10m`.
- Chuẩn hóa feature bằng `StandardScaler`.
- Hàm mất mát: `BCEWithLogitsLoss`.
- Train trên train, dùng valid để chọn số epoch.
- Sau đó gộp train + valid, train lại và đánh giá final trên test.
- Threshold khi đổi xác suất sang nhãn: `0.5`.
- File kết quả:
  - `data/results/logistic_regression_training_history.csv`
  - `data/results/logistic_regression_test_metrics.csv`
  - `data/results/logistic_regression_test_predictions.csv`

## 7. LSTM
- File code: `src/05_lstm_gpu.ipynb`
- Chạy bằng env `weather-predict` và bắt buộc dùng GPU/CUDA, không fallback sang CPU.
- Input là chuỗi 24 giờ thời tiết gần nhất.
- Model: LSTM 1 layer, hidden size 64, output 3 xác suất mưa sau 1h, 2h, 3h.
- Biến chu kỳ được đổi sang sin/cos: `hour`, `month`, `dayofweek`, `wind_direction_10m`.
- Không dùng các cột lag vì LSTM đã học trực tiếp từ chuỗi quá khứ.
- Chuẩn hóa feature bằng `StandardScaler`.
- Hàm mất mát: `BCEWithLogitsLoss`.
- Train trên train, dùng valid loss để chọn số epoch.
- Sau đó gộp train + valid, train lại và đánh giá final trên test.
- Threshold khi đổi xác suất sang nhãn: `0.5`.
- File kết quả:
  - `data/results/lstm_training_history.csv`
  - `data/results/lstm_test_metrics.csv`
  - `data/results/lstm_test_predictions.csv`

## 8. RANDOM FOREST
- File code: `src/06_random_forest.ipynb`
- Dùng `RandomForestClassifier` của scikit-learn.
- Train 3 model riêng cho 3 nhãn:
  - `rain_next_1h`
  - `rain_next_2h`
  - `rain_next_3h`
- Mỗi model trả về xác suất mưa bằng `predict_proba`.
- Dùng đúng cơ chế Random Forest: bootstrap mẫu dữ liệu và chọn ngẫu nhiên một phần feature ở mỗi cây (`max_features="sqrt"`).
- Giữ các biến thời gian ở dạng gốc: `hour`, `month`, `dayofweek`.
- Đổi hướng gió `wind_direction_10m` sang `wind_dir_sin`, `wind_dir_cos` vì đây là biến vòng tròn 0-360 độ.
- Không cần chuẩn hóa feature bằng `StandardScaler` vì Random Forest dựa trên cây quyết định.
- Train trên train, dùng valid để chọn bộ tham số tốt nhất theo `log_loss` từ 7 tổ hợp tham số.
- Sau đó gộp train + valid, train lại 3 model final và đánh giá cuối trên test.
- Threshold khi đổi xác suất sang nhãn: `0.5`.
- File kết quả:
  - `data/results/random_forest_tuning_results.csv`
  - `data/results/random_forest_test_metrics.csv`
  - `data/results/random_forest_test_predictions.csv`
