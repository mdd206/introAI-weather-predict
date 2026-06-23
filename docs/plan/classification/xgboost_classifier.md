# Kế hoạch XGBoost Classifier

Mục tiêu: dự đoán `target_rain`.

Các bước chính:

- Dùng dữ liệu trong `data/split`.
- Train trên train, chọn tham số bằng validation.
- Đánh giá accuracy, precision, recall, F1, ROC-AUC trên test.
- Lưu kết quả vào `results/classification/rain/xgboost_classifier/`.
