# Kế hoạch Thực hiện Lasso Regression 

## Bước 1: Tiền xử lý Đặc trưng (Feature Engineering)
- [ ] **Mã hóa chu kỳ (Cyclical):** Biến đổi `month` và `day_of_year` thành 4 cột mới: `month_sin`, `month_cos`, `day_sin`, `day_cos`.
- [ ] **Dọn rác:** Xóa 2 cột số nguyên `month` và `day_of_year` gốc.

## Bước 2: Chuẩn bị Dữ liệu (Splitting & Scaling)
- [ ] **Khởi tạo Scaler:** Dùng `StandardScaler`.
- [ ] **Chuẩn hóa:** - Chỉ `.fit_transform()` trên tập **Train**.
  - Chỉ `.transform()` trên tập **Valid** và **Test**.

## Bước 3: Tìm mức phạt tối ưu (Tuning Alpha)
- [ ] Lên danh sách các giá trị phạt $\alpha$ cần thử: `[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`.
- [ ] Lặp qua từng giá trị $\alpha$:
  - Huấn luyện Lasso trên tập **Train**.
  - Tính sai số MSE trên tập **Valid**.
- [ ] Ghi nhận giá trị $\alpha$ cho ra sai số thấp nhất trên Valid.

## Bước 4: Đánh giá Chung cuộc (Testing & Analysis) (Lưu kết quả đánh giá vào folder results)
- [ ] Khởi tạo mô hình Lasso cuối cùng với **$\alpha$ tốt nhất**.
- [ ] Gộp tập Train và Valid lại để mô hình học thêm
- [ ] **Dự đoán:** Đánh giá sai số (MAE, RMSE, R^2) chính thức trên tập **Test**.
- [ ] **Trích xuất Đặc trưng:** In danh sách các hệ số (coefficients). Phân tích xem Lasso đã vứt bỏ (gán = 0) và giữ lại những biến nào quan trọng nhất.