# Kế hoạch dự án: Dự đoán nhiệt độ và xác suất mưa tại Hà Nội

## 1. Mục tiêu dự án

Xây dựng mô hình Machine Learning và Deep Learning để dự đoán thời tiết ngày mai tại Hà Nội, gồm:

- Nhiệt độ trung bình ngày mai
- Xác suất xảy ra mưa ngày mai


## 2. Nguồn dữ liệu

Sử dụng dữ liệu thời tiết lịch sử từ Open-Meteo Historical Weather API.

Khu vực lấy dữ liệu:

- Thành phố: Hà Nội
- Tọa độ tham khảo: `21.0245, 105.8412`
- Khoảng thời gian: 2015–2025

## 3. Các chỉ số thời tiết cần lấy

Các feature gốc:

- `temperature_2m_mean`
- `temperature_2m_max`
- `temperature_2m_min`
- `relative_humidity_2m_mean`
- `pressure_msl_mean`
- `cloud_cover_mean`
- `wind_speed_10m_mean`
- `wind_speed_10m_max`
- `precipitation_sum`

## 4. Tiền xử lý dữ liệu

Các bước cần làm:

- Kiểm tra và xử lý missing values: Không có missing values
- Chuyển cột ngày tháng về dạng `datetime`
- Sắp xếp dữ liệu theo thời gian
- Tạo thêm feature từ dữ liệu quá khứ

Feature tự tạo:

- `temp_range`
- `temperature_lag1`, `temperature_lag2`, `temperature_lag3`
- `temperature_rolling_3days`, `temperature_rolling_7days`
- `humidity_lag1`, `humidity_rolling_3days`
- `rain_lag1`, `rain_lag2`
- `rain_rolling_3days`, `rain_rolling_7days`
- `pressure_lag1`, `pressure_change`
- `month`, `day_of_year`

## 5. Tạo target

Dự đoán nhiệt độ ngày mai:

```python
target_temp = temperature_2m_mean của ngày t+1
```

Dự đoán mưa ngày mai:

```python
target_rain = 1 nếu precipitation_sum của ngày t+1 > 0
target_rain = 0 nếu precipitation_sum của ngày t+1 = 0
```

## 6. Chia dữ liệu

Không chia ngẫu nhiên vì đây là dữ liệu chuỗi thời gian.

Gợi ý chia:

- Train: 2015–2021
- Validation: 2022–2023
- Test: 2024–2025
