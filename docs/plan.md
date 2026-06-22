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
- `rain_sum`

