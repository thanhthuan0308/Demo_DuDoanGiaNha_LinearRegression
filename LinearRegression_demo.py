import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Bước 1: Tải và chuẩn bị dữ liệu ---
print("--- Tải dữ liệu ---")
# Đọc dữ liệu từ file CSV
df = pd.read_csv('gia_nha_hcm.csv')

print("5 dòng dữ liệu đầu tiên:")
print(df.head())
print("\nThông tin về dữ liệu:")
df.info()

# Bước 2: Tiền xử lý dữ liệu ---
print("\n--- Tiền xử lý dữ liệu ---")
df_processed = pd.get_dummies(df, columns=['ViTri'], drop_first=True)
print("Dữ liệu sau khi xử lý cột 'ViTri' (One-Hot Encoding):")
print(df_processed.head())

# Tách dữ liệu thành features (X) và target (y)
X = df_processed.drop('GiaNha', axis=1) # Bỏ cột giá nhà để lấy features
y = df_processed['GiaNha']              # Cột giá nhà là target

# Bước 3: Chia dữ liệu thành tập train và test ---
# 80% dữ liệu dùng để huấn luyện mô hình, 20% dùng để kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nKích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# Bước 4: Xây dựng và Huấn luyện mô hình ---
print("\n--- Huấn luyện mô hình Hồi quy Tuyến tính ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Huấn luyện mô hình thành công!")

# Bước 5: Đánh giá mô hình ---
print("\n--- Đánh giá mô hình ---")
# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán các chỉ số đánh giá
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} (tỷ VNĐ)")
print(f"R-squared (R²): {r2:.2f}")
print(f"-> Mô hình của chúng ta giải thích được khoảng {r2:.0%} sự biến thiên của giá nhà.")

# In ra các hệ số của mô hình để xem mức độ ảnh hưởng của từng yếu tố
print("\n--- Các hệ số của mô hình ---")
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
print(f"Hệ số chặn (Intercept): {model.intercept_:.2f}")

# Bước 6: Sử dụng mô hình để dự đoán cho một ngôi nhà mới (CÁCH KHẮC PHỤC LỖI) ---
print("\n--- Dự đoán cho một ngôi nhà mới ---")

# Nhập thông tin nhà mới cần dự đoán ---
dien_tich_moi = 100
so_phong_ngu_moi = 3
vi_tri_moi = 'Bình Thạnh'

# 1. Tạo một DataFrame rỗng với đúng các cột của X_train và điền giá trị 0
# Đây là bước quan trọng để đảm bảo cấu trúc hoàn toàn khớp
nha_moi_df = pd.DataFrame(columns=X.columns)
nha_moi_df.loc[0] = 0

# 2. Điền các giá trị cơ bản (không phải vị trí)
nha_moi_df['DienTich'] = dien_tich_moi
nha_moi_df['SoPhongNgu'] = so_phong_ngu_moi

# 3. Xử lý cột vị trí (One-Hot Encoding) một cách tự động
vi_tri_column = f'ViTri_{vi_tri_moi}'

if vi_tri_column in nha_moi_df.columns:
    nha_moi_df[vi_tri_column] = 1

print("\nDữ liệu đầu vào cho việc dự đoán (đã được xử lý):")
print(nha_moi_df)

# 4. Thực hiện dự đoán với dữ liệu đã được chuẩn hóa đúng cấu trúc
gia_du_doan = model.predict(nha_moi_df)

print(f"\nThông tin nhà mới: {dien_tich_moi}m2, {so_phong_ngu_moi} phòng ngủ, tại {vi_tri_moi}")
print(f"Giá nhà dự đoán là: {gia_du_doan[0]:.2f} tỷ VNĐ")

vi_tri_moi_2 = 'Gò Vấp'
nha_moi_df_2 = pd.DataFrame(columns=X.columns)
nha_moi_df_2.loc[0] = 0
nha_moi_df_2['DienTich'] = dien_tich_moi
nha_moi_df_2['SoPhongNgu'] = so_phong_ngu_moi
