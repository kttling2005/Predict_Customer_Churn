from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
df = pd.read_csv('Churn_Modelling.csv')

#Loại bỏ các cột không cần thiết
X = df.drop(columns=['RowNumber', 'CustomerId','Surname'])

#Xử lý giá trị khuyết
print("Giá trị khuyết:")
print(X.isnull().sum())

#Kiểm tra dữ liệu trùng lặp
duplicates_count = X.duplicated().sum()
print(f'Số lượng dòng trùng lặp: {duplicates_count}')

#Chuyển đổi kiểu dữ liệu

# Chuyển đổi các biến danh mục sang dạng 'category' hoặc 'object' phục vụ cho mã hóa
categorical_cols = ['Geography', 'Gender']
for col in categorical_cols:
  X[col] = X[col].astype('category')

# Đảm bảo các biến số đúng định dạng số thực hoặc số nguyên
numeric_cols = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'EstimatedSalary',
    'IsActiveMember',
]
for col in numeric_cols:
  if 'int64' in str(X[col].dtype):
    X[col] = X[col].astype(np.int64)
  else:
    X[col] = X[col].astype(np.float64)

# Thực hiện One-Hot Encoding cho các biến phân loại Geography và Gender
X_encoded = pd.get_dummies(
    X, columns=['Geography', 'Gender'], drop_first=True #Lược bỏ cột đầu tiên trong mỗi nhóm biến phân loại
)

#Feature Engineering
# Tạo các đặc trưng mới từ dữ liệu gốc
X_encoded['BalanceSalaryRatio'] = X_encoded['Balance'] / (
    X_encoded['EstimatedSalary'] + 1e-5 #tranh loi chia cho 0
)
X_encoded['TenureByAge'] = X_encoded['Tenure'] / (X_encoded['Age'] + 1e-5)

#Chuẩn hóa dữ liệu
scaler = StandardScaler()

numerical_cols = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'EstimatedSalary',
    'BalanceSalaryRatio',
    'TenureByAge',
]
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
print(X_encoded)

#Chia tập dữ liệu
y = X_encoded['Exited']

X_train,X_test,y_train,y_test = train_test_split(
    X_encoded.drop(columns=['Exited']), y, test_size = 0.2, random_state = 42, stratify = y
)#random_state: là một số nguyên được dùng để khởi tạo bộ tạo số ngẫu nhiên

#Xử lý mất cân bằng dữ liệu bằng p2 mẫu nhân tạo
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

