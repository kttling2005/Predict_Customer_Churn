from data_preprocessing import X_train, y_train, X_test, y_test

from sklearn.linear_model import LogisticRegression
# Khởi tạo mô hình Logistic Regression
lr_model = LogisticRegression(class_weight=None, random_state=42)

#LogisticRegression
# Huấn luyện mô hình trên tập train
lr_model.fit(X_train, y_train)

# Dự đoán nhãn phân loại (0: Không rời bỏ, 1: Rời bỏ)
y_pred = lr_model.predict(X_test)

# Dự đoán xác suất rời bỏ (hữu ích cho việc đánh giá độ đo ROC-AUC)
y_prob = lr_model.predict_proba(X_test)[:, 1]

print((y_pred==1).sum())
print(y_pred,y_prob)

from sklearn.ensemble import RandomForestClassifier

#Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Dự đoán nhãn phân loại
y_pred2 = rf_model.predict(X_test)

# Dự đoán xác suất cho từng lớp
y_prob2 = rf_model.predict_proba(X_test)

print("Số người Exit: ",(y_pred2==1).sum())
print("y_pred2:", y_pred2)
print("y_prob2:",y_prob2, sep="\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)

from xgboost import XGBClassifier

xg_boost_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
xg_boost_model.fit(X_train, y_train)

# Dự đoán nhãn phân loại (0 hoặc 1)
y_pred3 = xg_boost_model.predict(X_test)

# Dự đoán xác suất cho từng lớp
y_prob3 = xg_boost_model.predict_proba(X_test)

print("Số người Exit:", (y_pred3==1).sum())
print("y_pred3: ",y_pred3)
print("y_prob3: ",y_prob3,sep="\n")
cm2 = confusion_matrix(y_test, y_pred3)
print(cm2)
