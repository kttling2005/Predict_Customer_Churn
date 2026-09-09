import shap
from Hyperparameter_tuning import rf_random,xgb_random
from data_preprocessing import X_train, X_test, y_train, y_test
import pandas as pd

# Feature Importance truyền thống (Random Forest / XGBoost)
importances = rf_random.best_estimator_.feature_importances_
feat_names = X_train.columns
imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
imp_df = imp_df.sort_values('Importance', ascending=False)
print(imp_df)

# Phân tích SHAP cho mô hình được chọn (ví dụ XGBoost)
explainer = shap.TreeExplainer(xgb_random.best_estimator_)
shap_values = explainer.shap_values(X_test)

# Biểu đồ tổng quan mức độ ảnh hưởng toàn cục
shap.summary_plot(shap_values, X_test, feature_names=feat_names)

# Biểu đồ giải thích cho một khách hàng cụ thể (local explanation)
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], feature_names=feat_names)

from Hyperparameter_tuning import models_tuned
# Phân nhóm khách hàng theo xác suất churn dự đoán
y_prob_best = models_tuned['XGBoost'].predict_proba(X_test)[:, 1]  # thay bằng best model
risk_df = pd.DataFrame({'CustomerIndex': X_test.index, 'ChurnProbability': y_prob_best})

def risk_group(p):
    if p >= 0.7:
        return 'Nguy cơ cao'
    elif p >= 0.4:
        return 'Nguy cơ trung bình'
    else:
        return 'Nguy cơ thấp'

risk_df['RiskGroup'] = risk_df['ChurnProbability'].apply(risk_group)
print(risk_df['RiskGroup'].value_counts())