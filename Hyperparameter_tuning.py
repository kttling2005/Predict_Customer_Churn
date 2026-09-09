from Train_model import lr_model, rf_model, xg_boost_model
from data_preprocessing import X_train, y_train, X_test, y_test
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Đánh giá sơ bộ từng mô hình bằng ROC-AUC trung bình qua 5 fold
for name, model in [('Logistic Regression', lr_model),
                     ('Random Forest', rf_model),
                     ('XGBoost', xg_boost_model)]:
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"{name}: AUC trung bình = {scores.mean():.4f} (+/- {scores.std():.4f})")

from sklearn.model_selection import RandomizedSearchCV

# Random Search cho Random Forest
rf_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 8, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=30, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rf_random.fit(X_train, y_train)
print("Best params (RF):", rf_random.best_params_)
print("Best AUC (RF):", rf_random.best_score_)

# Random Search cho XGBoost
xgb_param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss'),
    param_distributions=xgb_param_dist,
    n_iter=30, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
xgb_random.fit(X_train, y_train)
print("Best params (XGB):", xgb_random.best_params_)
print("Best AUC (XGB):", xgb_random.best_score_)

# Grid Search cho Logistic Regression (không gian tham số nhỏ, có thể vét cạn)
from sklearn.model_selection import GridSearchCV

lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs'],
}

lr_grid = GridSearchCV(
    LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    param_grid=lr_param_grid,
    cv=cv, scoring='roc_auc', n_jobs=-1
)
lr_grid.fit(X_train, y_train)
print("Best params (LR):", lr_grid.best_params_)
print("Best AUC (LR):", lr_grid.best_score_)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

models_tuned = {
    'Logistic Regression': lr_grid.best_estimator_,
    'Random Forest': rf_random.best_estimator_,
    'XGBoost': xgb_random.best_estimator_,
}

results = []
for name, model in models_tuned.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
    })

results_df = pd.DataFrame(results)
print(results_df)

for name, model in models_tuned.items():
    y_pred = model.predict(X_test)
    print(f"{name}: Số khách hàng dự đoán rời bỏ = {(y_pred==1).sum()} / {len(y_pred)}")