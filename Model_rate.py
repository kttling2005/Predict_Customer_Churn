from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Hyperparameter_tuning import models_tuned
from data_preprocessing import X_test, y_test
import matplotlib.pyplot as plt

#Confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models_tuned.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Không Churn', 'Churn'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.show()

#Accuracy, Precision, Recall và F1-score
from sklearn.metrics import classification_report

for name, model in models_tuned.items():
    y_pred = model.predict(X_test)
    print(f"===== {name} =====")
    print(classification_report(y_test, y_pred, target_names=['Không Churn', 'Churn']))

#ROC-AUC
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(7, 6))
for name, model in models_tuned.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('So sánh đường cong ROC giữa các mô hình')
plt.legend()
plt.show()