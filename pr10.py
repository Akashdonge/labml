import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load and prepare data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train models
ada = AdaBoostClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
xgb = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42).fit(X_train, y_train)

# Predictions
y_pred_ada = ada.predict(X_test)
y_pred_xgb = xgb.predict(X_test)

# Evaluation function with confusion matrix
def evaluate(y_true, y_pred, name):
    print(f"\n{name} Results:")
    print("Accuracy      :", accuracy_score(y_true, y_pred))
    print("Precision     :", precision_score(y_true, y_pred))
    print("Recall        :", recall_score(y_true, y_pred))
    print("F1 Score      :", f1_score(y_true, y_pred))
    print("ROC AUC       :", roc_auc_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Evaluate both models
evaluate(y_test, y_pred_ada, "AdaBoost")
evaluate(y_test, y_pred_xgb, "XGBoost")

# Final verdict
print("\n🟩 Final Verdict:")
if accuracy_score(y_test, y_pred_xgb) > accuracy_score(y_test, y_pred_ada):
    print("✅ XGBoost performed better overall.")
else:
    print("✅ AdaBoost performed better overall.")
