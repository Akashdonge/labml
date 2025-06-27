import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Train AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
ada_preds = ada_model.predict(X_test)

# Step 5: Train XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Step 6: Evaluate Both
def evaluate_model(name, y_true, y_pred, model):
    print(f"------ {name} ------")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, model.predict_proba(X_test)[:, 1]))
    print(classification_report(y_true, y_pred))

evaluate_model("AdaBoost", y_test, ada_preds, ada_model)
evaluate_model("XGBoost", y_test, xgb_preds, xgb_model)
