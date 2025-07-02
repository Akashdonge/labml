import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Vary n_estimators
tree_counts = [1, 5, 10, 20, 50, 100, 200]
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for n in tree_counts:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    f1_list.append(f1_score(y_test, y_pred))

# Step 4: Plot Results
plt.figure(figsize=(12, 8))
plt.plot(tree_counts, accuracy_list, label='Accuracy', marker='o')
plt.plot(tree_counts, precision_list, label='Precision', marker='s')
plt.plot(tree_counts, recall_list, label='Recall', marker='^')
plt.plot(tree_counts, f1_list, label='F1 Score', marker='d')
plt.title("Random Forest Performance vs Number of Trees")
plt.xlabel("Number of Trees (n_estimators)")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
