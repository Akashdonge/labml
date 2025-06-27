import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Step 1: Load dataset
data = pd.read_csv("diabetes.csv")//https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

# Step 2: Replace 0s with median in specific columns (as 0 is invalid)
invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in invalid_cols:
    data[col] = data[col].replace(0, np.nan)
    data[col] = data[col].fillna(data[col].median())

# Step 3: Normalize features using Min-Max Scaling
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Distance functions
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

# Step 6: KNN algorithm from scratch
def knn_predict(X_train, y_train, x_test, k=3, distance='euclidean'):
    distances = []
    for i in range(len(X_train)):
        if distance == 'euclidean':
            dist = euclidean(X_train[i], x_test)
        elif distance == 'manhattan':
            dist = manhattan(X_train[i], x_test)
        else:
            raise ValueError("Invalid distance metric")
        distances.append((dist, y_train.iloc[i]))
    
    distances.sort()
    k_nearest = [label for _, label in distances[:k]]
    prediction = Counter(k_nearest).most_common(1)[0][0]
    return prediction

# Step 7: Accuracy checker
def evaluate_knn(k, distance_metric):
    correct = 0
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test[i], k, distance=distance_metric)
        if pred == y_test.iloc[i]:
            correct += 1
    accuracy = correct / len(X_test)
    print(f"K={k}, Distance={distance_metric}, Accuracy={accuracy:.4f}")

# Step 8: Try multiple K values and distances
print("🔍 Evaluating KNN with different K and Distance Metrics:\n")
for k in [3, 5, 7, 9]:
    for metric in ['euclidean', 'manhattan']:
        evaluate_knn(k, metric)
