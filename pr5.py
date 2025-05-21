import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, mean_squared_error
)
import matplotlib.pyplot as plt

# Dummy dataset
data = {
    'Hours_Studied': [1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Features and target
X = df[['Hours_Studied']]
y = df['Passed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction on test set
y_pred = model.predict(X_test)

# Prediction on new data
new_data = pd.DataFrame({'Hours_Studied': [1.5, 3.5, 5.5, 7.5]})
new_pred = model.predict(new_data)

# Evaluation metrics
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Results
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("MSE:", mse)
print("RMSE:", rmse)
# Print predictions on new test data
print("\nPredictions on new test data:")
for hours, pred in zip(new_data['Hours_Studied'], new_pred):
    print(f"Hours Studied: {hours} -> Predicted Passed: {pred}")


# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict_proba(X)[:, 1], color='red', label='Logistic Regression Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Univariate Logistic Regression')
plt.legend()
plt.grid()
plt.show()
