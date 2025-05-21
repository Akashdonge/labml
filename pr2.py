from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load data
X, y = load_iris(return_X_y=True)

# Train model using entropy criterion
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

# Make prediction on a sample input
sample = [[5.1, 3.5, 1.4, 0.2]]  # This likely belongs to class 'setosa'
prediction = model.predict(sample)
print("Predicted class:", prediction[0])

# Plot the decision tree
plt.figure(figsize=(10, 8))
plot_tree(
    model,
    filled=True,
    feature_names=load_iris().feature_names,
    class_names=load_iris().target_names
)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()
