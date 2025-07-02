import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
print(iris.feature_names)
print(iris.target_names)
# Step 1: Load iris data and convert to DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target   # Add species as string labels\

# Step 3: Features and target
X = df[iris.feature_names]
y = df['species']

# Step 4: Train decision tree using entropy (ID3)
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X, y)

# Step 5: Print tree rules
print("\nDecision Tree Rules:\n")
print(export_text(clf, feature_names=X.columns.tolist()))

# Step 6: Visualize decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=X.columns, class_names=le_species.classes_, filled=True)
plt.title("Iris Decision Tree (ID3 - Entropy)")
plt.tight_layout()
plt.show()

# Step 7: Predict a new sample
# Example: sepal_length=6.0, sepal_width=3.0, petal_length=4.8, petal_width=1.8
sample = pd.DataFrame([{
    'sepal length (cm)': 6.0,
    'sepal width (cm)': 3.0,
    'petal length (cm)': 4.8,
    'petal width (cm)': 1.8
}])

prediction = clf.predict(sample)[0]
predicted_class = le_species.inverse_transform([prediction])[0]
print("\nPrediction for sample [6.0, 3.0, 4.8, 1.8]:", predicted_class)
