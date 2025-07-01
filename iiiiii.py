from sklearn.datasets import load_iris
iris = load_iris()

print(iris.feature_names)
print(iris.target_names)
from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()

# Create DataFrame with selected features
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Select only sepal length and petal length
X = df[['sepal length (cm)', 'petal length (cm)']]

# Target (species)
y = iris.target
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

