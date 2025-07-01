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
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

# Print dataset keys
print("Keys in the dataset:\n", data.keys())
print("\nFeature Names:\n", data.feature_names)
print("\nTarget Names:\n", data.target_names)
import pandas as pd

# Convert to DataFrame for readability
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print("\nFirst 5 rows of the dataset:\n", df.head())


