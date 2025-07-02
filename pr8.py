import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
# Step 1: Create the dataset
# data = {
#     'Age': ['Youth', 'Youth', 'Middle', 'Senior', 'Senior', 'Senior', 'Middle', 'Youth', 'Youth', 'Senior', 'Youth', 'Middle', 'Middle', 'Senior'],
#     'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium', 'High', 'Medium'],
#     'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# }
# df = pd.DataFrame(data)

#  Step 2: Encode categorical variables
# le_age = LabelEncoder()
# le_income = LabelEncoder()
# le_target = LabelEncoder()

# df['Age'] = le_age.fit_transform(df['Age'])
# df['Income'] = le_income.fit_transform(df['Income'])
# df['Buys_Computer'] = le_target.fit_transform(df['Buys_Computer'])

# Step 3: Use only Age and Income for clustering
# X = df[['Age', 'Income']].values
# Step 2: Reduce to 2 dimensions for visualization using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 3: Try different values of K
ks = [2, 3, 4]
plt.figure(figsize=(15, 4))

for idx,k in enumerate(ks):
    km=KMeans(n_clusters=k,random_state=42,n_init=10)
    cluster=km.fit_predict(X)
    plt.subplot(1,3,idx+1)
    plt.scatter(x_pca[:,0],x_pca[:,1],c=cluster,s=10)
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c='red', marker='X', s=100)
    plt.title(f'k mmeas clusterng for k={k}')
    plt.xlabel("pca 1")
    plt.ylabel("pca 2")
    plt.grid()
plt.show()
