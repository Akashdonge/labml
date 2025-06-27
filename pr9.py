import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load Iris dataset
iris = load_iris()
X = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------
# AGNES (Agglomerative)
# ------------------------
# Create linkage matrix
agnes_link = linkage(X_scaled, method='ward')

# Dendrogram for AGNES
plt.figure()
dendrogram(agnes_link)
plt.title("AGNES Dendrogram")
plt.show()

# Fit AGNES model with 3 clusters
agnes_model = AgglomerativeClustering(n_clusters=3, linkage='ward', metric='euclidean')
agnes_labels = agnes_model.fit_predict(X_scaled)

# Plot AGNES clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agnes_labels, cmap='viridis')
plt.title("AGNES Clustering (3 clusters)")
plt.show()

# ------------------------
# DIANA (Simulated)
# ------------------------
# Use 'complete' linkage for DIANA-like effect
diana_link = linkage(X_scaled, method='complete')

# Dendrogram for DIANA
plt.figure()
dendrogram(diana_link)
plt.title("DIANA Dendrogram (Simulated)")
plt.show()

# Form 3 clusters from DIANA
diana_labels = fcluster(diana_link, 3, criterion='maxclust')

# Plot DIANA clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=diana_labels, cmap='viridis')
plt.title("DIANA Clustering (3 clusters, simulated)")
plt.show()

# ------------------------
# Compare with Silhouette Score
# ------------------------
score_agnes = silhouette_score(X_scaled, agnes_labels)
score_diana = silhouette_score(X_scaled, diana_labels)

print(f"Silhouette Score - AGNES: {score_agnes:.2f}")
print(f"Silhouette Score - DIANA: {score_diana:.2f}")

if score_agnes > score_diana:
    print("AGNES gives better clusters.")
elif score_diana > score_agnes:
    print("DIANA gives better clusters.")
else:
    print("Both methods give similar results.")
