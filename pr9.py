from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load the Iris dataset
X = load_iris().data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ------------------------
# AGNES (Bottom-Up Clustering)
# ------------------------
agnes_linkage = linkage(X_scaled, method='ward')  # Ward linkage used for AGNES
agnes_labels = fcluster(agnes_linkage, 3, criterion='maxclust')

# ------------------------
# DIANA (Top-Down Clustering - Simulated)
# ------------------------
diana_linkage = linkage(X_scaled, method='complete')  # Complete linkage simulates DIANA
diana_labels = fcluster(diana_linkage, 3, criterion='maxclust')

# ------------------------
# Dendrograms
# ------------------------
plt.figure(figsize=(6, 4))
dendrogram(agnes_linkage)
plt.title("AGNES - Dendrogram (Ward Linkage)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
dendrogram(diana_linkage)
plt.title("DIANA - Dendrogram (Complete Linkage)")
plt.tight_layout()
plt.show()

# ------------------------
# Cluster Visualizations using first two features
# ------------------------
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=agnes_labels)
plt.title("AGNES Clustering (using first two features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=diana_labels)
plt.title("DIANA Clustering (using first two features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ------------------------
# Compare Using Silhouette Score
# ------------------------
agnes_score = silhouette_score(X_scaled, agnes_labels)
diana_score = silhouette_score(X_scaled, diana_labels)

print(f"AGNES Silhouette Score: {agnes_score:.4f}")
print(f"DIANA Silhouette Score: {diana_score:.4f}")

if agnes_score > diana_score:
    print("AGNES gives better clustering based on higher silhouette score.")
elif diana_score > agnes_score:
    print("DIANA gives better clustering based on higher silhouette score.")
else:
    print("Both methods perform equally well based on silhouette score.")
