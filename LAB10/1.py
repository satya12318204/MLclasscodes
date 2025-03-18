import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import ListedColormap

# Step 1: Generate the dataset
# a. 25 2-D random integer samples in the range of 1-30
a = np.random.randint(1, 31, size=(25, 2))

# b. 25 2-D random integer samples in the range of 45-75
b = np.random.randint(45, 76, size=(25, 2))

# c. 25 2-D random integer samples in the range of 150-200
c = np.random.randint(150, 201, size=(25, 2))

# Concatenate a, b, c to create the dataset
dataset = np.concatenate((a, b, c), axis=0)

# Step 2: Implement K-means clustering with the Silhouette method to find optimal k
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(dataset)
    score = silhouette_score(dataset, labels)
    silhouette_scores.append(score)

# Finding the optimal k (with maximum silhouette score)
optimal_k = k_values[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters (k): {optimal_k}")

# Step 3: Apply K-means with the optimal k value
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(dataset)
centroids = kmeans.cluster_centers_

# Plot the clustered results
plt.figure(figsize=(10, 6))
plt.title(f"K-Means Clustering with k={optimal_k}", fontsize=14)

# Create a discrete colormap with optimal_k colors
base_colors = plt.colormaps["tab10"](np.linspace(0, 1, optimal_k))  # Updated color map access
colors = [base_colors[i] for i in range(optimal_k)]  # Assign colors to each cluster

# Plot each cluster with a different color
for i in range(optimal_k):
    cluster_points = dataset[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], label=f'Cluster {i + 1}')
    
# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='red', marker='X', label='Centroids')

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()
