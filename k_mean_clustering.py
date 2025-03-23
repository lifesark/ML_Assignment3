# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import io


print("Uupload the dataset file :")
uploaded = files.upload()
file_name = next(iter(uploaded))
data = pd.read_csv(io.BytesIO(uploaded[file_name]))

print("Display the first few rows of the dataset")
print("Original Dataset:")
print(data.head())


data_normalized = (data - data.mean()) / data.std()
print("\nNormalized Dataset:")
print(data_normalized.head())

def k_means(data, k, max_iters=100):

    np.random.seed(42)


    centroids = data.sample(n=k).to_numpy()

    for i in range(max_iters):

        distances = np.linalg.norm(data.to_numpy()[:, np.newaxis] - centroids, axis=2)


        clusters = np.argmin(distances, axis=1)


        prev_centroids = centroids.copy()


        new_centroids = np.array([data.to_numpy()[clusters == j].mean(axis=0)
                                 for j in range(k)])


        if np.all(np.isclose(prev_centroids, new_centroids)):
            print(f"Converged after {i+1} iterations")
            break

        centroids = new_centroids

    return clusters, centroids


print("\nRunning K-means with k=2...")
clusters_k2, centroids_k2 = k_means(data_normalized, k=2)


print("\nRunning K-means with k=3...")
clusters_k3, centroids_k3 = k_means(data_normalized, k=3)


plt.figure(figsize=(10, 6))
plt.scatter(data_normalized['x1'], data_normalized['x2'],
           c=clusters_k2, cmap='viridis', label='Clusters')
plt.scatter(centroids_k2[:, 0], centroids_k2[:, 1],
           c='red', marker='x', s=100, label='Centroids')
plt.title('K-means Clustering (k=2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('kmeans_k2.png')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(data_normalized['x1'], data_normalized['x2'],
           c=clusters_k3, cmap='viridis', label='Clusters')
plt.scatter(centroids_k3[:, 0], centroids_k3[:, 1],
           c='red', marker='x', s=100, label='Centroids')
plt.title('K-means Clustering (k=3)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('kmeans_k3.png')
plt.show()


print("\nCluster statistics for k=2:")
for i in range(2):
    count = np.sum(clusters_k2 == i)
    print(f"Cluster {i+1}: {count} points")

print("\nCluster statistics for k=3:")
for i in range(3):
    count = np.sum(clusters_k3 == i)
    print(f"Cluster {i+1}: {count} points")

files.download('kmeans_k2.png')
files.download('kmeans_k3.png')
