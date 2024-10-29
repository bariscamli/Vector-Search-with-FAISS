# quantization/kmeans.py

import numpy as np

def euclidean_distance(point1, point2):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(point1 - point2, axis=0)

class KMeans:
    """Custom KMeans clustering algorithm."""
    def __init__(self, k=2, tolerance=0.001, max_iter=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iter
        self.centroids = {}

    def fit(self, data):
        """Fits the KMeans algorithm to the data."""
        for i in range(self.k):
            self.centroids[i] = data[i]

        for _ in range(self.max_iterations):
            self.classes = {j: [] for j in range(self.k)}
            for point in data:
                distances = [euclidean_distance(point, self.centroids[centroid]) for centroid in self.centroids]
                cluster_index = distances.index(min(distances))
                self.classes[cluster_index].append(point)
            previous = dict(self.centroids)
            for cluster_index in self.classes:
                self.centroids[cluster_index] = np.average(self.classes[cluster_index], axis=0)
            is_optimal = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr_centroid = self.centroids[centroid]
                if np.sum((curr_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    is_optimal = False
            if is_optimal:
                break

    def predict(self, data_point):
        """Predicts the closest cluster for a data point."""
        distances = [euclidean_distance(data_point, self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def get_cluster_centers(self):
        """Returns cluster centers."""
        return [self.centroids[centroid] for centroid in self.centroids.keys()]
