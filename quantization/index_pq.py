# quantization/index_pq.py

import logging
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from .kmeans import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomIndexPQ:
    """Custom Product Quantization (PQ) index implementation."""
    def __init__(self, d: int, m: int, nbits: int, kmeans_max_iter: int = 100):
        """
        Args:
            d (int): Dimensionality of the original vectors.
            m (int): Number of segments.
            nbits (int): Number of bits per segment.
            kmeans_max_iter (int, optional): Max iterations for KMeans. Defaults to 100.
        """
        if d % m != 0:
            raise ValueError("d needs to be a multiple of m")
        self.m = m
        self.k = 2 ** nbits
        self.d = d
        self.ds = d // m
        self.kmeans_max_iter = kmeans_max_iter
        self.estimators = [KMeans(k=self.k, max_iter=self.kmeans_max_iter) for _ in range(m)]
        self.is_trained = False
        self.dtype = np.uint8
        self.dtype_orig = np.float32
        self.codes = None

    def train(self, X: np.ndarray):
        """
        Trains all KMeans estimators.

        Args:
            X (np.ndarray): Training data.
        """
        if self.is_trained:
            raise ValueError("Training multiple times is not allowed")
        logger.info("Training Product Quantizer...")
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            logger.info(f"Fitting KMeans for the {i}-th segment...")
            estimator.fit(X_i)
        self.is_trained = True
        logger.info("Product Quantizer training completed.")

    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encodes original features into codes.

        Args:
            X (np.ndarray): Data to encode.

        Returns:
            np.ndarray: Encoded data.
        """
        n = len(X)
        result = np.empty((n, self.m), dtype=self.dtype)
        for i in range(self.m):
            estimator = self.estimators[i]
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            result[:, i] = [estimator.predict(x) for x in X_i]
        return result

    def add(self, X: np.ndarray):
        """
        Adds vectors to the database (their encoded versions).

        Args:
            X (np.ndarray): Data to add.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        logger.info("Encoding and adding vectors to the PQ index...")
        self.codes = self.encode(X)
        logger.info(f"Added {len(self.codes)} vectors to the PQ index.")

    def compute_asymmetric_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Computes asymmetric distances to all database codes.

        Args:
            X (np.ndarray): Query data.

        Returns:
            np.ndarray: Distance matrix.
        """
        if not self.is_trained:
            raise ValueError("The quantizer needs to be trained first.")
        if self.codes is None:
            raise ValueError("No codes detected. You need to run `add` first.")
        n_queries = len(X)
        n_codes = len(self.codes)
        distance_table = np.empty((n_queries, self.m, self.k), dtype=self.dtype_orig)
        for i in range(self.m):
            X_i = X[:, i * self.ds : (i + 1) * self.ds]
            centers = np.array(self.estimators[i].get_cluster_centers())
            distance_table[:, i, :] = euclidean_distances(X_i, centers, squared=True)
        distances = np.zeros((n_queries, n_codes), dtype=self.dtype_orig)
        for i in range(self.m):
            distances += distance_table[:, i, self.codes[:, i]]
        return distances

    def search(self, X: np.ndarray, k: int):
        """
        Finds k closest database codes to given queries.

        Args:
            X (np.ndarray): Query data.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Distances and indices of nearest neighbors.
        """
        distances_all = self.compute_asymmetric_distances(X)
        indices = np.argsort(distances_all, axis=1)[:, :k]
        distances = np.take_along_axis(distances_all, indices, axis=1)
        sorted_indices = np.argsort(distances, axis=1)
        distances = np.take_along_axis(distances, sorted_indices, axis=1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        return distances, indices
