# faiss_index.py

import logging
import faiss
import numpy as np
import pandas as pd
import time
from typing import List
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_faiss_index(embeddings: np.ndarray, d: int, m: int = 32) -> faiss.IndexHNSWFlat:
    """
    Builds a Faiss index.

    Args:
        embeddings (np.ndarray): Embeddings to index.
        d (int): Dimension of embeddings.
        m (int, optional): Number of connections for HNSW. Defaults to 32.

    Returns:
        faiss.IndexHNSWFlat: Built index.
    """
    logger.info("Building Faiss index...")
    faiss.omp_set_num_threads(1)
    index = faiss.IndexHNSWFlat(d, m)
    index.hnsw.efConstruction = 200
    index.verbose = True
    index.add(embeddings)
    logger.info(f"Faiss index contains {index.ntotal} vectors.")
    return index

def evaluate_faiss_index(index: faiss.IndexHNSWFlat, query_embeddings: np.ndarray, baseline: np.ndarray, efSearch_values: List[int]) -> pd.DataFrame:
    """
    Evaluates the Faiss index performance.

    Args:
        index (faiss.IndexHNSWFlat): Faiss index.
        query_embeddings (np.ndarray): Query embeddings.
        baseline (np.ndarray): Baseline indices.
        efSearch_values (List[int]): List of efSearch values to evaluate.

    Returns:
        pd.DataFrame: Performance metrics.
    """
    hnsw_perf = pd.DataFrame({
        'efSearch': [],
        'qps': [],
        'recall@1': []
    })
    logger.info("Evaluating Faiss index performance...")
    results = []  # Initialize a list to store results
    for efSearch in tqdm(efSearch_values):
        index.hnsw.efSearch = efSearch
        # Query dataset
        t0 = time.time()
        distances, labels = index.search(query_embeddings, k=1)
        # Calculate queries per second (QPS)
        qps = len(query_embeddings) / (time.time() - t0)
        # Calculate recall@1
        recall = np.mean(labels.reshape(-1) == baseline.reshape(-1))
        # Append the results to the list
        results.append({
            'efSearch': efSearch,
            'qps': qps,
            'recall@1': recall
        })
    # Create the DataFrame from the list of results
    hnsw_perf = pd.DataFrame(results)
    logger.info("Evaluation completed.")
    return hnsw_perf
