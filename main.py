# main.py

import logging
import numpy as np
import matplotlib.pyplot as plt
from data_processing import load_and_preprocess_lectures, load_and_preprocess_queries
from embeddings import load_embedding_model, compute_embeddings, normalize_embeddings
from faiss_index import build_faiss_index, evaluate_faiss_index
from quantization import CustomIndexPQ
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load and preprocess data
    lectures_list = load_and_preprocess_lectures(config.LECTURE_FILE)
    queries_list = load_and_preprocess_queries(config.QUERY_FILE)

    # Load embedding model
    model = load_embedding_model(config.EMBEDDING_MODEL_NAME)

    # Compute embeddings
    lecture_embeddings = compute_embeddings(lectures_list, model, batch_size=config.BATCH_SIZE)
    lecture_embeddings = normalize_embeddings(lecture_embeddings)
    query_embeddings = compute_embeddings(queries_list, model, batch_size=config.BATCH_SIZE)
    query_embeddings = normalize_embeddings(query_embeddings)

    # Compute distance matrix and baseline
    logger.info("Computing baseline distances...")
    dist = np.matmul(query_embeddings, lecture_embeddings.T)
    baseline = np.argmax(dist, axis=1).reshape(-1)

    # Build and evaluate Faiss index
    d = lecture_embeddings.shape[1]
    index = build_faiss_index(lecture_embeddings, d)
    hnsw_perf = evaluate_faiss_index(index, query_embeddings, baseline, config.FAISS_EFSEARCH_VALUES)

    # Plot the performance
    logger.info("Plotting performance metrics...")
    plt.figure(figsize=(10, 6))
    plt.plot(hnsw_perf['recall@1'], hnsw_perf['qps'], marker='o')
    plt.xlabel("Recall@1")
    plt.ylabel("Queries per Second (QPS)")
    plt.title("Faiss Index Performance")
    plt.grid(True)
    plt.show()

    # Quantization
    pq_index = CustomIndexPQ(d, config.PQ_M, config.PQ_NBITS, kmeans_max_iter=config.KMEANS_MAX_ITER)
    pq_index.train(lecture_embeddings)
    pq_index.add(lecture_embeddings)
    
    # Example search
    logger.info("Performing example search using PQ index...")
    _, indices = pq_index.search(lecture_embeddings[:3], 3)
    index_to_test = 2
    logger.info(f'Lecture: {lectures_list[index_to_test]}\n')
    for kth_index,example in enumerate([lectures_list[i] for i in indices[index_to_test]]):
        logger.info(f'Similar Lecture {kth_index+1}: {example}\n')

if __name__ == '__main__':
    main()
