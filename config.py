LECTURE_FILE = 'data/01_course_lecture.csv'
QUERY_FILE = 'data/01_query.csv'
EMBEDDING_MODEL_NAME = 'sentence-transformers/msmarco-distilbert-base-v4'
BATCH_SIZE = 64
FAISS_EFSEARCH_VALUES = [16, 32, 64, 128, 256, 512, 1024]
PQ_M = 8
PQ_NBITS = 8
KMEANS_MAX_ITER = 10 # For faster training in local