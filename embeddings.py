# embeddings.py

import logging
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """Returns the available device (GPU or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Loads the embedding model.

    Args:
        model_name (str): Name of the SentenceTransformer model.

    Returns:
        SentenceTransformer: Loaded model.
    """
    device = get_device()
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    return model

def compute_embeddings(text_list: List[str], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """
    Computes embeddings for a list of texts.

    Args:
        text_list (List[str]): List of texts to embed.
        model (SentenceTransformer): Embedding model.
        batch_size (int, optional): Batch size for processing. Defaults to 64.

    Returns:
        np.ndarray: Computed embeddings.
    """
    length = len(text_list)
    dims = model.get_sentence_embedding_dimension()
    embeddings = np.zeros((length, dims), dtype=np.float32)
    logger.info(f"Computing embeddings for {length} texts...")
    for i in tqdm(range(0, length, batch_size)):
        i_end = min(i + batch_size, length)
        batch_embeddings = model.encode(text_list[i:i_end], show_progress_bar=False)
        embeddings[i:i_end] = batch_embeddings
    return embeddings

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalizes embeddings.

    Args:
        embeddings (np.ndarray): Embeddings to normalize.

    Returns:
        np.ndarray: Normalized embeddings.
    """
    logger.info("Normalizing embeddings...")
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
