# data_processing.py

import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_lectures(lecture_file: str) -> list:
    """
    Loads and preprocesses lecture data.

    Args:
        lecture_file (str): Path to the lecture CSV file.

    Returns:
        list: Preprocessed list of lecture texts.
    """
    logger.info("Loading lecture dataset...")
    lectures_ds = load_dataset("csv", data_files=lecture_file, split="train")

    def concatenate_text(examples):
        return {
            "text": examples["course_title"] + " \n " + examples["lecture_title"]
        }

    lectures = (
        lectures_ds
        .remove_columns(["course_id"])
        .filter(lambda x: len(x["lecture_title"].split()) > 1 and len(x["course_title"].split()) > 1)
        .map(concatenate_text)
    )
    lectures_list = list(set(lectures['text']))
    logger.info(f"Loaded {len(lectures_list)} unique lectures.")
    return lectures_list

def load_and_preprocess_queries(query_file: str) -> list:
    """
    Loads and preprocesses query data.

    Args:
        query_file (str): Path to the query CSV file.

    Returns:
        list: Preprocessed list of query texts.
    """
    logger.info("Loading query dataset...")
    queries_ds = load_dataset("csv", data_files=query_file, split="train")
    queries_list = list(set(queries_ds['search_phrase']))
    logger.info(f"Loaded {len(queries_list)} unique queries.")
    return queries_list
