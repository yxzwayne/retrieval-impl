import os
import re
import argparse
import numpy as np
import jsonlines
import multiprocessing
from functools import partial
from collections import defaultdict


def score_term_doc(q_i, doc, vocab, N, k1, b, avg_field_len):
    """
    Calculates bm25 score for each term-document pair, returns the ranked indices used to retrieve documents in descending order.
    """
    n_q_i = vocab[q_i] if q_i in vocab else 0
    idf = np.log(1 + (N - n_q_i + 0.5) / (n_q_i + 0.5))
    f_q_i_d = doc.lower().count(q_i.lower())
    score = idf * (
        f_q_i_d * (k1 + 1) / (f_q_i_d + k1 * (1 - b + b * (len(doc) / avg_field_len)))
    )
    return score


def parallel_score(q_i, docs, vocab, N, k1, b, avg_field_len):
    """
    Parallelizes the bm25 score calculation for each term-document pair.
    """
    score_func = partial(
        score_term_doc, q_i, vocab=vocab, N=N, k1=k1, b=b, avg_field_len=avg_field_len
    )
    with multiprocessing.Pool() as pool:
        scores = pool.map(score_func, docs)
    return scores


def bm25_scoring(
    query: str,
    field_data: list,
    vocab: dict,
    N: int,
    avg_field_len: float,
    k1: float = 1.2,
    b: float = 0.75,
) -> list:
    """
    Calculates bm25 score for each document in the field, returns the ranked indices used to retrieve documents in descending order.
    Parameters:
        query: str, the query string
        field_data: list, the list of documents
        vocab: dict, the vocabulary
        N: int, the number of documents
        k1: float, the k1 parameter
        b: float, the b parameter
        avg_field_len: float, the average length of the documents
    Returns:
        indices: list, the indices of the top-k documents
    """
    total_scores = np.zeros(len(field_data))
    for q_i in query.split():
        scores = parallel_score(q_i, field_data, vocab, N, k1, b, avg_field_len)
        total_scores += np.array(scores)
    return np.argsort(total_scores)[
        ::-1
    ]  # the [::-1] is to sort the scores in descending order


if __name__ == "__main__":
    default_query = (
        "after the high school entrance examination, things start to get messy"
    )

    parser = argparse.ArgumentParser(description="BM25 search on IODIS dataset")
    parser.add_argument(
        "query",
        nargs="?",
        default=default_query,
        help=f"Search query (default: '{default_query}')",
    )
    args = parser.parse_args()

    query = args.query
    print(f"Search query: {query}")

    data_url = (
        "https://raw.githubusercontent.com/yxzwayne/iodis-data-rsch/main/iodis.jsonl"
    )
    chunks_url = "https://raw.githubusercontent.com/yxzwayne/iodis-data-rsch/main/iodis_chunks.jsonl"

    # Create the 'data' folder if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/iodis.jsonl"):
        os.system(f"wget {data_url}")

    if not os.path.exists("data/iodis_chunks.jsonl"):
        os.system(f"wget {chunks_url}")

    ids, chunks = [], []

    with jsonlines.open("data/iodis_chunks.jsonl") as reader:
        for obj in reader:
            ids.append(obj["video_id"])
            chunks.append(obj["chunk"])

    assert len(ids) == len(chunks)

    # designate field of interest
    field = chunks

    avg_field_len = sum(len(doc) for doc in field) / len(field)

    N = len(field)
    k1 = 1.2
    b = 0.75

    print(f"the number of documents is {N}")

    vocab = defaultdict(int)

    for doc in field:
        words = set(re.findall(r"\w+", doc.lower()))
        for word in words:
            vocab[word] += 1

    vocab = dict(vocab)

    print("the vocabulary size is", len(vocab))

    total_scores = np.zeros(len(field))

    for q_i in query.split():
        scores = parallel_score(q_i, field, vocab, N, k1, b, avg_field_len)
        total_scores += np.array(scores)

    # Get the indices of the top-k documents
    k = 5  # Number of top documents to retrieve
    top_k_indices = np.argsort(total_scores)[::-1][:k]

    # Print the top-k documents and their scores
    print("Top {} documents:".format(k))
    for i, idx in enumerate(top_k_indices):
        print(f"{i+1}. Document {idx}: Score {total_scores[idx]}")
        print(f"   Youtube link: https://www.youtube.com/watch?v={ids[idx]}")
        print(f"   Preview: {field[idx]}")
