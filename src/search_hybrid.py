"""
Hybrid search: BM25 + Vector search combined.
"""
import time
import numpy as np
from src.search import (
    search_bm25, search_vector, 
    _build_bm25_index, _build_vector_index
)
from src.store import get_documents_by_ids


def search_hybrid(query: str, n_results: int = 50, alpha: float = 0.5) -> dict:
    """
    Combine BM25 and vector search scores.
    alpha=1.0 → pure BM25
    alpha=0.0 → pure vector
    alpha=0.5 → equal weight
    """
    start = time.time()

    # Get more candidates from each method
    k = n_results * 3

    bm25_result = search_bm25(query, n_results=k)
    vector_result = search_vector(query, n_results=k)

    # Score normalization — BM25 scores need to be normalized to [0,1]
    # We use rank-based scoring: rank 1 = highest score
    def rank_scores(ids: list[str], k: int) -> dict[str, float]:
        return {fid: (k - rank) / k for rank, fid in enumerate(ids)}

    bm25_scores = rank_scores(bm25_result["retrieved_ids"], k)
    vector_scores = rank_scores(vector_result["retrieved_ids"], k)

    # Combine scores
    all_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
    combined = {
        fid: alpha * bm25_scores.get(fid, 0.0) + 
             (1 - alpha) * vector_scores.get(fid, 0.0)
        for fid in all_ids
    }

    # Sort by combined score and take top n_results
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    retrieved_ids = [fid for fid, _ in ranked[:n_results]]

    documents = get_documents_by_ids(retrieved_ids)
    elapsed = time.time() - start

    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": f"hybrid_alpha{alpha}",
    }