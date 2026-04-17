"""
Smart routing: select retrieval strategy based on question type.
Based on empirical analysis of per-template performance.
"""
from src.search import search_bm25, search_vector
from src.search_hybrid import search_hybrid


# Based on template analysis:
# BM25 excels at: procedure, microbiology, gender, specific lab tests
# Vector excels at: discharge time, marital status, emergency, admission
# Hybrid for everything else

BM25_KEYWORDS = [
    "procedure", "microbiology", "gender", "sex",
    "specimen", "organism", "culture", "lab test",
    "received a", "had any"
]

VECTOR_KEYWORDS = [
    "discharge", "marital", "emergency", "admission",
    "careunit", "icu", "hospital stay", "length of stay",
    "weight", "height", "vital", "drug", "dose",
    "medication", "prescribed", "visit", "change in"
]


def route_query(query: str) -> str:
    """Decide which strategy to use based on query content."""
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in BM25_KEYWORDS):
        return "bm25"
    elif any(kw in query_lower for kw in VECTOR_KEYWORDS):
        return "vector"
    else:
        return "hybrid"


def search_router(query: str, n_results: int = 100) -> dict:
    """Route query to best retrieval strategy."""
    strategy = route_query(query)
    
    if strategy == "bm25":
        return search_bm25(query, n_results=n_results)
    elif strategy == "vector":
        return search_vector(query, n_results=n_results)
    else:
        # Hybrid gets more candidates
        return search_hybrid(query, n_results=n_results * 2, alpha=0.5)