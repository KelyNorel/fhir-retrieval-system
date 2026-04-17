"""
Dynamic n_results: select n based on estimated query complexity.
Uses keyword-based heuristics derived from per-template ground-truth statistics.
"""
import time
from src.search_patient_filtered import search_patient_filtered


def estimate_n_results(query: str) -> int:
    """
    Estimate optimal n_results based on query complexity.
    Derived from mean_n_true_ids per template in FHIR-AgentBench analysis.
    These are MIMIC-specific — would need re-tuning for other datasets.
    A more robust approach: confidence-based n (expand until similarity scores drop)
    """
    query_lower = query.lower()

    # Aggregate queries — many true IDs (100-225)
    if any(kw in query_lower for kw in [
        "any lab", "any medication", "any procedure",
        "any diagnosis", "any drug", "any test",
        "been prescribed", "count the number of drug",
        "total volume", "total amount", "output"
    ]):
        return 1200

    # Count queries — medium true IDs (50-130)
    if any(kw in query_lower for kw in [
        "count the number", "how many", "number of times",
        "number of icu", "number of hospital"
    ]):
        return 500

    # Single resource queries — 1-3 true IDs
    if any(kw in query_lower for kw in [
        "gender", "marital", "discharge time",
        "admission type", "careunit", "length of stay",
        "length of icu", "first time", "last time",
        "what was the name", "what was the dose"
    ]):
        return 50

    # Default — medium complexity
    return 300


def search_dynamic(query: str) -> dict:
    """Patient-filtered search with dynamic n_results based on query complexity."""
    n = estimate_n_results(query)
    result = search_patient_filtered(query, n_results=n)
    result["strategy"] = f"dynamic_n{n}"
    result["estimated_n"] = n
    return result