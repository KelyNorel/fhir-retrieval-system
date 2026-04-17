"""
Unified evaluation script for all retrieval strategies.
Usage: python evaluate_all.py --strategy vector --n_results 100
"""
import numpy as np
from tqdm import tqdm
import argparse

from src.data_loader import load_ground_truth
from src.evaluation import retrieval_recall, retrieval_precision
from src.search import _build_bm25_index, _build_vector_index, _build_patient_id_map


def flatten_true_ids(true_fhir_ids: dict) -> list[str]:
    result = []
    for rtype, ids in true_fhir_ids.items():
        for fid in ids:
            result.append(fid if "/" in fid else f"{rtype}/{fid}")
    return result


def get_search_fn(strategy: str, n_results: int, **kwargs):
    """Return the search function for the given strategy."""
    if strategy == "bm25":
        from src.search import search_bm25
        return lambda q: search_bm25(q, n_results=n_results)
    elif strategy == "vector":
        from src.search import search_vector
        return lambda q: search_vector(q, n_results=n_results)
    elif strategy == "hybrid":
        from src.search_hybrid import search_hybrid
        alpha = kwargs.get("alpha", 0.5)
        return lambda q: search_hybrid(q, n_results=n_results, alpha=alpha)
    elif strategy == "router":
        from src.search_router import search_router
        return lambda q: search_router(q, n_results=n_results)
    elif strategy == "decomposed":
        from src.search_decomposed import search_decomposed
        return lambda q: search_decomposed(q, n_results=n_results)
    elif strategy == "patient_filtered":
        from src.search_patient_filtered import search_patient_filtered
        return lambda q: search_patient_filtered(q, n_results=n_results)
    elif strategy == "dynamic":
        from src.search_dynamic import search_dynamic
        return lambda q: search_dynamic(q)    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def evaluate(strategy: str, n_results: int = 100, **kwargs):
    print("Loading ground truth...")
    df = load_ground_truth()
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_df = test_df[test_df['true_fhir_ids'].apply(lambda x: len(x) > 0)]
    print(f"Test set: {len(test_df)} questions")

    # Build indexes
    if strategy in ["bm25", "hybrid", "router"]:
        print("Loading BM25 index...")
        _build_bm25_index()
    if strategy in ["vector", "hybrid", "router", "decomposed", "patient_filtered","dynamic"]:
        print("Loading vector index...")
        _build_vector_index()
    _build_patient_id_map()

    search_fn = get_search_fn(strategy, n_results, **kwargs)

    recalls = []
    precisions = []
    latencies = []
    f1s = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row["question"]
        true_ids = flatten_true_ids(row["true_fhir_ids"])

        result = search_fn(question)

        recall = retrieval_recall(result["retrieved_ids"], true_ids)
        precision = retrieval_precision(result["retrieved_ids"], true_ids)
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0

        recalls.append(recall)
        precisions.append(precision)
        latencies.append(result["latency_s"])
        f1s.append(f1)

    mean_r = np.mean(recalls)
    mean_p = np.mean(precisions)
    macro_f1 = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) > 0 else 0.0

    print(f"\n=== Results: {strategy} (n={n_results}) ===")
    print(f"Mean Recall:    {mean_r:.3f}")
    print(f"Mean Precision: {mean_p:.3f}")
    print(f"Macro F1:       {macro_f1:.3f}")
    print(f"Mean F1:        {np.mean(f1s):.3f}")
    print(f"Latency p50:    {np.percentile(latencies, 50)*1000:.1f} ms")
    print(f"Latency p95:    {np.percentile(latencies, 95)*1000:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["bm25", "vector", "hybrid", "router", "decomposed", "patient_filtered","dynamic"])
    parser.add_argument("--n_results", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.strategy, args.n_results, alpha=args.alpha)