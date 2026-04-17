"""
Evaluate Router retrieval against the FHIR-AgentBench test set.
"""
import numpy as np
from tqdm import tqdm
import argparse

from src.data_loader import load_ground_truth
from src.evaluation import retrieval_recall, retrieval_precision
from src.search import _build_bm25_index, _build_vector_index
from src.search_router import search_router, route_query


def flatten_true_ids(true_fhir_ids: dict) -> list[str]:
    result = []
    for rtype, ids in true_fhir_ids.items():
        for fid in ids:
            if "/" in fid:
                result.append(fid)
            else:
                result.append(f"{rtype}/{fid}")
    return result


def evaluate_router(n_results: int = 100):
    print("Loading ground truth...")
    df = load_ground_truth()
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_df = test_df[test_df['true_fhir_ids'].apply(lambda x: len(x) > 0)]
    print(f"Test set: {len(test_df)} questions")

    # Build all indexes once
    print("Loading BM25 index...")
    _build_bm25_index()
    print("Loading vector index...")
    _build_vector_index()

    recalls = []
    precisions = []
    latencies = []
    f1s = []
    strategy_counts = {"bm25": 0, "vector": 0, "hybrid": 0}

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row["question"]
        true_ids = flatten_true_ids(row["true_fhir_ids"])

        # Track which strategy was selected
        strategy = route_query(question)
        strategy_counts[strategy] += 1

        result = search_router(question, n_results=n_results)

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

    print(f"\n=== Router Results (test set, n={n_results}) ===")
    print(f"Mean Recall:    {mean_r:.3f}")
    print(f"Mean Precision: {mean_p:.3f}")
    print(f"Macro F1:       {macro_f1:.3f}")
    print(f"Mean F1:        {np.mean(f1s):.3f}")
    print(f"Latency p50:    {np.percentile(latencies, 50)*1000:.1f} ms")
    print(f"Latency p95:    {np.percentile(latencies, 95)*1000:.1f} ms")
    print(f"\nStrategy distribution: {strategy_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_results", type=int, default=100)
    args = parser.parse_args()
    evaluate_router(n_results=args.n_results)