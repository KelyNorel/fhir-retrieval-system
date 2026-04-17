"""
Evaluate Vector Search retrieval against the FHIR-AgentBench test set.
"""
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from src.data_loader import load_ground_truth
from src.evaluation import retrieval_recall, retrieval_precision
from src.search import search_vector, _build_vector_index


def flatten_true_ids(true_fhir_ids: dict) -> list[str]:
    result = []
    for rtype, ids in true_fhir_ids.items():
        for fid in ids:
            if "/" in fid:
                result.append(fid)
            else:
                result.append(f"{rtype}/{fid}")
    return result


def evaluate_vector(n_results: int = 20):
    print("Loading ground truth...")
    df = load_ground_truth()
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_df = test_df[test_df['true_fhir_ids'].apply(lambda x: len(x) > 0)]
    print(f"Test set: {len(test_df)} questions")

    # Build vector index once
    _build_vector_index()

    recalls = []
    precisions = []
    latencies = []
    f1s = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row["question"]
        true_ids = flatten_true_ids(row["true_fhir_ids"])

        result = search_vector(question, n_results=n_results)

        recall = retrieval_recall(result["retrieved_ids"], true_ids)
        precision = retrieval_precision(result["retrieved_ids"], true_ids)

        recalls.append(recall)
        precisions.append(precision)
        latencies.append(result["latency_s"])

        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0  
        f1s.append(f1)  

    mean_p = np.mean(precisions)
    mean_r = np.mean(recalls)
    macro_f1 = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) > 0 else 0.0
    print("\n=== Vector Search Results (test set) ===")
    print(f"Mean Recall:    {mean_r:.3f}")
    print(f"Mean Precision: {mean_p:.3f}")
    print(f"Macro F1:       {macro_f1:.3f}")
    print(f"Mean F1:        {np.mean(f1s):.3f}")    
    print(f"Latency p50:    {np.percentile(latencies, 50)*1000:.1f} ms")
    print(f"Latency p95:    {np.percentile(latencies, 95)*1000:.1f} ms")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_results", type=int, default=20)
    args = parser.parse_args()
    evaluate_vector(n_results=args.n_results)
