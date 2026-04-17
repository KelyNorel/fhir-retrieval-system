"""
CLI entry point for building the document store and running evaluation.

Usage:
    python main.py build        # Build SQLite document store from NDJSON files
    python main.py evaluate     # Run all benchmark questions through current search strategy
    python main.py demo         # Run a single demo query
"""

import sys

import pandas as pd

from src.data_loader import load_ground_truth
from src.evaluation import classify_results, retrieval_recall, retrieval_precision
from src.search import search_vector


def cmd_build():
    from src.store import build_store
    build_store()


def cmd_evaluate():
    gt = load_ground_truth()
    print(f"Loaded {len(gt)} benchmark questions.\n")

    rows = []
    for _, row in gt.iterrows():
        question = row["question"]
        true_ids_by_type = row["true_fhir_ids"]
        if not isinstance(true_ids_by_type, dict) or not true_ids_by_type:
            continue

        all_true = sum(true_ids_by_type.values(), [])
        result = search_vector(question, n_results=10)
        retrieved = result["retrieved_ids"]

        rows.append(
            {
                "question_id": row["question_id"],
                "recall": retrieval_recall(retrieved, all_true),
                "precision": retrieval_precision(retrieved, all_true),
                "latency_s": result["latency_s"],
                "n_retrieved": len(retrieved),
                "n_true": len(all_true),
            }
        )

    df = pd.DataFrame(rows)
    print(df.describe().round(4))
    print(f"\nMean Recall:    {df['recall'].mean():.4f}")
    print(f"Mean Precision: {df['precision'].mean():.4f}")
    print(f"P50 Latency:    {df['latency_s'].quantile(0.5):.3f} s")
    print(f"P95 Latency:    {df['latency_s'].quantile(0.95):.3f} s")


def cmd_demo():
    query = "What is the sex of patient 10006277?"
    print(f"Query: {query}\n")

    result = search_vector(query, n_results=5)
    print(f"Strategy: {result['strategy']}")
    print(f"Latency:  {result['latency_s']:.3f} s")
    print(f"Results:")
    for rid in result["retrieved_ids"]:
        print(f"  - {rid}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        cmd_build()
    elif cmd == "evaluate":
        cmd_evaluate()
    elif cmd == "demo":
        cmd_demo()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
