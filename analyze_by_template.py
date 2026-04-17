"""
Analyze retrieval performance by question template type.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import load_ground_truth
from src.evaluation import retrieval_recall, retrieval_precision

from src.search import search_bm25, _build_bm25_index
from src.search_patient_filtered import search_patient_filtered
from src.search import _build_vector_index




def flatten_true_ids(true_fhir_ids: dict) -> list[str]:
    result = []
    for rtype, ids in true_fhir_ids.items():
        for fid in ids:
            if "/" in fid:
                result.append(fid)
            else:
                result.append(f"{rtype}/{fid}")
    return result


def analyze():
    from dotenv import load_dotenv
    load_dotenv()

    df = load_ground_truth()
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    test_df = test_df[test_df['true_fhir_ids'].apply(lambda x: len(x) > 0)]

    _build_vector_index()
    ##_build_bm25_index()

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        true_ids = flatten_true_ids(row["true_fhir_ids"])
        ##result = search_vector(row["question"], n_results=100)
        ##result = search_bm25(row["question"], n_results=100)
        result = search_patient_filtered(row["question"], n_results=100)

        recall = retrieval_recall(result["retrieved_ids"], true_ids)
        
        # Simplify template name
        template = row["template"][:60] + "..." if len(row["template"]) > 60 else row["template"]
        
        results.append({
            "template": template,
            "recall": recall,
            "n_true_ids": len(true_ids)
        })

    results_df = pd.DataFrame(results)
    
    summary = results_df.groupby("template").agg(
        mean_recall=("recall", "mean"),
        n_questions=("recall", "count"),
        mean_n_true_ids=("n_true_ids", "mean")
    ).sort_values("mean_recall", ascending=False)

    print("\n=== Recall by Question Type ===")
    print(summary.to_string())
    
    print("\n=== Top 5 best handled ===")
    print(summary.head(5).to_string())
    
    print("\n=== Top 5 worst handled ===")
    print(summary.tail(5).to_string())


if __name__ == "__main__":
    analyze()