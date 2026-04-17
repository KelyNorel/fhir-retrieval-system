"""
Evaluate retrieval results against the FHIR-AgentBench ground truth.

Metrics mirror those used in the original benchmark:
  - Retrieval Recall:    fraction of true FHIR IDs present in retrieved set
  - Retrieval Precision: fraction of retrieved FHIR IDs that are in truth set
"""

from __future__ import annotations


def retrieval_recall(predicted_ids: list[str], true_ids: list[str]) -> float:
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids:
        return float("nan")
    if not predicted_ids:
        return 0.0
    pred_set = set(predicted_ids)
    return sum(1 for tid in true_ids if tid in pred_set) / len(true_ids)


def retrieval_precision(predicted_ids: list[str], true_ids: list[str]) -> float:
    if not true_ids and not predicted_ids:
        return 1.0
    if not true_ids:
        return 0.0
    if not predicted_ids:
        return float("nan")
    true_set = set(true_ids)
    return sum(1 for pid in predicted_ids if pid in true_set) / len(predicted_ids)


def classify_results(
    retrieved_ids: list[str],
    true_ids_by_type: dict[str, list[str]],
) -> dict:
    """Partition retrieved IDs into hits and misses relative to ground truth.

    Returns:
        {
            "hits":   [{"id": ..., "resource_type": ...}, ...],
            "false_positives": [{"id": ..., "resource_type": ...}, ...],
            "missed": [{"id": ..., "resource_type": ...}, ...],
            "recall": float,
            "precision": float,
        }
    """
    all_true: list[str] = []
    true_type_map: dict[str, str] = {}
    for rtype, ids in true_ids_by_type.items():
        for fid in ids:
            all_true.append(fid)
            true_type_map[fid] = rtype

    true_set = set(all_true)
    retrieved_set = set(retrieved_ids)

    hits = [
        {"id": rid, "resource_type": true_type_map.get(rid, "?")}
        for rid in retrieved_ids
        if rid in true_set
    ]
    false_positives = [
        {"id": rid, "resource_type": rid.split("/")[0] if "/" in rid else "?"}
        for rid in retrieved_ids
        if rid not in true_set
    ]
    missed = [
        {"id": tid, "resource_type": true_type_map.get(tid, "?")}
        for tid in all_true
        if tid not in retrieved_set
    ]

    return {
        "hits": hits,
        "false_positives": false_positives,
        "missed": missed,
        "recall": retrieval_recall(retrieved_ids, all_true),
        "precision": retrieval_precision(retrieved_ids, all_true),
    }
