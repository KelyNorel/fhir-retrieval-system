"""
Search strategies for FHIR record retrieval.

DEFAULT STRATEGY (in-context):
    Load the ENTIRE dataset into the LLM context window and ask it to find
    the relevant FHIR resources.  This is intentionally naive — it will
    exceed token limits and crash on the full dataset.

    Your task: replace or augment this with a scalable retrieval approach.
"""

import json
import os
import random
import time

import openai
import google.generativeai as genai

from src.config import OPENAI_API_KEY, GEMINI_API_KEY, FHIR_RECORDS_DIR
from src.data_loader import load_ndjson_records, fhir_resource_to_text
from src.store import get_documents_by_ids, get_random_ids


# ---------------------------------------------------------------------------
# Strategy 1: In-Context (default — will crash on full dataset)
# ---------------------------------------------------------------------------


def search_in_context(query: str, provider: str = "openai") -> dict:
    """Attempt to answer a query by stuffing ALL FHIR records into the prompt.

    This is the baseline approach provided for demonstration.  It works for
    tiny toy datasets but will fail on the full MIMIC-IV FHIR corpus due to
    context-window limits.
    """
    start = time.time()

    records = load_ndjson_records(FHIR_RECORDS_DIR)
    corpus = "\n\n".join(fhir_resource_to_text(r) for r in records)

    prompt = (
        "You are a clinical data assistant. Below is the COMPLETE set of "
        "FHIR resources from a patient database. Answer the user's question "
        "by identifying the relevant FHIR resource IDs.\n\n"
        "=== FHIR DATA START ===\n"
        f"{corpus}\n"
        "=== FHIR DATA END ===\n\n"
        f"Question: {query}\n\n"
        "Return ONLY a JSON list of the relevant FHIR resource IDs "
        '(e.g., ["Patient/12345", "Condition/67890"]).'
    )

    retrieved_ids: list[str] = []

    if provider == "openai":
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or "[]"
    elif provider == "gemini":
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text or "[]"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    try:
        retrieved_ids = json.loads(raw)
        if not isinstance(retrieved_ids, list):
            retrieved_ids = []
    except json.JSONDecodeError:
        retrieved_ids = []

    elapsed = time.time() - start
    return {
        "retrieved_ids": retrieved_ids,
        "documents": {},
        "latency_s": elapsed,
        "strategy": "in-context",
    }


# ---------------------------------------------------------------------------
# Strategy 2: Vector Search (starter — build your own!)
# ---------------------------------------------------------------------------


def search_vector(query: str, n_results: int = 10) -> dict:
    """Retrieve FHIR records via vector similarity.

    This is a stub — implement your own retrieval approach here.

    Ideas to explore:
      - Better document representations (not raw JSON dumps)
      - Hybrid search (keyword + vector)
      - Re-ranking with a cross-encoder
      - Graph-aware traversal of FHIR references
      - Query decomposition / expansion
    """
    start = time.time()

    # TODO: Replace with your retrieval implementation.
    retrieved_ids: list[str] = []

    documents = get_documents_by_ids(retrieved_ids)

    elapsed = time.time() - start
    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": "vector",
    }


# ---------------------------------------------------------------------------
# Strategy 3: Random (demo — exercises the UI without data or API keys)
# ---------------------------------------------------------------------------


def search_random(
    query: str,
    n_results: int = 10,
    true_fhir_ids: dict[str, list[str]] | None = None,
) -> dict:
    """Return a random mix of real FHIR resource IDs from the store.

    When ground-truth IDs are provided, the result intentionally includes some
    true positives, some false positives, and omits some true IDs — so the UI
    shows a realistic spread of hits (green), false positives (yellow), and
    misses (red).
    """
    start = time.time()
    retrieved_ids: list[str] = []

    if true_fhir_ids:
        all_true = [fid for ids in true_fhir_ids.values() for fid in ids]

        n_hits = max(1, random.randint(1, len(all_true)))
        retrieved_ids.extend(random.sample(all_true, min(n_hits, len(all_true))))

        n_fp = n_results - len(retrieved_ids)
        if n_fp > 0:
            fp_ids = get_random_ids(n_fp, exclude=set(all_true))
            retrieved_ids.extend(fp_ids)
    else:
        retrieved_ids = get_random_ids(n_results)

    random.shuffle(retrieved_ids)
    elapsed = time.time() - start

    documents = get_documents_by_ids(retrieved_ids)

    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": "random",
    }
