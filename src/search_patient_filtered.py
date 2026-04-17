"""
Patient-filtered search: filter by patient UUID metadata, then search semantically.
Two-stage retrieval: Stage 1 = patient filter via metadata, Stage 2 = vector search.
"""
import re
import time
import openai

from src.config import OPENAI_API_KEY
import src.search as _search_module
from src.search import _build_vector_index, _build_patient_id_map, expand_query
from src.store import get_documents_by_ids


def extract_clinical_patient_id(query: str) -> str:
    """Extract clinical patient ID from query using regex.
    Patient ID extraction is MIMIC-specific (regex pattern 100XXXXX). 
    A production system serving multiple institutions would require institution-aware 
    entity extraction or an LLM-based approach.
    """
    match = re.search(r'\b(100\d{5})\b', query)
    return match.group(1) if match else ""


def search_patient_filtered(query: str, n_results: int = 100) -> dict:
    """
    Two-stage retrieval:
    1. Filter ChromaDB to resources belonging to the patient in the query
    2. Vector search within that subset
    """
    start = time.time()

    if _search_module._chroma_collection is None:
        _build_vector_index()

    if not _search_module._patient_id_map:
        _build_patient_id_map()

    # Extract patient UUID
    clinical_id = extract_clinical_patient_id(query)
    patient_uuid = _search_module._patient_id_map.get(clinical_id, "")

    # Generate query embedding
    oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Expand query AFTER extracting clinical_id — order matters.
    # expand_query replaces clinical ID with UUID in the query string.
    # If called before extract_clinical_patient_id, the regex would fail
    # to find the numeric ID (e.g. "10025463") because it was already
    # replaced by a UUID (e.g. "28776290-4349-56d3-8c13-adc554feabb8").
    expanded_query = expand_query(query)
    query_embedding = oai_client.embeddings.create(
        input=[expanded_query],
        model="text-embedding-3-small"
    ).data[0].embedding

    # Filter by patient UUID using metadata field
    if patient_uuid:
        where_filter = {"patient_uuid": {"$eq": patient_uuid}}  #ChromaDB filter
    else:
        where_filter = None   #if true we look in all the corpus
     Graceful degradation — if no patient ID detected, falls back to full corpus search. 
     #In production, this case should be logged and handled explicitly.  

    results = _search_module._chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter
    )

    retrieved_ids = [
        meta["fhir_id"]
        for meta in results["metadatas"][0]
    ]

    documents = get_documents_by_ids(retrieved_ids)
    elapsed = time.time() - start

    # Return both IDs for debugging and per-template analysis —
    # clinical_id shows what was in the query, patient_uuid shows
    # what was used for filtering.
    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": "patient_filtered",
        "patient_uuid": patient_uuid,
        "clinical_id": clinical_id,
    }