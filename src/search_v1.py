"""
Search strategies for FHIR record retrieval.
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


import pickle

# ---------------------------------------------------------------------------
# Global indexes — built once, reused across queries
# ---------------------------------------------------------------------------
BM25_INDEX_PATH = "data/bm25_index.pkl"
CHROMA_DIR = "data/chroma_index"

_bm25_index = None
_bm25_ids = None
_patient_id_map: dict[str, str] = {}
_chroma_collection = None



def _build_bm25_index():
    global _bm25_index, _bm25_ids
    from rank_bm25 import BM25Okapi

    # if exists in disk, load it
    if os.path.exists(BM25_INDEX_PATH):
        print("Loading BM25 index from disk...")
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_ids, _bm25_index = pickle.load(f)
        print(f"BM25 index loaded: {len(_bm25_ids):,} documents")
        return

    # if does not exist, build it
    print("Building BM25 index (runs once)...")
    records = load_ndjson_records(FHIR_RECORDS_DIR)

    corpus_ids = []
    tokenized_corpus = []


    for rec in records:
        rtype = rec.get("resourceType", "Unknown")
        rid = rec.get("id", "")
        fhir_id = f"{rtype}/{rid}"
        text = fhir_resource_to_searchable_text(rec)
        
        tokens = text.lower().split()
        corpus_ids.append(fhir_id)
        tokenized_corpus.append(tokens)



    _bm25_ids = corpus_ids
    _bm25_index = BM25Okapi(tokenized_corpus)

    # save to disk
    print("Saving BM25 index to disk...")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump((_bm25_ids, _bm25_index), f)
    print(f"BM25 index built and saved: {len(_bm25_ids):,} documents")





def _build_patient_id_map():
    """Build lookup table {clinical_id: uuid} from Patient.ndjson"""
    global _patient_id_map
    patient_file = os.path.join(FHIR_RECORDS_DIR, "Patient.ndjson")
    with open(patient_file) as f:
        for line in f:
            patient = json.loads(line)
            uuid = patient.get("id", "")
            for identifier in patient.get("identifier", []):
                clinical_id = identifier.get("value", "")
                if clinical_id:
                    _patient_id_map[clinical_id] = uuid
    print(f"Patient ID map built: {len(_patient_id_map)} patients")


def expand_query(query: str) -> str:
    """Replace clinical patient IDs with FHIR UUIDs in the query."""
    expanded = query
    for clinical_id, uuid in _patient_id_map.items():
        if clinical_id in expanded:
            expanded = expanded.replace(clinical_id, uuid)
    return expanded

def expand_query(query: str) -> str:
    """Replace clinical patient IDs with FHIR UUIDs in the query."""
    expanded = query
    for clinical_id, uuid in _patient_id_map.items():
        if clinical_id in expanded:
            expanded = expanded.replace(clinical_id, uuid)
    return expanded    


# ---------------------------------------------------------------------------
# Text representation — smarter than raw JSON dump
# ---------------------------------------------------------------------------

def fhir_resource_to_searchable_text(resource: dict) -> str:
    """Convert a FHIR resource to a clean text string for BM25 indexing.

    Instead of dumping the entire JSON (which includes noise like URLs and
    system codes), we extract the most clinically meaningful fields.
    This is one of the core design decisions in this system.
    """
    parts = []

    rtype = resource.get("resourceType", "")
    rid = resource.get("id", "")
    parts.append(rtype)

    # Patient reference — appears in almost every resource
    subject = resource.get("subject", {}).get("reference", "")
    if subject:
        parts.append(subject)
   
    #adding discharge into index
    if rtype == "Encounter":
        parts.append(resource.get("status", ""))
        for coding in resource.get("type", [{}])[0].get("coding", []):
            parts.append(coding.get("display", ""))
        period = resource.get("period", {})
        parts.append(period.get("start", ""))
        parts.append(period.get("end", ""))
        # Add discharge disposition
        discharge = resource.get("hospitalization", {}).get("dischargeDisposition", {})
        for coding in discharge.get("coding", []):
            parts.append(coding.get("display", ""))
            parts.append("discharge")  # explicit keyword    

    # Patient resource itself
    if rtype == "Patient":
        for name in resource.get("name", []):
            parts.append(name.get("family", ""))
            parts.extend(name.get("given", []))
        parts.append(resource.get("birthDate", ""))
        parts.append(resource.get("gender", ""))

    # Condition
    if rtype == "Condition":
        for coding in resource.get("code", {}).get("coding", []):
            parts.append(coding.get("display", ""))
            parts.append(coding.get("code", ""))
        parts.append(resource.get("onsetDateTime", ""))
        parts.append(resource.get("recordedDate", ""))

        parts.append(resource.get("clinicalStatus", {})
                     .get("coding", [{}])[0].get("code", ""))

    # Observation (lab results, vitals)
    if rtype == "Observation":
        for coding in resource.get("code", {}).get("coding", []):
            parts.append(coding.get("display", ""))
            parts.append(coding.get("code", ""))
        val = resource.get("valueQuantity", {})
        if val:
            parts.append(str(val.get("value", "")))
            parts.append(val.get("unit", ""))
        parts.append(resource.get("effectiveDateTime", ""))

        parts.append(resource.get("issued", ""))

        for cat in resource.get("category", []):
            for coding in cat.get("coding", []):
                parts.append(coding.get("display", ""))

    # MedicationRequest
    if rtype == "MedicationRequest":
        med = resource.get("medicationCodeableConcept", {})
        for coding in med.get("coding", []):
            parts.append(coding.get("display", ""))
            parts.append(coding.get("code", ""))
        parts.append(resource.get("authoredOn", ""))

        parts.append(resource.get("status", ""))

    # Medication
    if rtype == "Medication":
        for coding in resource.get("code", {}).get("coding", []):
            parts.append(coding.get("display", ""))
            parts.append(coding.get("code", ""))

    # Procedure
    if rtype == "Procedure":
        for coding in resource.get("code", {}).get("coding", []):
            parts.append(coding.get("display", ""))
            parts.append(coding.get("code", ""))
        parts.append(resource.get("performedDateTime", ""))
        period = resource.get("performedPeriod", {})
        parts.append(period.get("start", ""))
        parts.append(period.get("end", ""))



    # Encounter
    if rtype == "Encounter":
        parts.append(resource.get("status", ""))
        for coding in resource.get("type", [{}])[0].get("coding", []):
            parts.append(coding.get("display", ""))
        period = resource.get("period", {})
        parts.append(period.get("start", ""))
        parts.append(period.get("end", ""))
 


    # Always include the resource ID (useful for patient ID lookups)
    parts.append(rid)

    # Filter empty strings and join
    return " ".join(p for p in parts if p and isinstance(p, str))


# ---------------------------------------------------------------------------
# Strategy 1: In-Context (default — will crash on full dataset)
# ---------------------------------------------------------------------------

def search_in_context(query: str, provider: str = "openai") -> dict:
    """Attempt to answer a query by stuffing ALL FHIR records into the prompt.
    Intentionally crashes on the full dataset — demonstrates why retrieval matters.
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
# Strategy 2: BM25 keyword search
# ---------------------------------------------------------------------------



def search_bm25(query: str, n_results: int = 20) -> dict:
    """Retrieve FHIR records using BM25 keyword search.

    Design decisions:
    - Uses smarter text representation (fhir_resource_to_searchable_text)
      instead of raw JSON dump — reduces noise, improves precision
    - Index is built once and cached globally for speed
    - Returns top-n results by BM25 score
    """
    global _bm25_index, _bm25_ids

    start = time.time()

    # Build index on first call
    if _bm25_index is None:
        _build_bm25_index()

    # Query expansion — replace clinical IDs with UUIDs
    if not _patient_id_map:
        _build_patient_id_map()
    query = expand_query(query)    

    # Tokenize query the same way as documents
    tokenized_query = query.lower().split()

    # Get BM25 scores for all documents
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-n document indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:n_results]

    # Filter out zero-score results
    retrieved_ids = [
        _bm25_ids[i] for i in top_indices if scores[i] > 0
    ]

    documents = get_documents_by_ids(retrieved_ids)
    elapsed = time.time() - start

    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": "bm25",
    }


# ---------------------------------------------------------------------------
# Strategy 3: Vector Search (stub — to be implemented)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Strategy 3: Vector Search con ChromaDB + OpenAI embeddings
# ---------------------------------------------------------------------------

def extract_patient_uuid_from_record(rec: dict) -> str:
    """Extract patient UUID from a FHIR resource."""
    rtype = rec.get("resourceType", "")
    
    # Patient resource itself
    if rtype == "Patient":
        return rec.get("id", "")
    
    # Other resources reference patient via subject
    subject_ref = rec.get("subject", {}).get("reference", "")
    if subject_ref:
        return subject_ref.replace("Patient/", "")
    
    # Some resources use patient field
    patient_ref = rec.get("patient", {}).get("reference", "")
    if patient_ref:
        return patient_ref.replace("Patient/", "")
    
    return ""


def _build_vector_index():
    """Build ChromaDB vector index from FHIR records. Runs once."""
    global _chroma_collection
    import chromadb
    from tqdm import tqdm

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Si ya existe, cargarlo
    try:
        _chroma_collection = client.get_collection("fhir_resources")
        print(f"Vector index loaded: {_chroma_collection.count()} documents")
        return
    except Exception:
        pass

    print("Building vector index (runs once)...")
    _chroma_collection = client.create_collection(
        name="fhir_resources",
        metadata={"hnsw:space": "cosine"}
    )

    records = load_ndjson_records(FHIR_RECORDS_DIR)
    oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    BATCH_SIZE = 500
    ids_batch = []
    texts_batch = []
    fhir_ids_batch = []
    patient_uuids_batch = [] 

    def flush_batch():
        if not texts_batch:
            return
        response = oai_client.embeddings.create(
            input=texts_batch,
            model="text-embedding-3-small"
        )
        embeddings = [r.embedding for r in response.data]
        _chroma_collection.add(
            ids=ids_batch,
            embeddings=embeddings,
            documents=texts_batch,
            metadatas=[{
                "fhir_id": fid,
                "patient_uuid": puuid
            } for fid, puuid in zip(fhir_ids_batch, patient_uuids_batch)]
         
        )
        ids_batch.clear()
        texts_batch.clear()
        fhir_ids_batch.clear()
        patient_uuids_batch.clear()

    for i, rec in enumerate(tqdm(records, desc="Embedding")):
        rtype = rec.get("resourceType", "Unknown")
        rid = rec.get("id", "")
        fhir_id = f"{rtype}/{rid}"
        text = fhir_resource_to_searchable_text(rec)

        ids_batch.append(str(i))
        texts_batch.append(text[:1000])
        fhir_ids_batch.append(fhir_id)

        patient_uuids_batch.append(extract_patient_uuid_from_record(rec))

        if len(ids_batch) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    print(f"Vector index built: {_chroma_collection.count()} documents")

def detect_resource_types(query: str) -> list[str]:
    """Detect likely FHIR resource types from query keywords."""
    query_lower = query.lower()
    
    type_keywords = {
        "Observation": [
            "lab", "test", "result", "blood", "pressure", "rate", 
            "temperature", "weight", "height", "oxygen", "glucose",
            "sodium", "potassium", "creatinine", "hemoglobin",
            "vital", "measurement", "level", "value", "measured"
        ],
        "MedicationRequest": [
            "prescribed", "prescription", "ordered", "medication",
            "drug", "medicine", "dose", "dosage"
        ],
        "Medication": [
            "medication", "drug", "medicine"
        ],
        "MedicationAdministration": [
            "administered", "given", "received medication", "infusion"
        ],
        "Condition": [
            "diagnosis", "condition", "disease", "disorder",
            "diagnosed", "problem", "complaint"
        ],
        "Procedure": [
            "procedure", "surgery", "operation", "performed",
            "conducted", "treatment", "intervention"
        ],
        "Encounter": [
            "visit", "admission", "admitted", "discharge",
            "hospital", "encounter", "stay", "icu"
        ],
    }
    
    detected = []
    for rtype, keywords in type_keywords.items():
        if any(kw in query_lower for kw in keywords):
            detected.append(rtype)
    
    return detected if detected else []  # empty = no filter    


def search_vector(query: str, n_results: int = 20) -> dict:
    """Retrieve FHIR records via vector similarity search."""
    global _chroma_collection

    start = time.time()

    if _chroma_collection is None:
        _build_vector_index()

     # Query expansion — replace clinical IDs with UUIDs
    if not _patient_id_map:
        _build_patient_id_map()
    query = expand_query(query)    

    oai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    query_embedding = oai_client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    ).data[0].embedding

    # Resource type filtering  ← ACÁ
    detected_types = detect_resource_types(query)
    where_filter = None
    if len(detected_types) == 1:
        where_filter = {"fhir_id": {"$like": f"{detected_types[0]}/%"}}
    elif len(detected_types) > 1:
        where_filter = {"$or": [
            {"fhir_id": {"$like": f"{rt}/%"}} for rt in detected_types
        ]}

    results = _chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    retrieved_ids = [
        meta["fhir_id"]
        for meta in results["metadatas"][0]
    ]

    documents = get_documents_by_ids(retrieved_ids)
    elapsed = time.time() - start

    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": "vector",
    }


# ---------------------------------------------------------------------------
# Strategy 4: Random (demo)
# ---------------------------------------------------------------------------

def search_random(
    query: str,
    n_results: int = 10,
    true_fhir_ids: dict[str, list[str]] | None = None,
) -> dict:
    """Return a random mix of real FHIR resource IDs from the store."""
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
