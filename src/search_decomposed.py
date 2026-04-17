"""
Query decomposition: use LLM to decompose complex clinical questions
into simpler search terms before retrieval.
"""
import json
import time
import openai
from src.config import OPENAI_API_KEY
from src.search import search_vector, _build_patient_id_map, expand_query
from src.store import get_documents_by_ids


def decompose_query(query: str) -> list[str]:
    """Use LLM to decompose a complex clinical question into search terms."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    prompt = """Extract FHIR search terms from this clinical question.
For each concept in the question, generate a search string with the FHIR resource type.
Always include 2-4 search strings. Always include the patient ID in each string.

Examples:
Question: "Has patient 10025463 received any lab test in 11/2136?"
Output: ["Observation laboratory 10025463", "lab test result 10025463", "Patient 10025463"]

Question: "What was the name of the drug prescribed to patient 10018081?"
Output: ["MedicationRequest drug 10018081", "medication prescribed 10018081", "Patient 10018081"]

Question: "Did patient 10029291 undergo a procedure in the first hospital visit?"
Output: ["Procedure 10029291", "Encounter hospital visit 10029291", "Patient 10029291"]

Question: {query}
Output:""".format(query=query)

    response = client.chat.completions.create(
        #model="gpt-4o-mini",
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    raw = response.choices[0].message.content or "[]"
    try:
        terms = json.loads(raw)
        if not isinstance(terms, list):
            return [query]
        return terms
    except json.JSONDecodeError:
        return [query]


def search_decomposed(query: str, n_results: int = 100) -> dict:
    """Decompose query then search with vector search."""
    start = time.time()
    
    # Query expansion
    expanded_query = expand_query(query)
    
    # Decompose into sub-queries
    sub_queries = decompose_query(expanded_query)
    
    # Search each sub-query and union results
    all_ids = {}  # fhir_id -> best rank
    
    for sq in sub_queries:
        sq_expanded = expand_query(sq)
        result = search_vector(sq_expanded, n_results=n_results)
        for rank, fid in enumerate(result["retrieved_ids"]):
            if fid not in all_ids:
                all_ids[fid] = rank
            else:
                all_ids[fid] = min(all_ids[fid], rank)
    
    # Sort by best rank across all sub-queries
    ranked = sorted(all_ids.items(), key=lambda x: x[1])
    retrieved_ids = [fid for fid, _ in ranked[:n_results]]
    
    documents = get_documents_by_ids(retrieved_ids)
    elapsed = time.time() - start
    
    return {
        "retrieved_ids": retrieved_ids,
        "documents": documents,
        "latency_s": elapsed,
        "strategy": "decomposed",
    }