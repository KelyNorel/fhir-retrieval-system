# FHIR Resource Retrieval System
### A multi-strategy approach to clinical question answering

**Benchmarked against FHIR-AgentBench** (2,931 real-world clinical questions) on the MIMIC-IV FHIR Demo dataset (928,935 resources, 100 patients).

---

## The Problem

Retrieve relevant FHIR resources from 928K records in response to natural language clinical questions — evaluated against ground truth FHIR resource IDs.

---

## Strategies Implemented

| Strategy | Description |
|---|---|
| **BM25** | Keyword search with query expansion |
| **Vector Search** | Semantic search with OpenAI embeddings + ChromaDB |
| **Hybrid** | Rank-based combination of BM25 + vector |
| **Router** | Query-type based strategy selection |
| **Patient-Filtered** ⭐ | Two-stage retrieval: filter by patient UUID, then vector search within ~9K resources |
| **Query Decomposition** | LLM extracts FHIR-specific search terms from complex clinical questions |

BM25 and Vector Search are standard techniques; **Hybrid, Router, Patient-Filtered, and Query Decomposition are original contributions**.

---

## Key Results

| Strategy | Recall | Precision | Macro F1 | p50 Latency |
|---|---|---|---|---|
| BM25 | 0.142 | 0.007 | 0.013 | 853ms |
| Vector | 0.219 | 0.013 | 0.022 | 329ms |
| Hybrid | 0.291 | 0.013 | 0.025 | 1349ms |
| Router | 0.289 | 0.012 | 0.023 | 574ms |
| Decomposed | 0.273 | 0.015 | 0.028 | 2347ms |
| **Patient-Filtered** | **0.434** | **0.015** | **0.029** | **513ms** |
| **Patient-Filtered (best, n=1200)** | **0.821** | 0.008 | 0.015 | 1083ms |
| *Paper SOTA (agentic)* | *0.810* | — | — | — |

**Our best system (recall 0.821) matches paper SOTA (0.810) using pure retrieval — no LLM in the retrieval loop.**

---

## Key Design Decisions

- **Patient-level pre-filtering** — reduces search space from 928K → ~9K resources (100x), best recall overall
- **Query expansion** — maps clinical patient IDs to FHIR UUIDs, improves recall across all strategies
- **Smarter text representation** — field selection and truncation tuned by failure analysis
- **Failure-driven iteration** — per-template analysis revealed missing fields and guided improvements

---

## Project Structure

```
src/
├── config.py               # Configuration
├── data_loader.py          # NDJSON parsing, FHIR-to-text flattening
├── store.py                # SQLite key-value store for raw FHIR JSON
├── evaluation.py           # Recall, precision, F1 against ground truth
├── search.py               # Base search strategies
├── search_hybrid.py        # Hybrid BM25 + vector
├── search_router.py        # Query-type router
├── search_decomposed.py    # LLM query decomposition
├── search_patient_filtered.py  # Two-stage patient-filtered retrieval
├── search_dynamic.py       # Dynamic n_results
└── search_CH.py            # Experimental strategies
app.py                      # Streamlit UI
evaluate_all.py             # Full evaluation runner
scripts/setup_data.sh       # Downloads MIMIC-IV FHIR + ground truth
```

---

## Data

This project uses publicly available datasets:
- **MIMIC-IV FHIR Demo** — [PhysioNet](https://physionet.org/content/mimic-iv-fhir-demo/2.0/)
- **FHIR-AgentBench** — [arXiv:2509.19319](https://arxiv.org/abs/2509.19319) · [GitHub](https://github.com/fhir-agentbench/fhir-agentbench)

Data files are not included in this repo. Run `scripts/setup_data.sh` to download and set up.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI API key
bash scripts/setup_data.sh
python evaluate_all.py
```

---

## References

- Lee et al., 2025. *FHIR-AgentBench*. arXiv:2509.19319
- MIMIC-IV FHIR Demo — PhysioNet
- HL7 FHIR R4 — hl7.org/fhir
