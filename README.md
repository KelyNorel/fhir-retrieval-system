# Counsel Health — ML Research Scientist Take-Home

## Overview

**Goal:** Design and implement a scalable search-and-retrieval system for FHIR clinical records, then benchmark it against the [FHIR-AgentBench](https://arxiv.org/abs/2509.19319) evaluation suite (2,931 real-world clinical questions grounded in the HL7 FHIR standard).

You have **3 days** to complete this assessment. We are looking for thoughtful experiment design, measurable retrieval quality, and clear communication of trade-offs, not a production system. Feel free to use any AI coding tools / additional Python packages that you would like.

---

## The Problem

We provide a way to download a local copy of the **MIMIC-IV FHIR Demo** dataset (Patient, Condition, Observation, MedicationRequest, Encounter, Procedure, etc. resources) alongside the FHIR-AgentBench ground truth which maps each clinical question to the correct FHIR resource IDs.

A **random** aproach is included that just loads in a random subset of records that are part of the correct response. There's also sample code for an example **in-context** approach that tries to load the _entire_ dataset into an LLM context window which will (intentionally crash), and stubbed code for the beginning of **vector** search. These are just examples of how you could think about search and retrieval (you do NOT have to implement them)

---

## What We Provide

| Component      | Location                | Purpose                                                                                    |
| -------------- | ----------------------- | ------------------------------------------------------------------------------------------ |
| Streamlit UI   | `app.py`                | Search box, latency display, accuracy scoring, missed-result panel                         |
| Data loader    | `src/data_loader.py`    | NDJSON parsing, FHIR-to-text flattening                                                    |
| Document store | `src/store.py`          | SQLite key-value lookup for raw FHIR JSON by resource ID                                   |
| Evaluation     | `src/evaluation.py`     | Recall, precision, hit/miss classification against ground truth (you can add more metrics) |
| Search stubs   | `src/search.py`         | In-context (crashes), vector (stub), and random (demo) strategies                          |
| Setup script   | `scripts/setup_data.sh` | Downloads ground truth & FHIR data, builds document store                                  |
| API keys       | `.env` (you create)     | Sandbox OpenAI & Gemini keys provided via email                                            |

---

## Setup

```bash
# 1. Create environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with the keys provided in the assignment email

# 3. Download data & build index
bash scripts/setup_data.sh
# Follow the prompts to place MIMIC-IV FHIR NDJSON files in data/fhir_records/

# 4. Launch the UI
streamlit run app.py
```

---

## Your Task

### 1. Play around with the Streamlit UI

Run the app with the **random** strategy selected. Pick a benchmark question from the **test** set, and play around with the outputs. Your evaluations should run against something from the **test** set (note the buttons for bulk-running particular metrics for a particular strategy and saving them to the `/data/eval_results/` folder).

### 2. Design experiments

Design a set of experiments that systematically evaluate different retrieval strategies. Consider dimensions such as:

- **Document representation** — How should FHIR resources be flattened for embedding? How does raw JSON do vs. structured templates vs concatenating key fields?
- **Chunking & indexing** — One embedding per resource? Per-field? Hierarchical?
- **Query processing** — Direct embedding? Query expansion? Decomposition of complex questions?
- **Retrieval method** — Pure vector search? Keyword (BM25) hybrid? Re-ranking with a cross-encoder? Graph-aware traversal of FHIR references?
- **LLM integration** — When (if at all) should an LLM be in the retrieval loop vs. used only for final answer synthesis?

### 3. Implement & measure

Implement at least **two** meaningfully different retrieval strategies and compare them on some of the following metrics:

| Metric                  | Source                                                      |
| ----------------------- | ----------------------------------------------------------- |
| **Retrieval Recall**    | Fraction of ground-truth FHIR IDs in the retrieved set      |
| **Retrieval Precision** | Fraction of retrieved FHIR IDs that are in the ground truth |
| **Latency (p50 / p95)** | Wall-clock time per query                                   |
| **Scalability**         | How does performance change as dataset size grows?          |

The evaluation helpers in `src/evaluation.py` mirror the metrics from the FHIR-AgentBench paper, feel free to add more if you would like.

### 4. Write up your findings

Prepare and share a 3-4 slide Google Doc presentation covering:

- Your retrieval architecture and why you chose it
- Experiment results (tables / charts)
- What worked, what didn't, and what you would try next with more time

---

## Evaluation Rubric

We do **not** expect a perfect system, or the code behind it. We value thoughtful analysis of trade-offs and honest reporting of what works and what doesn't.

---

## Deliverables

Clone this repo (or copy to a private repo) and share access with your interviewer. Include:

1. Your source code
2. Your short slide deck (share via email)
3. Any other scripts needed to reproduce your results

---

## References

- **FHIR-AgentBench** — Lee et al., 2025. [arXiv:2509.19319](https://arxiv.org/abs/2509.19319) · [GitHub](https://github.com/glee4810/FHIR-AgentBench)
- **MIMIC-IV FHIR Demo** — [PhysioNet](https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/)
- **HL7 FHIR R4** — [hl7.org/fhir](https://hl7.org/fhir/)
