"""
FHIR Record Search & Retrieval Benchmark — Streamlit UI

Run with:  streamlit run app.py
"""

import ast
import datetime
import math
import os
import re
import time

import pandas as pd
import streamlit as st

from src.store import get_documents_by_ids
from src.config import EVAL_RESULTS_DIR
from src.data_loader import load_ground_truth
from src.evaluation import classify_results
from src.search import search_in_context, search_random, search_vector

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FHIR Retrieval Benchmark",
    page_icon="🏥",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hit-row  { background-color: #d4edda; border-radius: 6px; padding: 6px 10px; margin: 3px 0; font-family: monospace; }
    .miss-row { background-color: #f8d7da; border-radius: 6px; padding: 6px 10px; margin: 3px 0; font-family: monospace; }
    .fp-row   { background-color: #fff3cd; border-radius: 6px; padding: 6px 10px; margin: 3px 0; font-family: monospace; }
    .resource-row { background-color: #e2e3e5; border-radius: 6px; padding: 6px 10px; margin: 3px 0; font-family: monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load ground truth ────────────────────────────────────────────────────────


@st.cache_data
def load_gt():
    try:
        return load_ground_truth()
    except FileNotFoundError:
        return None


gt_df = load_gt()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Settings")

strategy = st.sidebar.radio(
    "Retrieval strategy",
    ["random", "in-context", "vector"],
    index=0,
    help="**Random** returns random real resources to preview the UI.  "
    "**In-context** stuffs everything into one prompt (will crash).  "
    "**Vector** is a stub for your retrieval implementation.",
)

if strategy == "in-context":
    provider = st.sidebar.selectbox("LLM provider", ["openai", "gemini"])
else:
    provider = None

n_results = st.sidebar.number_input("Results to retrieve", min_value=1, max_value=100, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Benchmark data** from "
    "[FHIR-AgentBench](https://github.com/glee4810/FHIR-AgentBench)  \n"
    "Paper: [arXiv 2509.19319](https://arxiv.org/abs/2509.19319)"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_fhir_ids(val) -> dict[str, list[str]]:
    """Parse ground-truth FHIR IDs and qualify bare UUIDs.

    Ground truth stores IDs as bare UUIDs keyed by resource type, e.g.
    {"Observation": ["b2c4828b-..."]}.  The store indexes them as
    "Observation/b2c4828b-...".  This function normalises to the qualified
    format so lookups and comparisons work consistently.
    """
    if isinstance(val, str) and val:
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return {}
    if not isinstance(val, dict):
        return {}

    qualified: dict[str, list[str]] = {}
    for rtype, ids in val.items():
        qualified[rtype] = [
            fid if "/" in fid else f"{rtype}/{fid}" for fid in ids
        ]
    return qualified


def _run_search(query: str, true_fhir_ids=None):
    """Dispatch to the selected strategy and return result dict."""
    if strategy == "random":
        return search_random(query, n_results=n_results, true_fhir_ids=true_fhir_ids)
    elif strategy == "in-context":
        return search_in_context(query, provider=provider)
    else:
        return search_vector(query, n_results=n_results)


def _try_pretty_json(raw: str) -> str:
    """Attempt to pretty-print a JSON string; return as-is on failure."""
    try:
        import json
        return json.dumps(json.loads(raw), indent=2)
    except Exception:
        return raw


_NO_DOC_MSG = (
    "Resource not found in store. "
    "Run `bash scripts/setup_data.sh` or `python main.py build` to build."
)


def _render_retrieved_resources(retrieved_ids: list[str], documents: dict[str, str] | None = None):
    """Display retrieved FHIR resource IDs (no accuracy info)."""
    if not retrieved_ids:
        st.info("No resources retrieved.")
        return
    for rid in retrieved_ids:
        rtype = rid.split("/")[0] if "/" in rid else "Unknown"
        with st.expander(f"{rtype}  —  `{rid}`"):
            doc = (documents or {}).get(rid)
            if doc:
                st.code(_try_pretty_json(doc), language="json")
            else:
                st.caption(_NO_DOC_MSG)


def _render_evaluated_results(
    retrieved_ids: list[str],
    evaluation: dict,
    documents: dict[str, str] | None = None,
):
    """Display results color-coded against ground truth, hits first."""
    st.subheader("Retrieved FHIR Resources")
    if not retrieved_ids:
        st.info("No resources retrieved.")
        return

    hit_ids = {h["id"] for h in evaluation["hits"]}

    # Sort: hits first, then false positives
    sorted_ids = sorted(retrieved_ids, key=lambda rid: (0 if rid in hit_ids else 1, rid))

    for rid in sorted_ids:
        is_hit = rid in hit_ids

        with st.expander(
            f"{'✅' if is_hit else '⚠️'} {rid}{'' if is_hit else '  — not in ground truth'}",
            expanded=False,
        ):
            doc = (documents or {}).get(rid)
            if doc:
                st.code(_try_pretty_json(doc), language="json")
            else:
                st.caption(_NO_DOC_MSG)

    # ── Missed resources ─────────────────────────────────────────────────

    if evaluation["missed"]:
        st.subheader("Missed Resources")
        st.caption("Present in ground truth but not retrieved")
        for m in evaluation["missed"]:
            st.markdown(
                f'<div class="miss-row">&#x274C; <code>{m["id"]}</code> '
                f'({m["resource_type"]})</div>',
                unsafe_allow_html=True,
            )


# ── Tabs ─────────────────────────────────────────────────────────────────────

st.title("FHIR Record Search & Retrieval Benchmark")

tab_bench, tab_search = st.tabs(["Benchmark Evaluation", "Free-Text Search"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Free-Text Search
# ═══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.markdown(
        "Type any clinical question to search the FHIR record store.  "
        "Retrieved resources are displayed below along with query latency.  "
        "No ground-truth scoring is performed here — this is a generic "
        "search and retrieval demo."
    )

    free_query = st.text_area(
        "Clinical question",
        placeholder="e.g. What medications has patient 10006277 been prescribed?",
        height=80,
        key="free_query",
    )

    if st.button("Search", type="primary", key="free_search_btn"):
        if not free_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching…"):
                try:
                    result = _run_search(free_query)
                except Exception as exc:
                    st.error(f"**Search failed:** {exc}")
                    st.info(
                        "If the in-context strategy crashed due to token limits, "
                        "that is expected — the full MIMIC-IV FHIR corpus is too "
                        "large for a single LLM context window.  "
                        "**Your task is to design a better retrieval approach.**"
                    )
                    st.stop()

            retrieved_ids = result["retrieved_ids"]
            documents = result.get("documents", {})
            latency = result["latency_s"]

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Latency", f"{latency:.2f} s")
            col2.metric("Retrieved", len(retrieved_ids))

            st.subheader("Retrieved FHIR Resources")
            _render_retrieved_resources(retrieved_ids, documents)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

with tab_bench:
    if gt_df is None or gt_df.empty:
        st.warning(
            "Ground truth data not found.  Place `ground_truth.csv` in `data/` "
            "— see `data/README.md` for instructions."
        )
        st.stop()

    st.markdown(
        "Select a question from the **FHIR-AgentBench** dataset.  "
        "Retrieved results are scored against the known ground-truth "
        "FHIR resource IDs, showing recall, precision, hits, and misses."
    )

    # ── Filters ──────────────────────────────────────────────────────────

    # Detect available columns
    split_col = None
    for candidate in ("split", "data_split", "db_id"):
        if candidate in gt_df.columns:
            split_col = candidate
            break

    has_template = "template" in gt_df.columns

    # Extract patient IDs from question text (patterns like "patient 10006277")
    _PATIENT_RE = re.compile(r"patient\s+(\d{5,})", re.IGNORECASE)

    @st.cache_data
    def _extract_patient_ids(questions: pd.Series) -> pd.Series:
        def _extract(q):
            m = _PATIENT_RE.search(str(q))
            return m.group(1) if m else None
        return questions.apply(_extract)

    gt_df_local = gt_df.copy()
    gt_df_local["_patient_id"] = _extract_patient_ids(gt_df_local["question"])

    with st.expander("Filters", expanded=True):
        filter_cols = st.columns(3)

        # Split filter
        with filter_cols[0]:
            if split_col:
                available_splits = sorted(
                    gt_df_local[split_col].dropna().unique().tolist()
                )
                selected_split = st.selectbox(
                    "Dataset split",
                    ["all"] + available_splits,
                    index=0,
                    key="split_select",
                )
            else:
                selected_split = "all"
                st.text_input("Dataset split", value="(no split column)", disabled=True)

        # Patient ID filter
        with filter_cols[1]:
            known_patients = sorted(
                gt_df_local["_patient_id"].dropna().unique().tolist()
            )
            selected_patient = st.selectbox(
                "Patient ID",
                ["all"] + known_patients,
                index=0,
                key="patient_select",
            )

        # Template / question type filter
        with filter_cols[2]:
            if has_template:
                known_templates = sorted(
                    gt_df_local["template"].dropna().unique().tolist()
                )
                selected_template = st.selectbox(
                    "Question type (template)",
                    ["all"] + known_templates,
                    index=0,
                    key="template_select",
                )
            else:
                selected_template = "all"
                st.text_input(
                    "Question type (template)",
                    value="(no template column)",
                    disabled=True,
                )

    # Apply filters
    bench_df = gt_df_local.copy()
    if selected_split != "all" and split_col:
        bench_df = bench_df[bench_df[split_col] == selected_split]
    if selected_patient != "all":
        bench_df = bench_df[bench_df["_patient_id"] == selected_patient]
    if selected_template != "all" and has_template:
        bench_df = bench_df[bench_df["template"] == selected_template]

    active_filters = []
    if selected_split != "all":
        active_filters.append(f"split={selected_split}")
    if selected_patient != "all":
        active_filters.append(f"patient={selected_patient}")
    if selected_template != "all":
        active_filters.append(f"template=…{selected_template[-40:]}")

    filter_summary = " · ".join(active_filters) if active_filters else "none"
    st.caption(
        f"**{len(bench_df):,}** questions matched  "
        f"(active filters: {filter_summary})"
    )

    # ── Single-question evaluation ───────────────────────────────────────

    st.markdown("#### Single Question")

    if bench_df.empty:
        st.info("No questions match the current filters.")
    else:
        selected_idx = st.selectbox(
            "Pick a benchmark question",
            bench_df.index,
            format_func=lambda i: (
                f"[{bench_df.loc[i, 'question_id']}] "
                f"{bench_df.loc[i, 'question'][:120]}"
            ),
            key="bench_question_select",
        )

        if st.button("Run Selected Question", type="primary", key="bench_single_btn"):
            row = bench_df.loc[selected_idx]
            true_ids = _parse_fhir_ids(row["true_fhir_ids"])

            with st.spinner("Searching…"):
                try:
                    result = _run_search(row["question"], true_fhir_ids=true_ids)
                except Exception as exc:
                    st.error(f"**Search failed:** {exc}")
                    st.stop()

            retrieved_ids = result["retrieved_ids"]
            documents = result.get("documents", {})
            latency = result["latency_s"]
            evaluation = classify_results(retrieved_ids, true_ids)

            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Latency", f"{latency:.2f} s")
            col2.metric("Retrieved", len(retrieved_ids))
            recall_val = evaluation["recall"]
            prec_val = evaluation["precision"]
            col3.metric("Recall", "N/A" if (isinstance(recall_val, float) and math.isnan(recall_val)) else f"{recall_val:.0%}")
            col4.metric("Precision", "N/A" if (isinstance(prec_val, float) and math.isnan(prec_val)) else f"{prec_val:.0%}")

            _render_evaluated_results(retrieved_ids, evaluation, documents)

            st.markdown("---")
            st.subheader("Ground Truth")
            st.markdown(
                f"**Question ID:** `{row['question_id']}`  \n"
                f"**True Answer:** {row['true_answer']}"
            )

            all_gt_ids = [
                fid for ids in true_ids.values() for fid in ids
            ]
            gt_docs = get_documents_by_ids(all_gt_ids)

            st.markdown(f"**Expected FHIR Resources** ({len(all_gt_ids)}):")
            for rtype, ids in true_ids.items():
                for fid in ids:
                    with st.expander(f"✅ {fid}  ({rtype})"):
                        doc = gt_docs.get(fid)
                        if doc:
                            st.code(_try_pretty_json(doc), language="json")
                        else:
                            st.caption(_NO_DOC_MSG)

    # ── Batch evaluation ─────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("#### Batch Evaluation")

    batch_df = bench_df
    batch_label = f"Run Filtered Set ({len(batch_df):,} questions)"

    st.caption(
        "Runs every question matching the current filters through the "
        "selected retrieval strategy.  Results are displayed below and "
        "automatically saved to `data/eval_results/`."
    )

    if batch_df.empty:
        st.info("No questions match the current filters.")
    elif st.button(batch_label, key="bench_batch_btn"):
        rows_out: list[dict] = []
        progress = st.progress(0, text="Starting batch evaluation…")
        total = len(batch_df)

        for i, (idx, row) in enumerate(batch_df.iterrows()):
            true_ids = _parse_fhir_ids(row["true_fhir_ids"])
            all_true = [fid for ids in true_ids.values() for fid in ids]

            if not all_true:
                continue

            try:
                result = _run_search(row["question"], true_fhir_ids=true_ids)
            except Exception:
                result = {"retrieved_ids": [], "latency_s": 0.0, "strategy": strategy}

            retrieved = result["retrieved_ids"]
            evaluation = classify_results(retrieved, true_ids)

            row_out = {
                "question_id": row["question_id"],
                "question": row["question"],
                "recall": evaluation["recall"],
                "precision": evaluation["precision"],
                "hits": len(evaluation["hits"]),
                "missed": len(evaluation["missed"]),
                "false_pos": len(evaluation["false_positives"]),
                "latency_s": result["latency_s"],
                "strategy": result.get("strategy", strategy),
            }
            if has_template and pd.notnull(row.get("template")):
                row_out["template"] = row["template"]
            if pd.notnull(row.get("_patient_id")):
                row_out["patient_id"] = row["_patient_id"]

            rows_out.append(row_out)

            progress.progress(
                (i + 1) / total,
                text=f"Evaluated {i + 1}/{total} questions…",
            )

        progress.empty()

        if not rows_out:
            st.warning("No evaluable questions found (all had empty ground truth).")
        else:
            results_df = pd.DataFrame(rows_out)

            # ── Aggregate metrics ────────────────────────────────────────

            st.subheader("Aggregate Metrics")
            agg_col1, agg_col2, agg_col3, agg_col4 = st.columns(4)
            mean_recall = results_df["recall"].mean()
            mean_prec = results_df["precision"].mean()
            agg_col1.metric("Mean Recall", f"{mean_recall:.1%}")
            agg_col2.metric("Mean Precision", f"{mean_prec:.1%}")
            p50 = results_df["latency_s"].quantile(0.5)
            p95 = results_df["latency_s"].quantile(0.95)
            agg_col3.metric("P50 Latency", f"{p50:.3f} s")
            agg_col4.metric("P95 Latency", f"{p95:.3f} s")

            # ── Per-question table ───────────────────────────────────────

            st.subheader("Per-Question Results")
            display_cols = [
                c
                for c in [
                    "question_id",
                    "patient_id",
                    "template",
                    "recall",
                    "precision",
                    "hits",
                    "missed",
                    "false_pos",
                    "latency_s",
                ]
                if c in results_df.columns
            ]
            st.dataframe(
                results_df[display_cols].style.format(
                    {
                        "recall": "{:.0%}",
                        "precision": "{:.0%}",
                        "latency_s": "{:.3f}",
                    }
                ),
                use_container_width=True,
                height=400,
            )

            # ── Save results ─────────────────────────────────────────────

            os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            strat_slug = strategy.split("(")[0].strip().replace(" ", "_")
            fname = f"eval_{strat_slug}_{ts}.csv"
            save_path = os.path.join(EVAL_RESULTS_DIR, fname)
            results_df.to_csv(save_path, index=False)

            st.success(f"Results saved to `{save_path}`")

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
            )


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Counsel Health  ·  ML Research Scientist Take-Home  ·  "
    "Data: FHIR-AgentBench (CC-BY-4.0)"
)
