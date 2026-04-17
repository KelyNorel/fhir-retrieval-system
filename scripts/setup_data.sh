#!/usr/bin/env bash
#
# Download FHIR-AgentBench ground-truth data and build the local
# document store.
#
# Usage:
#   bash scripts/setup_data.sh
#
set -euo pipefail

DATA_DIR="data"
FHIR_RECORDS_DIR="$DATA_DIR/fhir_records"
REPO_URL="https://github.com/glee4810/FHIR-AgentBench"

mkdir -p "$DATA_DIR" "$FHIR_RECORDS_DIR"

# ── 1. Clone the benchmark repo (shallow) to grab the ground-truth CSV ──────

BENCH_DIR="$DATA_DIR/_fhir_agentbench"
if [ ! -d "$BENCH_DIR" ]; then
    echo "▸ Cloning FHIR-AgentBench (shallow)…"
    git clone --depth 1 "$REPO_URL" "$BENCH_DIR"
else
    echo "▸ FHIR-AgentBench repo already present, skipping clone."
fi

GT_SRC="$BENCH_DIR/final_dataset/questions_answers_sql_fhir.csv"
if [ -f "$GT_SRC" ]; then
    cp "$GT_SRC" "$DATA_DIR/ground_truth.csv"
    echo "✓ Ground truth copied → $DATA_DIR/ground_truth.csv"
else
    echo "⚠  Ground-truth CSV not found in the cloned repo."
    echo "   You may need to run the FHIR-AgentBench data pipeline first."
    echo "   See: $REPO_URL#data-preparation"
fi

# ── 2. MIMIC-IV FHIR records ────────────────────────────────────────────────

PHYSIONET_URL="https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/"
DOWNLOAD_DIR="$DATA_DIR/_physionet_raw"

if ls "$FHIR_RECORDS_DIR"/*.ndjson 1>/dev/null 2>&1; then
    echo "▸ NDJSON files already present in $FHIR_RECORDS_DIR, skipping download."
else
    echo "▸ Downloading MIMIC-IV FHIR Demo from PhysioNet…"
    mkdir -p "$DOWNLOAD_DIR"
    wget -r -N -c -np -P "$DOWNLOAD_DIR" "$PHYSIONET_URL"

    echo "▸ Decompressing .ndjson.gz files…"
    NDJSON_SRC=$(find "$DOWNLOAD_DIR" -name "*.ndjson.gz" -print -quit | xargs -I{} dirname {})
    if [ -z "$NDJSON_SRC" ]; then
        NDJSON_SRC=$(find "$DOWNLOAD_DIR" -name "*.ndjson" -print -quit | xargs -I{} dirname {})
    fi

    if [ -n "$NDJSON_SRC" ]; then
        gunzip -k "$NDJSON_SRC"/*.ndjson.gz 2>/dev/null || true

        echo "▸ Copying and renaming NDJSON files (stripping Mimic prefix)…"
        for f in "$NDJSON_SRC"/*.ndjson; do
            basename=$(basename "$f")
            renamed=${basename#Mimic}
            cp "$f" "$FHIR_RECORDS_DIR/$renamed"
        done
        echo "✓ NDJSON files extracted → $FHIR_RECORDS_DIR/"
    else
        echo "⚠  Could not locate .ndjson or .ndjson.gz files in the download."
        echo "   Check $DOWNLOAD_DIR and move .ndjson files to $FHIR_RECORDS_DIR/ manually."
    fi
fi

# ── 3. Build document store (fast key-value lookup, no embeddings) ───────

STORE_PATH="$DATA_DIR/fhir_store.db"

if [ -f "$STORE_PATH" ]; then
    echo "▸ Document store already exists at $STORE_PATH, skipping."
elif ls "$FHIR_RECORDS_DIR"/*.ndjson 1>/dev/null 2>&1; then
    echo ""
    echo "▸ Building document store from NDJSON files…"
    python -c "
from src.store import build_store
build_store()
"
else
    echo ""
    echo "▸ No .ndjson files found in $FHIR_RECORDS_DIR — skipping store build."
    echo "  Place your NDJSON files there and re-run this script."
fi

echo ""
echo "Setup complete."
