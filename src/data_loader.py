"""
Download and prepare the FHIR-AgentBench evaluation dataset.

Ground truth comes from: https://github.com/glee4810/FHIR-AgentBench
FHIR records come from:  MIMIC-IV Clinical Database Demo on FHIR (PhysioNet)
                         https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/
"""

import json
import os
import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import GROUND_TRUTH_PATH, FHIR_RECORDS_DIR


def load_ground_truth() -> pd.DataFrame:
    """Load the ground-truth question/answer/FHIR-ID dataset.

    Expected CSV columns (from FHIR-AgentBench):
        question_id, question, true_answer, true_fhir_ids, template, ...

    The file should be placed at data/ground_truth.csv.  You can obtain it by
    running the FHIR-AgentBench pipeline, or by downloading the pre-built
    release artifact (see data/README.md).
    """
    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(
            f"Ground truth not found at {GROUND_TRUTH_PATH}. "
            "See data/README.md for download instructions."
        )

    df = pd.read_csv(GROUND_TRUTH_PATH)
    required = {"question_id", "question", "true_answer", "true_fhir_ids"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth CSV is missing columns: {missing}")

    import ast

    def _qualify_ids(raw):
        """Parse and qualify bare UUIDs → ResourceType/UUID."""
        if pd.isnull(raw) or raw == "":
            return {}
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
        if not isinstance(parsed, dict):
            return {}
        return {
            rtype: [fid if "/" in fid else f"{rtype}/{fid}" for fid in ids]
            for rtype, ids in parsed.items()
        }

    df["true_fhir_ids"] = df["true_fhir_ids"].apply(_qualify_ids)
    return df


def load_ndjson_records(ndjson_dir: str | None = None) -> list[dict]:
    """Load FHIR NDJSON bundles from disk.

    Each .ndjson file contains one JSON resource per line.  The MIMIC-IV FHIR
    demo dataset ships as a directory of such files (Patient.ndjson,
    Condition.ndjson, etc.).
    """
    ndjson_dir = ndjson_dir or FHIR_RECORDS_DIR
    if not os.path.isdir(ndjson_dir):
        raise FileNotFoundError(
            f"FHIR records directory not found at {ndjson_dir}. "
            "See data/README.md for download instructions."
        )

    records: list[dict] = []
    for fpath in sorted(glob.glob(os.path.join(ndjson_dir, "*.ndjson"))):
        resource_type = Path(fpath).stem
        print(f"  Loading {resource_type}...")
        with open(fpath, "r") as f:
            for line in tqdm(f, desc=resource_type, leave=False):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    print(f"Loaded {len(records):,} FHIR resources from {ndjson_dir}")
    return records


def fhir_resource_to_text(resource: dict) -> str:
    """Flatten a FHIR JSON resource into a plain-text string for embedding.

    This is deliberately simple — improving this function is one of the core
    goals of the assignment.
    """
    return json.dumps(resource, indent=None, default=str)


