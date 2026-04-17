"""
Lightweight document store for FHIR resources.

Stores raw JSON keyed by qualified FHIR ID (e.g. "Patient/abc-123") in a
SQLite database.  No embeddings, no ML models — just fast key-value lookups
so the UI can display resource content.

Candidates should build their own retrieval index (vector, keyword, graph,
etc.) on top of the raw data in data/fhir_records/.
"""

import json
import os
import sqlite3

from src.config import FHIR_STORE_PATH, FHIR_RECORDS_DIR


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(FHIR_STORE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS resources "
        "(id TEXT PRIMARY KEY, resource_type TEXT, doc TEXT)"
    )
    return conn


def build_store() -> None:
    """Scan NDJSON files and build the SQLite lookup store."""
    import glob
    from tqdm import tqdm

    if not os.path.isdir(FHIR_RECORDS_DIR):
        raise FileNotFoundError(f"FHIR records not found at {FHIR_RECORDS_DIR}")

    conn = _get_conn()
    cursor = conn.cursor()
    count = 0

    for fpath in sorted(glob.glob(os.path.join(FHIR_RECORDS_DIR, "*.ndjson"))):
        fname = os.path.basename(fpath)
        print(f"  Indexing {fname}…")
        with open(fpath, "r") as f:
            batch: list[tuple[str, str, str]] = []
            for line in tqdm(f, desc=fname, leave=False):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rtype = rec.get("resourceType", "Unknown")
                rid = rec.get("id", "")
                fhir_id = f"{rtype}/{rid}"
                batch.append((fhir_id, rtype, line))

                if len(batch) >= 5000:
                    cursor.executemany(
                        "INSERT OR REPLACE INTO resources (id, resource_type, doc) VALUES (?, ?, ?)",
                        batch,
                    )
                    count += len(batch)
                    batch.clear()

            if batch:
                cursor.executemany(
                    "INSERT OR REPLACE INTO resources (id, resource_type, doc) VALUES (?, ?, ?)",
                    batch,
                )
                count += len(batch)

    conn.commit()
    conn.close()
    print(f"Indexed {count:,} resources into {FHIR_STORE_PATH}")


def get_random_ids(n: int, exclude: set[str] | None = None) -> list[str]:
    """Return up to n random FHIR resource IDs from the store."""
    if not os.path.exists(FHIR_STORE_PATH):
        return []
    conn = _get_conn()
    if exclude:
        placeholders = ",".join("?" for _ in exclude)
        cursor = conn.execute(
            f"SELECT id FROM resources WHERE id NOT IN ({placeholders}) "
            "ORDER BY RANDOM() LIMIT ?",
            [*exclude, n],
        )
    else:
        cursor = conn.execute(
            "SELECT id FROM resources ORDER BY RANDOM() LIMIT ?", (n,)
        )
    result = [row[0] for row in cursor.fetchall()]
    conn.close()
    return result


def get_documents_by_ids(ids: list[str]) -> dict[str, str]:
    """Look up raw JSON for a list of FHIR resource IDs."""
    if not ids:
        return {}
    if not os.path.exists(FHIR_STORE_PATH):
        return {}

    conn = _get_conn()
    placeholders = ",".join("?" for _ in ids)
    cursor = conn.execute(
        f"SELECT id, doc FROM resources WHERE id IN ({placeholders})", ids
    )
    result = {row[0]: row[1] for row in cursor.fetchall()}
    conn.close()
    return result
