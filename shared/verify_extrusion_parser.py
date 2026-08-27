"""Replay shared/extrusion_parser.py against every real processed extrusion
file and confirm its row counts match what the deployed pipeline actually
recorded in the manifest table.

This is the Phase 2.1 verification step: the shared module is a straight
extraction of the deployed main.py logic, and this script is how that claim
gets checked against real data instead of assumed. Read-only: downloads
files and queries BigQuery, writes nothing.

Usage: python3 shared/verify_extrusion_parser.py
"""

import hashlib
import sys
from pathlib import Path

from google.cloud import bigquery
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.extrusion_parser import extract_extrusion_dataframe

PROJECT_ID = "notpla-machine-data"
BUCKET_NAME = "notpla-machine-data"
PROCESSED_PREFIX = "machine-collin-e25e/machine-collin-e25e-processed/"


def load_manifest_by_checksum():
    client = bigquery.Client(project=PROJECT_ID)
    query = """
        SELECT checksum, source_file, rows_inserted, rows_rejected, processed_at
        FROM `notpla-machine-data.films_pipeline_ops.films_pipeline_manifest`
        WHERE pipeline = 'extrusion' AND status = 'success'
        ORDER BY processed_at
    """
    manifest = {}
    for row in client.query(query).result():
        # Later rows overwrite earlier ones for the same checksum, i.e. keep
        # the most recent manifest entry, since a file can be reprocessed.
        manifest[row.checksum] = {
            "source_file": row.source_file,
            "rows_inserted": row.rows_inserted,
            "rows_rejected": row.rows_rejected,
        }
    return manifest


def main():
    print("Loading manifest (pipeline=extrusion, status=success)...")
    manifest = load_manifest_by_checksum()
    print(f"  {len(manifest)} distinct checksums recorded.\n")

    storage_client = storage.Client(project=PROJECT_ID)
    blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix=PROCESSED_PREFIX))
    blobs = [b for b in blobs if b.name.lower().endswith(".csv")]
    print(f"Found {len(blobs)} processed extrusion CSVs to replay.\n")

    matched = 0
    mismatched = []
    no_manifest_entry = []
    parse_errors = []

    for i, blob in enumerate(blobs, 1):
        csv_bytes = blob.download_as_bytes()
        checksum = hashlib.md5(csv_bytes).hexdigest()
        filename = blob.name.split("/")[-1]

        entry = manifest.get(checksum)
        if entry is None:
            no_manifest_entry.append(filename)

        try:
            df, rows_dropped, _ = extract_extrusion_dataframe(csv_bytes, source_file=filename)
            rows_inserted = len(df)
        except Exception as exc:
            parse_errors.append((filename, str(exc)))
            continue

        if entry is None:
            continue

        if rows_inserted == entry["rows_inserted"] and rows_dropped == entry["rows_rejected"]:
            matched += 1
        else:
            mismatched.append({
                "filename": filename,
                "replayed": {"rows_inserted": rows_inserted, "rows_rejected": rows_dropped},
                "manifest": {"rows_inserted": entry["rows_inserted"], "rows_rejected": entry["rows_rejected"]},
            })

        if i % 100 == 0:
            print(f"  ...{i}/{len(blobs)} replayed")

    print()
    print(f"Matched exactly:        {matched}")
    print(f"Mismatched counts:      {len(mismatched)}")
    print(f"Parse raised on replay: {len(parse_errors)}")
    print(f"No manifest entry:      {len(no_manifest_entry)}")

    if mismatched:
        print("\n--- Mismatches ---")
        for m in mismatched:
            print(f"  {m['filename']}: replayed={m['replayed']} manifest={m['manifest']}")

    if parse_errors:
        print("\n--- Parse errors on replay ---")
        for filename, err in parse_errors:
            print(f"  {filename}: {err}")

    if no_manifest_entry:
        print(f"\n--- No manifest entry ({len(no_manifest_entry)}) ---")
        for filename in no_manifest_entry[:20]:
            print(f"  {filename}")
        if len(no_manifest_entry) > 20:
            print(f"  ...and {len(no_manifest_entry) - 20} more")


if __name__ == "__main__":
    main()
