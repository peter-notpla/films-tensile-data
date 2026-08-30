"""Replay shared/curve_parser.py against every real file in the tensile
raw-samples backlog (not a sample - see CLAUDE.md's verification
discipline), and separately confirm shared/curve_linking.py's nullable
behaviour against two known real cases from Phase 5 planning: a raw file
close enough in time to link, and one far enough away that it should not.

Read-only: downloads files and queries BigQuery, writes nothing.

Usage: python3 shared/verify_curve_parser.py
"""

import sys
from pathlib import Path

from google.cloud import bigquery
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.curve_parser import downsample_curve_minmax, extract_curve_dataframe
from shared.curve_linking import find_specimen_link

PROJECT_ID = "notpla-machine-data"
BUCKET_NAME = "notpla-machine-data"
TENSILE_BASE = (
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
)
RAW_SAMPLES_PREFIX = f"{TENSILE_BASE}tensiletester-films-tensile-raw-samples/"
# Files scattered across watch/processed/failed folders from an earlier
# partial run - scan all three so "every real file" actually means every
# real file, not just whatever's still sitting in the watch folder today.
ALL_TENSILE_RAW_PREFIXES = [
    RAW_SAMPLES_PREFIX,
    f"{TENSILE_BASE}tensiletester-films-tensile-raw-samples-processed/",
    f"{TENSILE_BASE}tensiletester-films-tensile-raw-samples-failed-processing/",
]
TENSILE_RESULTS_TABLE = f"{PROJECT_ID}.films_tensile_london.films_tensile_results_all_revisions"


def verify_parser():
    storage_client = storage.Client(project=PROJECT_ID)
    blobs = []
    for prefix in ALL_TENSILE_RAW_PREFIXES:
        blobs.extend(storage_client.list_blobs(BUCKET_NAME, prefix=prefix))
    blobs = [b for b in blobs if b.name.lower().endswith(".csv")]

    print(f"Found {len(blobs)} real raw files across watch/processed/failed folders")

    succeeded = 0
    failed = []
    total_rows = 0
    total_dropped = 0
    total_downsampled_rows = 0
    minmax_mismatches = []

    for blob in blobs:
        filename = blob.name.split("/")[-1]
        try:
            content = blob.download_as_bytes()
            df, row_errors = extract_curve_dataframe(content, source_file=filename)
            succeeded += 1
            total_rows += len(df)
            total_dropped += len(row_errors)

            small = downsample_curve_minmax(df)
            total_downsampled_rows += len(small)
            if len(df) > 200 and len(small) > len(df):
                minmax_mismatches.append((filename, "downsampled output is larger than input"))
            if not df.empty:
                if small["load_n"].max() != df["load_n"].max() or small["load_n"].min() != df["load_n"].min():
                    minmax_mismatches.append((filename, "global load_n min/max not preserved"))
        except Exception as exc:
            failed.append((filename, str(exc)))

    print(f"Succeeded: {succeeded}/{len(blobs)}")
    print(f"Total curve-point rows produced: {total_rows} (rows dropped as all-NaN: {total_dropped})")
    print(
        f"Total rows after min/max downsampling: {total_downsampled_rows} "
        f"({100 * total_downsampled_rows / total_rows:.1f}% of full resolution)"
    )
    if minmax_mismatches:
        print(f"Downsampling problems: {len(minmax_mismatches)}")
        for filename, reason in minmax_mismatches:
            print(f"  {filename}: {reason}")
    else:
        print("Downsampling preserved global load_n min/max on every file.")
    if failed:
        print(f"Failed: {len(failed)}")
        for filename, reason in failed:
            print(f"  {filename}: {reason}")
    else:
        print("No failures.")

    return len(blobs), succeeded, failed, minmax_mismatches


def verify_linking():
    bq = bigquery.Client(project=PROJECT_ID)
    storage_client = storage.Client(project=PROJECT_ID)

    cases = [
        ("raw-TensileTest-Films(V1)-sample-100.csv", "expected: linkable, small delta"),
        ("raw-TensileTest-Films(V1)-sample-107.csv", "expected: unlinkable (outside window)"),
    ]

    print("\nLinking verification against known real cases:")
    for filename, expectation in cases:
        blob = storage_client.bucket(BUCKET_NAME).blob(f"{RAW_SAMPLES_PREFIX}{filename}")
        blob.reload()
        created_at = blob.time_created

        specimen_key, delta_seconds = find_specimen_link(bq, TENSILE_RESULTS_TABLE, created_at)
        print(f"  {filename} (created {created_at}, {expectation})")
        print(f"    -> specimen_key={specimen_key} delta_seconds={delta_seconds}")


if __name__ == "__main__":
    total, succeeded, failed, minmax_mismatches = verify_parser()
    verify_linking()
    if failed or succeeded != total or minmax_mismatches:
        sys.exit(1)
