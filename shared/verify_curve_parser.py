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
    """Synthetic cases against a real specimen row, not live GCS blob
    metadata: once a raw file is moved to processed/failed (as this whole
    backlog now is), `move_blob`'s copy_blob-then-delete gives the copy a
    new object generation and therefore a new `time_created` reflecting the
    move, not the original upload - confirmed 4 September 2026, see
    curve_linking.py's docstring. Real blob timestamps from this point
    onward are meaningless for verification, so this constructs
    `gcs_created_at` directly from a known-real specimen's own
    `timestamp_start` instead, which stays valid regardless of file moves."""
    bq = bigquery.Client(project=PROJECT_ID)

    row = list(bq.query(f"""
        SELECT specimen_key, template_name, timestamp_start
        FROM `{TENSILE_RESULTS_TABLE}`
        WHERE row_state = 'current' AND template_name IS NOT NULL
        ORDER BY timestamp_start DESC
        LIMIT 1
    """).result())[0]

    print("\nLinking verification against a real specimen row, synthetic timestamps:")

    cases = [
        ("same template, 60s off", row["template_name"], 60, True),
        ("same template, outside window", row["template_name"], 3600, False),
        ("different template, 60s off", row["template_name"] + "-DOES-NOT-EXIST", 60, False),
    ]
    all_ok = True
    for label, template_name, offset_seconds, expect_match in cases:
        from datetime import timedelta
        synthetic_created_at = row["timestamp_start"] + timedelta(seconds=offset_seconds)
        specimen_key, delta_seconds = find_specimen_link(
            bq, TENSILE_RESULTS_TABLE, synthetic_created_at, template_name
        )
        got_match = specimen_key is not None
        ok = got_match == expect_match
        all_ok = all_ok and ok
        print(f"  {label}: expect_match={expect_match} got specimen_key={specimen_key} "
              f"delta_seconds={delta_seconds} {'OK' if ok else 'MISMATCH'}")

    return all_ok


if __name__ == "__main__":
    total, succeeded, failed, minmax_mismatches = verify_parser()
    linking_ok = verify_linking()
    if failed or succeeded != total or minmax_mismatches or not linking_ok:
        sys.exit(1)
