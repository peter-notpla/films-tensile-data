"""One-off backfill of an existing raw-curve backlog (tensile or friction)
into its *_curve_points table, using the same shared/curve_parser.py and
shared/curve_linking.py the corresponding live raw-processor pipeline uses,
and writing to the same manifest/row-errors tables so backfilled files are
as observable as anything the live pipeline handles. Moves each file to the
same PROCESSED_PREFIX/FAILED_PREFIX the live pipeline uses afterward, so the
watch folder ends up in the same state it would be in had the live pipeline
processed this backlog itself.

Mirrors backfill/backfill.py's shape (see CLAUDE.md's Phase 0.3 note), but
that script predates the Phase 1 manifest table; this one writes to it.

All config below reads from the environment with tensile's values as
defaults, so `scripts/deploy_curve_backfill_job.sh` can run this as a
friction backfill by passing different env vars at execute time - one
script, not two near-identical copies that can drift out of sync with each
other the way films-friction-raw-processor's early draft did (dataset
mismatch against the live friction pipeline, caught during Phase 5 step 3).

Usage: python3 scripts/backfill_curve_points.py
       PIPELINE_NAME=friction_raw WATCH_PREFIX=... BQ_DATASET=... BQ_TABLE=... RESULTS_TABLE=... python3 scripts/backfill_curve_points.py
"""

import hashlib
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.cloud import storage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.bq_retry import load_dataframe_with_retry
from shared.curve_linking import find_specimen_link
from shared.curve_parser import downsample_curve_minmax, extract_curve_dataframe

PROJECT_ID = "notpla-machine-data"
BUCKET = "notpla-machine-data"

WATCH_PREFIX = os.environ.get(
    "WATCH_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-raw-samples/",
)
PROCESSED_PREFIX = os.environ.get(
    "PROCESSED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-raw-samples-processed/",
)
FAILED_PREFIX = os.environ.get(
    "FAILED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-raw-samples-failed-processing/",
)

BQ_DATASET = os.environ.get("BQ_DATASET", "films_tensile_london")
BQ_TABLE = os.environ.get("BQ_TABLE", "films_tensile_curve_points")
RESULTS_TABLE = os.environ.get(
    "RESULTS_TABLE",
    f"{PROJECT_ID}.films_tensile_london.films_tensile_results_all_revisions",
)

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "tensile_raw")
MACHINE_ID = "tensiletester-1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_curve_points")


def write_manifest(bq, source_file, checksum, status, rows_total, rows_inserted, rows_rejected, error_message):
    try:
        row = {
            "pipeline": PIPELINE_NAME,
            "source_file": source_file,
            "checksum": checksum,
            "machine_id": MACHINE_ID,
            "status": status,
            "rows_total": rows_total,
            "rows_inserted": rows_inserted,
            "rows_rejected": rows_rejected,
            "error_message": error_message,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "excel_processed": None,
        }
        errors = bq.insert_rows_json(MANIFEST_TABLE, [row])
        if errors:
            logger.warning("Manifest insert returned errors: %s", errors)
    except Exception:
        logger.exception("Manifest insert failed (backfill result unaffected)")


def write_row_errors(bq, row_errors, source_file, checksum):
    if not row_errors:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "pipeline": PIPELINE_NAME,
                "source_file": source_file,
                "checksum": checksum,
                "row_number": e["row_number"],
                "reason": e["reason"],
                "raw_row": e["raw_row"],
                "processed_at": now,
            }
            for e in row_errors
        ]
        errors = bq.insert_rows_json(ROW_ERRORS_TABLE, rows)
        if errors:
            logger.warning("Row-errors insert returned errors: %s", errors)
    except Exception:
        logger.exception("Row-errors insert failed (backfill result unaffected)")


def move_blob(gcs, source_name, dest_name):
    bucket = gcs.bucket(BUCKET)
    blob = bucket.blob(source_name)
    bucket.copy_blob(blob, bucket, new_name=dest_name)
    blob.delete()


def main():
    gcs = storage.Client(project=PROJECT_ID)
    bq = bigquery.Client(project=PROJECT_ID)
    bucket = gcs.bucket(BUCKET)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

    blobs = list(gcs.list_blobs(BUCKET, prefix=WATCH_PREFIX))
    csv_blobs = [b for b in blobs if b.name.lower().endswith(".csv")]

    # Dedupe by name: a GCS list() called right after a large burst of moves
    # on the same prefix can transiently return the same object twice
    # (observed 30 August 2026, ~40 duplicate entries after moving hundreds
    # of files back into this folder). Keeping only the first occurrence
    # means a duplicate is silently skipped rather than double-processed -
    # the second attempt would otherwise 404 on an already-moved source and
    # be wrongly quarantined as a failure.
    seen_names = set()
    deduped = []
    for b in csv_blobs:
        if b.name not in seen_names:
            seen_names.add(b.name)
            deduped.append(b)
    if len(deduped) != len(csv_blobs):
        logger.warning("Listing had %d duplicate entries, deduped", len(csv_blobs) - len(deduped))
    csv_blobs = deduped

    logger.info("Found %d CSVs to backfill", len(csv_blobs))

    processed_files = 0
    failed_files = 0
    inserted_rows = 0
    linked_files = 0

    for i, blob in enumerate(csv_blobs, start=1):
        name = blob.name
        filename = name.split("/")[-1]
        gcs_uri = f"gs://{BUCKET}/{name}"
        checksum = None

        try:
            content = bucket.blob(name).download_as_bytes()
        except NotFound:
            # Source object already gone - already handled by an earlier
            # entry (see the dedup above) or moved by something else since
            # listing. Not a real failure: skip without quarantining or
            # logging to the manifest, since nothing was loaded yet and
            # there's nothing here to record that isn't already recorded
            # by whatever processed it first.
            logger.warning("[%d/%d] SKIP (already gone): %s", i, len(csv_blobs), filename)
            continue

        try:
            checksum = hashlib.md5(content).hexdigest()
            blob.reload()
            gcs_created_at = blob.time_created

            df, row_errors = extract_curve_dataframe(content, source_file=filename)
            rows_total = len(df) + len(row_errors)
            df = downsample_curve_minmax(df)

            specimen_key, delta_seconds = find_specimen_link(bq, RESULTS_TABLE, gcs_created_at)
            df["linked_specimen_key"] = specimen_key
            df["link_time_delta_seconds"] = delta_seconds
            if specimen_key is not None:
                linked_files += 1

            load_dataframe_with_retry(bq, df, table_id)

            move_blob(gcs, name, f"{PROCESSED_PREFIX}{filename}")
            write_row_errors(bq, row_errors, source_file=gcs_uri, checksum=checksum)
            write_manifest(
                bq, source_file=gcs_uri, checksum=checksum, status="success",
                rows_total=rows_total, rows_inserted=len(df), rows_rejected=len(row_errors),
                error_message=None,
            )

            inserted_rows += len(df)
            processed_files += 1
            logger.info(
                "[%d/%d] OK: %s inserted=%d linked=%s delta_s=%s",
                i, len(csv_blobs), filename, len(df), specimen_key is not None, delta_seconds,
            )

        except Exception as exc:
            logger.exception("[%d/%d] FAIL: %s", i, len(csv_blobs), filename)
            try:
                move_blob(gcs, name, f"{FAILED_PREFIX}{filename}")
            except Exception:
                logger.exception("Also failed to move %s to failed-processing", filename)
            write_manifest(
                bq, source_file=gcs_uri, checksum=checksum, status="failed",
                rows_total=None, rows_inserted=0, rows_rejected=0,
                error_message=str(exc)[:1500],
            )
            failed_files += 1

    logger.info(
        "DONE. processed_files=%d failed_files=%d inserted_rows=%d linked_files=%d",
        processed_files, failed_files, inserted_rows, linked_files,
    )


if __name__ == "__main__":
    main()
