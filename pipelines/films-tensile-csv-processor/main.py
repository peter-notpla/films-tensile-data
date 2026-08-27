import os
import hashlib
import logging
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

from shared.tensile_parser import extract_relevant_dataframe


# ---------------- Config ----------------
PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BQ_DATASET = os.environ.get("BQ_DATASET", "films_tensile_london")
BQ_TABLE = os.environ.get("BQ_TABLE", "films_tensile_results")

WATCH_PREFIX = os.environ.get(
    "WATCH_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-summary-tables/"
)
PROCESSED_PREFIX = os.environ.get(
    "PROCESSED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-processed/"
)
FAILED_PREFIX = os.environ.get(
    "FAILED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-failed-processing/"
)

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"
PIPELINE_NAME = "tensile"
MACHINE_ID = "tensiletester-1"

logger = logging.getLogger("tensile_processor")
logger.setLevel(logging.INFO)

logger.info("PHASE_4_CUTOVER: tensile parsing now sourced from shared.tensile_parser (27 Aug 2026)")


def should_process(object_name: str) -> bool:
    # Only process CSVs landing under WATCH_PREFIX
    if not object_name.startswith(WATCH_PREFIX):
        return False
    # Never process anything already moved
    if object_name.startswith(PROCESSED_PREFIX) or object_name.startswith(FAILED_PREFIX):
        return False
    return object_name.lower().endswith(".csv")


def load_to_bigquery(df: pd.DataFrame) -> int:
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

    # Ensure columns match BigQuery schema order
    df = df[
        [
            "sample",
            "youngs_modulus_mpa",
            "offset_yield_mpa",
            "max_load_n",
            "max_stress_mpa",
            "break_pct",
            "toughness_mpa",
            "timestamp_start",
            "pellet_id",
            "extrusion_id",
            "test_direction",
            "sample_number",
            "sample_thickness_mm",
            "relative_humidity_pct",
            "notes",
            "user_initials",
            "source_file",
            "processed_at",
        ]
    ].copy()

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND),
    )
    job.result()
    return len(df)


def move_blob(bucket_name: str, source_name: str, dest_name: str) -> None:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_name)

    bucket.copy_blob(blob, bucket, new_name=dest_name)
    blob.delete()


def delete_stale_failed_copy(bucket_name: str, filename: str) -> None:
    """Best-effort: once this filename has successfully reprocessed, remove
    any earlier failed copy sitting under FAILED_PREFIX, so a fixed file
    doesn't leave what looks like an unresolved failure there forever."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        blob = storage_client.bucket(bucket_name).blob(f"{FAILED_PREFIX}{filename}")
        if blob.exists():
            blob.delete()
            logger.info("Deleted stale failed-processing copy", extra={"target_filename": filename})
    except Exception:
        logger.exception(
            "Failed to delete stale failed-processing copy; pipeline result unaffected",
            extra={"target_filename": filename},
        )


def write_manifest(source_file, checksum, status, rows_total, rows_inserted,
                    rows_rejected, error_message):
    """Best-effort manifest row. Never allowed to fail the pipeline run."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
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
        }
        errors = client.insert_rows_json(MANIFEST_TABLE, [row])
        if errors:
            logger.error("Manifest insert returned errors", extra={"errors": errors, "source_file": source_file})
    except Exception:
        logger.exception("Manifest insert failed; pipeline result unaffected", extra={"source_file": source_file})


def write_row_errors(row_errors, source_file, checksum):
    """Best-effort row-errors write. Never allowed to fail the pipeline run."""
    if not row_errors:
        return
    try:
        client = bigquery.Client(project=PROJECT_ID)
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
        errors = client.insert_rows_json(ROW_ERRORS_TABLE, rows)
        if errors:
            logger.error("Row-errors insert returned errors", extra={"errors": errors, "source_file": source_file})
    except Exception:
        logger.exception("Row-errors insert failed; pipeline result unaffected", extra={"source_file": source_file})


def process_gcs_event(event, context=None):
    bucket = event.get("bucket") if isinstance(event, dict) else None
    name = event.get("name") if isinstance(event, dict) else None

    if not bucket or not name:
        logger.error("Missing bucket/name in event", extra={"event": str(event)})
        return

    logger.info("Event received", extra={"bucket": bucket, "object_name": name})

    if not should_process(name):
        logger.info("Skipping (not in watch scope)", extra={"bucket": bucket, "object_name": name})
        return

    storage_client = storage.Client(project=PROJECT_ID)
    blob = storage_client.bucket(bucket).blob(name)
    gcs_uri = f"gs://{bucket}/{name}"
    checksum = None

    try:
        csv_bytes = blob.download_as_bytes()
        checksum = hashlib.md5(csv_bytes).hexdigest()
        df, rows_dropped, row_errors = extract_relevant_dataframe(csv_bytes, source_file=name)

        rows = load_to_bigquery(df)

        filename = name.split("/")[-1]
        dest = f"{PROCESSED_PREFIX}{filename}"
        move_blob(bucket, name, dest)

        logger.info("Processed OK", extra={"source_file": name, "rows_inserted": rows, "moved_to": dest})

        delete_stale_failed_copy(bucket, filename)

        write_row_errors(row_errors, source_file=gcs_uri, checksum=checksum)

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="success",
            rows_total=rows + rows_dropped,
            rows_inserted=rows,
            rows_rejected=rows_dropped,
            error_message=None,
        )

    except Exception as exc:
        try:
            filename = name.split("/")[-1]
            dest = f"{FAILED_PREFIX}{filename}"
            move_blob(bucket, name, dest)
            logger.exception("Processing failed; moved to failed-processing", extra={"source_file": name, "moved_to": dest})
        except Exception:
            logger.exception("Processing failed; also failed to move to failed-processing", extra={"source_file": name})

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="failed",
            rows_total=None,
            rows_inserted=0,
            rows_rejected=0,
            error_message=str(exc)[:1500],
        )

        raise
