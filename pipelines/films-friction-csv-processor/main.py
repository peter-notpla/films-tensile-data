import os
import hashlib
import tempfile
import traceback
from datetime import datetime, timezone

from google.cloud import bigquery
from google.cloud import storage

from shared.friction_parser import extract_friction_dataframe
from shared.excel_detection import is_excel_processed
from shared.revision_handling import dedupe_within_file, apply_revision_handling


PIPELINE_NAME = "friction"
MACHINE_ID = "tensiletester-1"

print("PHASE_4_CUTOVER: friction parsing now sourced from shared.friction_parser (27 Aug 2026)")


def write_manifest(project_id, source_file, checksum, status, rows_total,
                    rows_inserted, rows_rejected, error_message, excel_processed=None):
    """Best-effort manifest row. Never allowed to fail the pipeline run."""
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.films_pipeline_ops.films_pipeline_manifest"
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
            "excel_processed": excel_processed,
        }
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print(f"Manifest insert returned errors: {errors}")
    except Exception as manifest_exc:
        print(f"Manifest insert failed (pipeline result unaffected): {manifest_exc}")


def delete_stale_failed_copy(bucket, failed_path):
    """Best-effort: once this filename has successfully reprocessed, remove
    any earlier failed copy sitting at failed_path, so a fixed file doesn't
    leave what looks like an unresolved failure there forever."""
    try:
        blob = bucket.blob(failed_path)
        if blob.exists():
            blob.delete()
            print(f"Deleted stale failed-processing copy: {failed_path}")
    except Exception as cleanup_exc:
        print(f"Failed to delete stale failed-processing copy (pipeline result unaffected): {cleanup_exc}")


def write_row_errors(project_id, row_errors, source_file, checksum):
    """Best-effort row-errors write. Never allowed to fail the pipeline run."""
    if not row_errors:
        return
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.films_pipeline_ops.films_pipeline_row_errors"
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
        errors = client.insert_rows_json(table_id, rows)
        if errors:
            print(f"Row-errors insert returned errors: {errors}")
    except Exception as row_errors_exc:
        print(f"Row-errors insert failed (pipeline result unaffected): {row_errors_exc}")


def gcs_csv_to_bigquery(data, context):
    PROJECT_ID = os.environ["PROJECT_ID"]
    BQ_DATASET = os.environ["BQ_DATASET"]
    BQ_TABLE = os.environ["BQ_TABLE"]
    WATCH_PREFIX = os.environ["WATCH_PREFIX"].rstrip("/") + "/"
    PROCESSED_PREFIX = os.environ["PROCESSED_PREFIX"].rstrip("/") + "/"
    FAILED_PREFIX = os.environ["FAILED_PREFIX"].rstrip("/") + "/"

    bucket_name = data["bucket"]
    blob_name = data["name"]

    if not blob_name.startswith(WATCH_PREFIX) or blob_name.endswith("/"):
        print(f"Skipping: {blob_name}")
        return

    storage_client = storage.Client()
    bq_client = bigquery.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)

    filename = blob_name.split("/")[-1]
    processed_path = f"{PROCESSED_PREFIX}{filename}"
    failed_path = f"{FAILED_PREFIX}{filename}"
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    checksum = None
    excel_processed = None

    try:
        print(f"Processing gs://{bucket_name}/{blob_name}")

        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            blob.download_to_filename(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            content = f.read()
        checksum = hashlib.md5(content).hexdigest()

        os.remove(tmp_path)

        # Padding-only pre-check, available even if parsing fails below (a
        # padded row 1 can itself be the cause of a parse failure).
        excel_processed = is_excel_processed(content)
        df, rows_dropped, row_errors = extract_friction_dataframe(content, source_file=gcs_uri)
        # Refine with the timestamp signal now that parsing succeeded.
        excel_processed = is_excel_processed(content, df["timestamp_start"])

        # Metadata revision handling (Phase 2.5): a specimen already present
        # as row_state='current' gets archived, this file's row becomes the
        # new current one, instead of both existing as an undetected
        # duplicate under plain append. dedupe_within_file must run first -
        # two rows in one file both claiming 'current' for the same key
        # would defeat the point.
        df, row_errors = dedupe_within_file(df, row_errors)
        if df.empty:
            raise ValueError("No valid rows remain after removing within-file duplicate specimen_keys")
        rows_dropped = len(row_errors)
        rows_total = len(df) + rows_dropped

        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        df = apply_revision_handling(df, bq_client, table_id, source_file=gcs_uri)
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()

        bucket.copy_blob(blob, bucket, processed_path)
        blob.delete()

        print(f"SUCCESS -> {processed_path}")

        delete_stale_failed_copy(bucket, failed_path)

        write_row_errors(PROJECT_ID, row_errors, source_file=gcs_uri, checksum=checksum)

        write_manifest(
            project_id=PROJECT_ID,
            source_file=gcs_uri,
            checksum=checksum,
            status="success",
            rows_total=rows_total,
            rows_inserted=len(df),
            rows_rejected=rows_dropped,
            error_message=None,
            excel_processed=excel_processed,
        )

    except Exception as e:
        print(f"FAILED: {e}")
        print(traceback.format_exc())

        try:
            blob = bucket.blob(blob_name)
            if blob.exists():
                bucket.copy_blob(blob, bucket, failed_path)
                blob.delete()
                print(f"Moved to failed: {failed_path}")
        except Exception as move_err:
            print(f"Move failed: {move_err}")

        write_manifest(
            project_id=PROJECT_ID,
            source_file=gcs_uri,
            checksum=checksum,
            status="failed",
            rows_total=None,
            rows_inserted=0,
            rows_rejected=0,
            error_message=str(e)[:1500],
            excel_processed=excel_processed,
        )

        raise
