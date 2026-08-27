import os
import hashlib
import traceback
from datetime import datetime, timezone

import functions_framework
from google.cloud import bigquery, storage

from shared.extrusion_parser import extract_extrusion_dataframe

# Phase 3.3: Excel detection deliberately does not apply here. Row 1 of an
# extrusion export is a real section-header row (e.g. "Film Thickness
# Profile Average,,,Pellets QC,,,...") that legitimately ends in a comma by
# normal structure, not as an Excel artifact - confirmed by checking every
# real processed extrusion file, all 6 of which "flagged" on that signal
# alone, a 100% rate that was the tell it was a false positive, not a real
# finding. Extrusion also has no seconds-resolution timestamp field to fall
# back on (raw_films_extrusion's "date" is a DATE, no time-of-day). Unlike
# tensile/friction, extrusion's source machine (Collin E25E) isn't part of
# the "opened in Excel during the manual check step" workflow CLAUDE.md
# describes for the Mecmesin tensile tester's VectorPro exports.


PROJECT_ID = os.environ["PROJECT_ID"]
BQ_DATASET = os.environ["BQ_DATASET"]
BQ_TABLE = os.environ["BQ_TABLE"]
WATCH_PREFIX = os.environ["WATCH_PREFIX"].strip("/")
PROCESSED_PREFIX = os.environ["PROCESSED_PREFIX"].strip("/")
FAILED_PREFIX = os.environ["FAILED_PREFIX"].strip("/")

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"
PIPELINE_NAME = "extrusion"
MACHINE_ID = "collin-e25e"

storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)

print("PHASE_4_CUTOVER: extrusion parsing now sourced from shared.extrusion_parser (27 Aug 2026)")


def move_blob(bucket_name, blob_name, new_prefix):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    new_name = f"{new_prefix}/{blob_name.split('/')[-1]}"
    bucket.copy_blob(blob, bucket, new_name)
    blob.delete()


def delete_stale_failed_copy(bucket_name, blob_name):
    """Best-effort: once this filename has successfully reprocessed, remove
    any earlier failed copy sitting under FAILED_PREFIX, so a fixed file
    doesn't leave what looks like an unresolved failure there forever."""
    try:
        filename = blob_name.split("/")[-1]
        failed_path = f"{FAILED_PREFIX}/{filename}"
        blob = storage_client.bucket(bucket_name).blob(failed_path)
        if blob.exists():
            blob.delete()
            print(f"Deleted stale failed-processing copy: {failed_path}")
    except Exception as cleanup_exc:
        print(f"Failed to delete stale failed-processing copy (pipeline result unaffected): {cleanup_exc}")


def write_manifest(source_file, checksum, status, rows_total, rows_inserted,
                    rows_rejected, error_message, excel_processed=None):
    """Best-effort manifest row. Never allowed to fail the pipeline run."""
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
            "excel_processed": excel_processed,
        }
        errors = bq_client.insert_rows_json(MANIFEST_TABLE, [row])
        if errors:
            print(f"Manifest insert returned errors: {errors}")
    except Exception as manifest_exc:
        print(f"Manifest insert failed (pipeline result unaffected): {manifest_exc}")


def write_row_errors(row_errors, source_file, checksum):
    """Best-effort row-errors write. Never allowed to fail the pipeline run."""
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
        errors = bq_client.insert_rows_json(ROW_ERRORS_TABLE, rows)
        if errors:
            print(f"Row-errors insert returned errors: {errors}")
    except Exception as row_errors_exc:
        print(f"Row-errors insert failed (pipeline result unaffected): {row_errors_exc}")


@functions_framework.cloud_event
def process_file(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    blob_name = data["name"]

    print(f"Received file: {blob_name}")

    if not blob_name.startswith(WATCH_PREFIX):
        print("Skipping: outside watch folder")
        return

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    checksum = None
    excel_processed = None

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        checksum = hashlib.md5(content).hexdigest()
        # excel_processed intentionally stays None for extrusion - see the
        # module-level comment on the excel_detection import removal above.

        df, rows_dropped, row_errors = extract_extrusion_dataframe(
            content, source_file=blob_name.split("/")[-1]
        )
        rows_total = len(df) + rows_dropped

        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()

        print(f"Loaded {len(df)} rows to {table_id}")

        move_blob(bucket_name, blob_name, PROCESSED_PREFIX)

        delete_stale_failed_copy(bucket_name, blob_name)

        write_row_errors(row_errors, source_file=gcs_uri, checksum=checksum)

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="success",
            rows_total=rows_total,
            rows_inserted=len(df),
            rows_rejected=rows_dropped,
            error_message=None,
            excel_processed=excel_processed,
        )

    except Exception as exc:
        # ---------------------------------------------------------------
        # RESTORED 20 Aug 2026.
        #
        # The deployed version of this block printed the traceback and then
        # returned normally. That meant a failed file was never moved to the
        # failed folder, the function reported success to Eventarc, and the
        # file stayed in to-be-processed where the uploader's blob.exists()
        # check treated every future copy as a duplicate. Failures were
        # completely invisible.
        #
        # The two behaviours below are what make a failure observable:
        #   1. move the file so its location reflects its state
        #   2. re-raise so Eventarc and Cloud Logging record an error
        #
        # Re-raising is safe here because the trigger uses
        # RETRY_POLICY_DO_NOT_RETRY, so this will not loop.
        # ---------------------------------------------------------------
        print(f"EXTRUSION_PIPELINE_FAILURE file={blob_name} error={exc}")
        print(traceback.format_exc())

        try:
            if storage_client.bucket(bucket_name).blob(blob_name).exists():
                move_blob(bucket_name, blob_name, FAILED_PREFIX)
                print(f"Moved failed file to {FAILED_PREFIX}/{blob_name.split('/')[-1]}")
            else:
                print("Blob no longer exists, nothing to move")
        except Exception as move_exc:
            print(f"EXTRUSION_PIPELINE_FAILURE could not move to failed prefix: {move_exc}")
            print(traceback.format_exc())

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="failed",
            rows_total=None,
            rows_inserted=0,
            rows_rejected=0,
            error_message=str(exc)[:1500],
            excel_processed=excel_processed,
        )

        raise
