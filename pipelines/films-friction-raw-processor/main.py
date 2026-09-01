import os
import hashlib
import traceback
from datetime import datetime, timezone
from html import escape

import functions_framework
from google.cloud import bigquery, storage

from shared import email_style, gmail_sender
from shared.bq_retry import load_dataframe_with_retry
from shared.curve_parser import downsample_curve_minmax, extract_curve_dataframe
from shared.curve_linking import find_specimen_link

PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BQ_DATASET = os.environ.get("BQ_DATASET", "machine_data")
BQ_TABLE = os.environ.get("BQ_TABLE", "films_friction_curve_points")
RESULTS_TABLE = os.environ.get(
    "RESULTS_TABLE",
    f"{PROJECT_ID}.machine_data.films_friction_raw_all_revisions",
)
# Same trailing-slash anchoring as films-tensile-raw-processor - see that
# file's comment. PROCESSED_PREFIX/FAILED_PREFIX are sibling folders under
# the same watch parent here too.
WATCH_PREFIX = os.environ["WATCH_PREFIX"]
if not WATCH_PREFIX.endswith("/"):
    WATCH_PREFIX += "/"
PROCESSED_PREFIX = os.environ["PROCESSED_PREFIX"].strip("/")
FAILED_PREFIX = os.environ["FAILED_PREFIX"].strip("/")
FAILURE_ALERT_RECIPIENT = os.environ.get("FAILURE_ALERT_RECIPIENT", "peter@notpla.com")

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"
PIPELINE_NAME = "friction_raw"
MACHINE_ID = "tensiletester-1"

storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)

LOGS_URL = f"https://console.cloud.google.com/functions/details/europe-west2/films-friction-raw-processor?project={PROJECT_ID}&tab=logs"


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
                    rows_rejected, error_message):
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
            "excel_processed": None,
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


def send_failure_email(blob_name, error_message):
    """Best-effort, like write_manifest above: a failed send here must never
    mask the real processing failure this is reporting on."""
    try:
        filename = blob_name.split("/")[-1]
        body = email_style.section_header("What happened")
        body += email_style.key_value_table([
            ("File", f"<code>{escape(filename)}</code>"),
            ("Error", escape(error_message)),
        ])
        body += email_style.paragraph(
            "No data was lost: the file has been moved to the "
            "failed-processing folder, untouched. No rows were written to "
            "BigQuery for this file."
        )
        body += email_style.divider()
        body += email_style.section_header("Technical details")
        body += email_style.paragraph(
            "See the full traceback via the link below, immediately after "
            "the line beginning FRICTION_RAW_PIPELINE_FAILURE."
        )
        body += email_style.cta_link("View pipeline logs", LOGS_URL)
        html = email_style.wrap_email("Hello,", body)

        send_result = gmail_sender.send_html_email(
            PROJECT_ID, FAILURE_ALERT_RECIPIENT,
            f"[Alert] Friction raw curve: {filename} failed processing", html,
        )
        print(f"FRICTION_RAW_FAILURE_ALERT_SENT file={filename} message_id={send_result.get('id')}")
    except Exception as send_exc:
        print(f"FRICTION_RAW_FAILURE_ALERT_SEND_FAILED file={blob_name} error={send_exc}")


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

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        checksum = hashlib.md5(content).hexdigest()
        blob.reload()
        gcs_created_at = blob.time_created

        filename = blob_name.split("/")[-1]
        df, row_errors = extract_curve_dataframe(content, source_file=filename)
        rows_total = len(df) + len(row_errors)
        df = downsample_curve_minmax(df)

        specimen_key, delta_seconds = find_specimen_link(bq_client, RESULTS_TABLE, gcs_created_at)
        df["linked_specimen_key"] = specimen_key
        df["link_time_delta_seconds"] = delta_seconds

        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        load_dataframe_with_retry(bq_client, df, table_id)

        print(
            f"Loaded {len(df)} rows to {table_id} "
            f"(linked_specimen_key={specimen_key} delta_seconds={delta_seconds})"
        )

        move_blob(bucket_name, blob_name, PROCESSED_PREFIX)
        delete_stale_failed_copy(bucket_name, blob_name)
        write_row_errors(row_errors, source_file=gcs_uri, checksum=checksum)

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="success",
            rows_total=rows_total,
            rows_inserted=len(df),
            rows_rejected=len(row_errors),
            error_message=None,
        )

    except Exception as exc:
        # Same restored-failure-path shape as films-extrusion-csv-processor
        # and films-tensile-raw-processor: move the file so its location
        # reflects its state, then re-raise so Eventarc and Cloud Logging
        # record an error. Safe to re-raise since the trigger uses
        # RETRY_POLICY_DO_NOT_RETRY.
        print(f"FRICTION_RAW_PIPELINE_FAILURE file={blob_name} error={exc}")
        print(traceback.format_exc())

        try:
            if storage_client.bucket(bucket_name).blob(blob_name).exists():
                move_blob(bucket_name, blob_name, FAILED_PREFIX)
                print(f"Moved failed file to {FAILED_PREFIX}/{blob_name.split('/')[-1]}")
            else:
                print("Blob no longer exists, nothing to move")
        except Exception as move_exc:
            print(f"FRICTION_RAW_PIPELINE_FAILURE could not move to failed prefix: {move_exc}")
            print(traceback.format_exc())

        send_failure_email(blob_name, str(exc))

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
