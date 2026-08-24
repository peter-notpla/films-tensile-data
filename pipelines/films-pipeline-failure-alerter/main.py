import io
import json
import os
from datetime import datetime, timezone

import functions_framework
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage

PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BUCKET = os.environ.get("BUCKET", "notpla-machine-data")

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"
ALERTS_SENT_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_alerts_sent"
USER_DIRECTORY_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_user_directory"

# Each processor moves a failed file here before the manifest row's
# source_file (captured pre-move, under the pipeline's WATCH_PREFIX) stops
# pointing at anything real. Lifted from each pipeline's deployed env vars.
FAILED_PREFIXES = {
    "tensile": (
        "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
        "tensiletester-films-tensile-failed-processing/"
    ),
    "friction": (
        "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/"
        "tensiletester-films-friction-failed-processing/"
        "tensiletester-films-friction-failed-processing-summary/"
    ),
    "extrusion": "machine-collin-e25e/machine-collin-e25e-failed-processing/",
}

INITIALS_COLUMN_HINT = "user initials"

PIPELINE_READABLE = {
    "tensile": "Tensile Testing",
    "friction": "Friction Testing",
    "extrusion": "Extrusion",
}


def find_new_failures(bq):
    query = f"""
        SELECT m.pipeline, m.source_file, m.checksum, m.error_message, m.processed_at
        FROM `{MANIFEST_TABLE}` m
        WHERE m.status = 'failed'
          AND NOT EXISTS (
            SELECT 1 FROM `{ALERTS_SENT_TABLE}` a
            WHERE a.pipeline = m.pipeline AND a.checksum = m.checksum
          )
    """
    return list(bq.query(query).result())


def find_previous_failure(bq, pipeline, filename, current_processed_at):
    """The manifest row immediately before this one for the same pipeline
    and filename, if any. Matched on filename rather than checksum because
    a fixed re-upload is a new file with a new checksum; the documented fix
    procedure keeps the filename unchanged, which is what makes this a
    reliable repeat-failure signal. Returns None if this filename has never
    been seen before."""
    query = f"""
        SELECT status, error_message, processed_at
        FROM `{MANIFEST_TABLE}`
        WHERE pipeline = @pipeline
          AND ENDS_WITH(source_file, CONCAT('/', @filename))
          AND processed_at < @current_processed_at
        ORDER BY processed_at DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("pipeline", "STRING", pipeline),
        bigquery.ScalarQueryParameter("filename", "STRING", filename),
        bigquery.ScalarQueryParameter("current_processed_at", "TIMESTAMP", current_processed_at),
    ])
    rows = list(bq.query(query, job_config=job_config).result())
    return rows[0] if rows else None


def load_directory(bq):
    rows = bq.query(f"SELECT user_initials, route FROM `{USER_DIRECTORY_TABLE}`").result()
    return {r["user_initials"].strip().upper(): r["route"] for r in rows if r["user_initials"]}


def extract_initials(csv_bytes):
    """Best-effort: find a column whose header mentions 'user initials' and
    return its last non-blank value. Tensile and friction exports carry this
    column (not always: older friction files lack it); extrusion's template
    never does, so this naturally returns None for extrusion files.
    """
    try:
        text = csv_bytes.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) < 2:
            return None
        # First line is the title row, same convention as every processor.
        df = pd.read_csv(io.StringIO("\n".join(lines[1:])), dtype=str, keep_default_na=False)
        match_col = next(
            (c for c in df.columns if INITIALS_COLUMN_HINT in str(c).strip().lower()),
            None,
        )
        if match_col is None:
            return None
        values = [v.strip() for v in df[match_col].astype(str) if v.strip()]
        return values[-1].upper() if values else None
    except Exception:
        return None


def resolve_route(pipeline, source_file, directory):
    failed_prefix = FAILED_PREFIXES.get(pipeline)
    if failed_prefix is None:
        return "default", f"unknown_pipeline:{pipeline}"

    filename = source_file.split("/")[-1]
    storage_client = storage.Client(project=PROJECT_ID)
    blob = storage_client.bucket(BUCKET).blob(f"{failed_prefix}{filename}")

    try:
        csv_bytes = blob.download_as_bytes()
    except Exception as exc:
        print(f"ALERT_REREAD_FAILURE pipeline={pipeline} file={filename} error={exc}")
        return "default", "could_not_reread_file"

    initials = extract_initials(csv_bytes)
    if not initials:
        return "default", "no_initials_column"

    route = directory.get(initials)
    if route is None:
        return "default", f"initials_not_in_directory:{initials}"

    return route, f"initials:{initials}"


@functions_framework.http
def check_and_alert(request):
    bq = bigquery.Client(project=PROJECT_ID)

    failures = find_new_failures(bq)
    directory = load_directory(bq)

    alerted = 0
    dedup_write_failed = 0

    for row in failures:
        pipeline = row["pipeline"]
        source_file = row["source_file"]
        checksum = row["checksum"]
        error_message = row["error_message"] or ""
        filename = source_file.split("/")[-1]
        pipeline_readable = PIPELINE_READABLE.get(pipeline, pipeline)
        failed_at = (
            row["processed_at"].strftime("%d %b %Y %H:%M UTC")
            if row["processed_at"] else "unknown"
        )

        route, route_reason = resolve_route(pipeline, source_file, directory)

        # This is the line the pipeline_failure_alert log-based metric keys
        # off; Cloud Monitoring does the actual email send from here. route/
        # reason come before error deliberately: error_message can contain
        # embedded newlines (seen in real BigQuery error text), which split
        # into separate log entries downstream and would silently truncate
        # anything printed after the first one, including routing. Same
        # reasoning keeps pipeline_readable/failed_at (both may contain
        # spaces) before file/route/reason rather than after.
        print(
            f"PIPELINE_FAILURE_ALERT pipeline={pipeline} pipeline_readable={pipeline_readable} "
            f"failed_at={failed_at} file={filename} "
            f"route={route} reason={route_reason} error={error_message}"
        )

        previous = find_previous_failure(bq, pipeline, filename, row["processed_at"])
        if previous is not None and previous["status"] == "failed":
            previous_failed_at = (
                previous["processed_at"].strftime("%d %b %Y %H:%M UTC")
                if previous["processed_at"] else "unknown"
            )
            # Newlines flattened here (unlike error= below) so this field
            # can safely sit before error= without risking the same-entry
            # truncation the comment above describes.
            previous_error = (previous["error_message"] or "").replace("\n", " ")
            print(
                f"PIPELINE_FAILURE_ESCALATION pipeline={pipeline} pipeline_readable={pipeline_readable} "
                f"file={filename} failed_at={failed_at} previous_failed_at={previous_failed_at} "
                f"previous_error={previous_error} error={error_message}"
            )

        try:
            errors = bq.insert_rows_json(
                ALERTS_SENT_TABLE,
                [{
                    "pipeline": pipeline,
                    "source_file": source_file,
                    "checksum": checksum,
                    "route": route,
                    "route_reason": route_reason,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }],
            )
            if errors:
                print(f"ALERT_DEDUP_WRITE_FAILURE pipeline={pipeline} file={filename} errors={errors}")
                dedup_write_failed += 1
                continue
        except Exception as exc:
            # Not marked as sent, so this file is retried next hour instead
            # of silently never being recorded (and never alerting again).
            print(f"ALERT_DEDUP_WRITE_FAILURE pipeline={pipeline} file={filename} error={exc}")
            dedup_write_failed += 1
            continue

        alerted += 1

    summary = {"checked": len(failures), "alerted": alerted, "dedup_write_failed": dedup_write_failed}
    print(f"ALERT_RUN_SUMMARY {json.dumps(summary)}")
    return summary
