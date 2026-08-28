import os
from datetime import datetime, timedelta, timezone
from html import escape

import functions_framework
from google.cloud import bigquery

from shared import email_style, gmail_sender

PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
DIGEST_RECIPIENT = os.environ.get("DIGEST_RECIPIENT", "peter@notpla.com")

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"

# Failed files older than this are still shown but flagged as needing a
# second look, matching the "already had its own alert" framing below.
FAILED_FILES_LIMIT = 25

# A most-recent-test date older than this, relative to the digest window's
# end, gets called out explicitly rather than left for the reader to notice.
STALE_TEST_THRESHOLD_DAYS = 14

PIPELINE_READABLE = {
    "tensile": "Tensile Testing",
    "friction": "Friction Testing",
    "extrusion": "Extrusion",
}

# Where each pipeline's real results live, and which column holds the actual
# instrument test time (not films_pipeline_manifest.processed_at, which is
# when the pipeline ran, not when the test happened). This is what lets the
# digest catch a machine that has gone silent: files_processed can be 0 for
# a genuinely quiet week, but most_recent_test stalling is the signal that
# nothing is coming off the machine at all, the case a failure alert can't
# catch because there's no failed file to alert on.
RESULTS_TABLES = {
    "tensile": {
        "table": f"{PROJECT_ID}.films_tensile_london.films_tensile_results",
        "date_column": "timestamp_start",
    },
    "friction": {
        "table": f"{PROJECT_ID}.machine_data.films_friction_raw",
        "date_column": "timestamp_start",
    },
    "extrusion": {
        "table": f"{PROJECT_ID}.machine_collin_e25e.raw_films_extrusion",
        "date_column": "date",
    },
}


def manifest_counts(bq, window_start):
    """files_processed, files_failed, specimens_ingested per pipeline over
    the digest window. Pipelines with no manifest activity in the window
    (the silence case) simply don't appear here; the caller fills zeros."""
    query = f"""
        SELECT
            pipeline,
            COUNTIF(status = 'success') AS files_processed,
            COUNTIF(status = 'failed') AS files_failed,
            SUM(IF(status = 'success', rows_inserted, 0)) AS specimens_ingested
        FROM `{MANIFEST_TABLE}`
        WHERE processed_at >= @window_start
        GROUP BY pipeline
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("window_start", "TIMESTAMP", window_start),
    ])
    rows = bq.query(query, job_config=job_config).result()
    return {r["pipeline"]: r for r in rows}


def failed_files(bq, pipeline, window_start):
    """Individual failed files in the window, most recent first, capped at
    FAILED_FILES_LIMIT so a backlog spike can't produce an unreadable email;
    the email says how many more there are if the cap is hit."""
    query = f"""
        SELECT source_file, error_message, processed_at
        FROM `{MANIFEST_TABLE}`
        WHERE pipeline = @pipeline
          AND status = 'failed'
          AND processed_at >= @window_start
        ORDER BY processed_at DESC
        LIMIT {FAILED_FILES_LIMIT + 1}
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("pipeline", "STRING", pipeline),
        bigquery.ScalarQueryParameter("window_start", "TIMESTAMP", window_start),
    ])
    return list(bq.query(query, job_config=job_config).result())


def most_recent_test(bq, pipeline):
    """MAX() of the real test-time column in that pipeline's results table.
    Not windowed: this should reflect the true most recent test regardless
    of when the digest last ran, so a stale value stays visibly stale.
    Returns the raw value (a datetime, a date, or None) so the caller can
    both format it for display and compare it for staleness."""
    config = RESULTS_TABLES[pipeline]
    query = f"SELECT MAX({config['date_column']}) AS most_recent FROM `{config['table']}`"
    rows = list(bq.query(query).result())
    return rows[0]["most_recent"] if rows else None


def format_test_date(value):
    if value is None:
        return "none on record"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    return value.strftime("%Y-%m-%d")


def is_stale(value, window_end):
    if value is None:
        return True
    if isinstance(value, datetime):
        age = window_end - value
    else:
        age = window_end.date() - value
    return age > timedelta(days=STALE_TEST_THRESHOLD_DAYS)


LOGS_URL = f"https://console.cloud.google.com/functions/details/europe-west2/films-pipeline-digest?project={PROJECT_ID}&tab=logs"


def build_email(pipeline, pipeline_readable, files_processed, files_failed,
                 specimens_ingested, window_start, window_end,
                 recent_test_value, failures):
    recent_test_display = format_test_date(recent_test_value)
    stale = is_stale(recent_test_value, window_end)

    second_look = []
    if files_failed > 0:
        second_look.append(
            "Files failed is above zero. You should already have had a "
            "separate failure alert for each one; this is just a weekly "
            "roundup."
        )
    if stale:
        second_look.append(
            f"Most recent test ({escape(recent_test_display)}) is more than "
            f"{STALE_TEST_THRESHOLD_DAYS} days old. This pipeline may have "
            "gone quiet without any file ever reaching the failed-processing "
            "folder, which a failure alert cannot catch. Worth checking "
            "whether the machine or the upload step has stalled."
        )

    body = email_style.section_header("Weekly summary")
    body += email_style.paragraph(
        f"Covering {window_start.strftime('%Y-%m-%d')} to {window_end.strftime('%Y-%m-%d')}."
    )
    body += email_style.key_value_table([
        ("Files processed successfully", files_processed),
        ("Files failed", files_failed),
        ("Specimens ingested", specimens_ingested),
        ("Most recent test on record", escape(recent_test_display)),
    ])
    body += email_style.muted_note(
        "This email arrives every Friday morning regardless of whether "
        "anything went wrong, so a quiet week with nothing to report is "
        "expected and not a problem on its own."
    )

    if failures:
        body += email_style.divider()
        body += email_style.section_header("Failed files this week")
        rows = [
            (escape(f["source_file"]), f["processed_at"].strftime("%Y-%m-%d %H:%M UTC"))
            for f in failures[:FAILED_FILES_LIMIT]
        ]
        body += email_style.data_table(["File", "Time"], rows, font_size=14)
        if len(failures) > FAILED_FILES_LIMIT:
            body += email_style.muted_note(
                f"...and {len(failures) - FAILED_FILES_LIMIT} more. See the technical section below for the rest."
            )

    if second_look:
        body += email_style.divider()
        body += email_style.section_header("Worth a second look")
        for note in second_look:
            body += email_style.paragraph(note)

    body += email_style.divider()
    body += email_style.section_header("Technical details")
    body += email_style.key_value_table([
        ("Internal pipeline name", escape(pipeline)),
        ("Window (UTC)", f"{window_start.isoformat()} to {window_end.isoformat()}"),
    ])
    if failures:
        tech_rows = [
            (
                escape(f["source_file"]),
                f["processed_at"].strftime("%Y-%m-%d %H:%M:%S UTC"),
                escape(f["error_message"] or ""),
            )
            for f in failures[:FAILED_FILES_LIMIT]
        ]
        body += email_style.data_table(["source_file", "processed_at", "error_message"], tech_rows)
    body += email_style.cta_link("View pipeline logs", LOGS_URL)

    html = email_style.wrap_email("Hello,", body)

    if files_failed > 0:
        subject = f"[Digest] {pipeline_readable}: {files_failed} file(s) failed this week"
    else:
        subject = f"[Digest] {pipeline_readable}: all clear"

    return subject, html


@functions_framework.http
def send_weekly_digest(request):
    bq = bigquery.Client(project=PROJECT_ID)

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=7)
    counts = manifest_counts(bq, window_start)

    results = {}
    errors = {}
    for pipeline, pipeline_readable in PIPELINE_READABLE.items():
        row = counts.get(pipeline)
        files_processed = row["files_processed"] if row else 0
        files_failed = row["files_failed"] if row else 0
        specimens_ingested = (row["specimens_ingested"] or 0) if row else 0
        recent_test_value = most_recent_test(bq, pipeline)
        failures = failed_files(bq, pipeline, window_start) if files_failed else []

        subject, html = build_email(
            pipeline, pipeline_readable, files_processed, files_failed,
            specimens_ingested, window_start, window_end, recent_test_value,
            failures,
        )

        try:
            send_result = gmail_sender.send_html_email(
                PROJECT_ID, DIGEST_RECIPIENT, subject, html,
            )
            print(f"DIGEST_SENT pipeline={pipeline} message_id={send_result.get('id')}")
            results[pipeline] = {
                "files_processed": files_processed,
                "files_failed": files_failed,
                "specimens_ingested": specimens_ingested,
                "most_recent_test": format_test_date(recent_test_value),
                "gmail_message_id": send_result.get("id"),
            }
        except Exception as exc:
            print(f"DIGEST_SEND_FAILED pipeline={pipeline} error={exc}")
            errors[pipeline] = str(exc)

    print(f"DIGEST_RUN_SUMMARY {results}")
    if errors:
        return {"sent": results, "errors": errors}, 500
    return {"sent": results}
