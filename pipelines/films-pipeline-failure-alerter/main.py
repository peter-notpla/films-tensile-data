import io
import json
import os
import uuid
from datetime import date, datetime, timezone
from html import escape

import functions_framework
import pandas as pd
from google.cloud import bigquery
from google.cloud import storage

from shared import email_style, gmail_sender

PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BUCKET = os.environ.get("BUCKET", "notpla-machine-data")

LOGS_URL = f"https://console.cloud.google.com/functions/details/europe-west2/films-pipeline-failure-alerter?project={PROJECT_ID}&tab=logs"

# Where each route's alert actually gets sent. "default" catches files with
# no identifiable owner, or from a pipeline (extrusion) that doesn't record
# initials at all - see resolve_route().
ROUTE_EMAILS = {
    "katie": "katie@notpla.com",
    "emily": "emily@notpla.com",
    "default": "peter@notpla.com",
}

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
    "tensile_raw": (
        "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
        "tensiletester-films-tensile-raw-samples-failed-processing/"
    ),
    "friction_raw": (
        "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/"
        "tensiletester-films-friction-failed-processing/"
        "tensiletester-films-friction-failed-processing-raw/"
    ),
}

INITIALS_COLUMN_HINT = "user initials"

PIPELINE_READABLE = {
    "tensile": "Tensile Testing",
    "friction": "Friction Testing",
    "extrusion": "Extrusion",
    "tensile_raw": "Tensile Raw Curve Data",
    "friction_raw": "Friction Raw Curve Data",
}

# ---------------------------------------------------------------------
# Per-row flag + rescue (added 8 September 2026). films_pipeline_row_errors
# already captured hard-rejected rows (never inserted); it's now also the
# home for "flagged" rows - rows that DID get inserted, but with
# validation_status != 'valid' (shared/id_validation.py is deliberately
# flag-don't-reject for malformed pellet/extrusion IDs, so a typo never
# shows up as a rejection - only a manual anomaly scan ever caught these
# before). Both categories share one table and one alert path; see
# ROW_ERRORS_TABLE below and shared/row_rescue.py for the reconstruction
# logic the rescue Cloud Function uses once a row's link is clicked.
#
# Deliberately excludes the two raw-curve pipelines (tensile_raw,
# friction_raw): their "rows" are curve timepoints, not hand-entered
# metadata, so there's nothing there worth a human rescuing.
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"

# Insert-only, mirroring ALERTS_SENT_TABLE's dedup pattern (see
# find_new_failures above) - added 9 September 2026 to replace an `UPDATE
# ... SET alerted_at` on films_pipeline_row_errors, which failed every run:
# rows just written by find_new_flagged_rows (or by a csv-processor within
# the last ~90 minutes) sit in BigQuery's streaming buffer, which rejects
# UPDATE/DELETE outright. A SELECT/JOIN against the streaming buffer is
# fine, so tracking "already alerted" as insert-only rows here, checked via
# NOT EXISTS, sidesteps the restriction entirely instead of racing it.
ROW_ISSUE_ALERTS_SENT_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_issue_alerts_sent"

# Set once the films-pipeline-row-rescue Cloud Function is deployed and its
# URL is known (chicken-and-egg: it doesn't exist on this alerter's own
# first deploy). Row-issue emails go out without a working link until this
# is set - the row is still tracked and won't be re-emailed once it is.
ROW_RESCUE_URL = os.environ.get("ROW_RESCUE_URL", "")

FLAGGED_ROW_SOURCES = {
    "tensile": {
        "table": f"{PROJECT_ID}.films_tensile_london.films_tensile_results_all_revisions",
        "identity_column": "specimen_key",
        "has_row_state": True,
    },
    "friction": {
        "table": f"{PROJECT_ID}.machine_data.films_friction_raw_all_revisions",
        "identity_column": "specimen_key",
        "has_row_state": True,
    },
    "extrusion": {
        "table": f"{PROJECT_ID}.machine_collin_e25e.raw_films_extrusion",
        "identity_column": "key",
        "has_row_state": False,
    },
}


def _json_default(value):
    """Timestamps/dates must round-trip back through shared/row_rescue.py's
    CSV reconstruction in the exact format each parser's own format list
    accepts (e.g. '%Y-%m-%d %H:%M:%S') - Python's default str(datetime)
    includes a UTC offset the parsers don't try to match."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def find_new_flagged_rows(bq):
    """Scans each results table for validation_status != 'valid' rows not
    yet tracked in films_pipeline_row_errors, and inserts them there as
    category='flagged'. Returns the number inserted."""
    inserted = 0
    for pipeline, cfg in FLAGGED_ROW_SOURCES.items():
        row_state_clause = "AND row_state = 'current'" if cfg["has_row_state"] else ""
        query = f"""
            SELECT *
            FROM `{cfg['table']}`
            WHERE validation_status != 'valid'
              AND validation_status IS NOT NULL
              {row_state_clause}
              AND {cfg['identity_column']} IS NOT NULL
              AND {cfg['identity_column']} NOT IN (
                SELECT identity_key FROM `{ROW_ERRORS_TABLE}`
                WHERE pipeline = @pipeline AND category = 'flagged' AND identity_key IS NOT NULL
              )
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("pipeline", "STRING", pipeline),
        ])
        rows_to_insert = []
        for row in bq.query(query, job_config=job_config).result():
            row_dict = dict(row.items())
            identity_key = row_dict[cfg["identity_column"]]
            source_file = row_dict.get("source_file") or ""
            rows_to_insert.append({
                "pipeline": pipeline,
                "source_file": source_file,
                "checksum": None,
                "row_number": None,
                "reason": f"flagged: {row_dict.get('validation_status')}",
                "raw_row": json.dumps(row_dict, default=_json_default),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "row_error_id": str(uuid.uuid4()),
                "category": "flagged",
                "identity_key": identity_key,
                "status": "open",
            })
        if rows_to_insert:
            errors = bq.insert_rows_json(ROW_ERRORS_TABLE, rows_to_insert)
            if errors:
                print(f"FLAGGED_ROW_INSERT_ERROR pipeline={pipeline} errors={errors}")
            else:
                inserted += len(rows_to_insert)
    return inserted


def find_unalerted_row_issues(bq):
    query = f"""
        SELECT row_error_id, pipeline, source_file, category, reason, raw_row
        FROM `{ROW_ERRORS_TABLE}` e
        WHERE status = 'open'
          AND NOT EXISTS (
            SELECT 1 FROM `{ROW_ISSUE_ALERTS_SENT_TABLE}` a
            WHERE a.row_error_id = e.row_error_id
          )
    """
    return list(bq.query(query).result())


def build_row_issue_bundle_email(row_issues):
    """One email per run covering every row flagged or rejected since the
    last run (added 8 September 2026, replacing one-email-per-row) - each
    row still gets its own rescue link, since a fix/discard action is
    inherently per-row, but the digest itself is a single email."""
    count = len(row_issues)
    rejected_count = sum(1 for i in row_issues if i["category"] == "rejected")
    flagged_count = count - rejected_count

    body = email_style.section_header("What happened?")
    parts = []
    if rejected_count:
        parts.append(
            f"{rejected_count} row{'s' if rejected_count != 1 else ''} rejected "
            f"outright (never reached BigQuery)"
        )
    if flagged_count:
        parts.append(
            f"{flagged_count} row{'s' if flagged_count != 1 else ''} already in "
            f"BigQuery but flagged with a malformed Pellet/Extrusion ID"
        )
    body += email_style.paragraph(" and ".join(parts) + ".")

    rows_html = []
    any_link_missing = False
    for issue in row_issues:
        pipeline_readable = PIPELINE_READABLE.get(issue["pipeline"], issue["pipeline"])
        filename = (issue["source_file"] or "").split("/")[-1] or "(no file)"
        rescue_url = f"{ROW_RESCUE_URL}?token={issue['row_error_id']}" if ROW_RESCUE_URL else None
        if rescue_url:
            link_html = f'<a href="{escape(rescue_url)}" style="color:{email_style.ACCENT}; font-weight:bold;">Fix / discard</a>'
        else:
            link_html = '<span style="color:#999;">(link not configured)</span>'
            any_link_missing = True
        rows_html.append([
            escape(pipeline_readable),
            escape(issue["category"]),
            f"<code>{escape(filename)}</code>",
            escape(issue["reason"] or ""),
            link_html,
        ])

    body += email_style.data_table(["Pipeline", "Type", "File", "Why", ""], rows_html)

    body += email_style.divider()
    body += email_style.muted_note(
        "Each link opens the full row, editable, with the option to resubmit "
        "a corrected version through the real pipeline (re-validated the same "
        "way any new file is) or discard the row entirely."
    )
    if any_link_missing:
        body += email_style.muted_note(
            "Some rows show '(link not configured)' - ROW_RESCUE_URL isn't set "
            "on this function yet. Those rows are tracked and won't be "
            "re-alerted once it is."
        )

    html = email_style.wrap_email("Hello,", body)
    subject = f"[Row flags] {count} row{'s' if count != 1 else ''} need a look"
    return subject, html



def find_new_failures(bq):
    # Only alert on a file that has NEVER succeeded (no 'success' row
    # exists anywhere in its manifest history) - not just "most recent
    # row is failed", because a duplicate GCS listing during a burst of
    # file moves can produce a late phantom 404 several minutes AFTER the
    # file's real success already landed and moved it out of the watch
    # folder (observed 30 August 2026: sample-205's real success logged at
    # 14:25:57, a phantom "no such object" failure for the same file at
    # 14:27:49 - the phantom row is chronologically latest but the file
    # was never actually broken). Excluding by "ever succeeded" rather
    # than "latest row" is what makes this immune to that ordering trap.
    #
    # A retry storm (that same incident) can also leave a file with many
    # failed rows before it eventually succeeds; alerting on every one of
    # them reports transient, self-resolved infra retries as if they were
    # real bad-data failures needing a human to fix a CSV. That's what
    # produced ~690 emails for a backlog where 1,243/1,244 files
    # ultimately landed fine - see pipeline-roadmap.md's Phase 5 entry.
    #
    # checksum is NULL whenever a file fails before it can be downloaded
    # (e.g. a GCS read timeout) - `a.checksum = m.checksum` is never true
    # for two NULLs, so those rows looked "never alerted" on every run and
    # re-alerted (and re-escalated) hourly forever. source_file plus a
    # null-safe checksum match is the real identity of "have we alerted on
    # this exact failure before".
    query = f"""
        WITH ever_succeeded AS (
            SELECT DISTINCT pipeline, source_file
            FROM `{MANIFEST_TABLE}`
            WHERE status = 'success'
        )
        SELECT m.pipeline, m.source_file, m.checksum, m.error_message, m.processed_at
        FROM `{MANIFEST_TABLE}` m
        LEFT JOIN ever_succeeded s
          ON s.pipeline = m.pipeline AND s.source_file = m.source_file
        WHERE m.status = 'failed'
          AND s.source_file IS NULL
          AND NOT EXISTS (
            SELECT 1 FROM `{ALERTS_SENT_TABLE}` a
            WHERE a.pipeline = m.pipeline
              AND a.source_file = m.source_file
              AND a.checksum IS NOT DISTINCT FROM m.checksum
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


def build_failure_bundle_email(route, items):
    """One email per route per run (added 8 September 2026, replacing the
    former one-email-per-file design) - items is every file that failed
    and routed here in this run. Raw+summary files land together in a
    batch (Peter's description), so with an hourly poll this naturally
    bundles a whole upload batch into one email instead of several."""
    count = len(items)
    escalations = [it for it in items if it["is_escalation"]]

    body = email_style.section_header("What happened?")
    summary_text = f"{count} file{'s' if count != 1 else ''} failed automated processing this run."
    if escalations:
        summary_text += (
            f" {len(escalations)} of these failed before too, under the same "
            f"name - see 'Needs a closer look' below."
        )
    body += email_style.paragraph(summary_text)

    body += email_style.data_table(
        ["Pipeline", "File", "Time", "Error"],
        [
            [
                escape(it["pipeline_readable"]),
                f"<code>{escape(it['filename'])}</code>" + (" ⚠️" if it["is_escalation"] else ""),
                escape(it["failed_at"]),
                escape(it["error_message"]),
            ]
            for it in items
        ],
    )

    body += email_style.divider()
    body += email_style.section_header("How do I fix it?")
    if route == "default":
        body += email_style.paragraph(
            '1. Find each file above in the "Uploaded" folder on the lab computer.<br>'
            "2. Check it for anything obviously wrong (missing headers, extra blank "
            'rows at the top, wrong export settings), fix it, then move it back into '
            'the "To Be Uploaded" folder, keeping the file name exactly the same. '
            "It'll be picked up and reprocessed automatically.<br>"
            "3. Not sure what's wrong, or don't have access to fix it? Leave the "
            "file where it is until someone with access can check it."
        )
        body += email_style.muted_note(
            "These came to you by default: either the file had no identifiable "
            "owner, or it's from the Extrusion pipeline, which doesn't record "
            "initials."
        )
    else:
        body += email_style.paragraph(
            '1. Find each file above in the "Uploaded" folder on the lab computer.<br>'
            "2. Check it for anything obviously wrong (missing headers, extra blank "
            'rows at the top, wrong export settings), fix it, then move it back into '
            'the "To Be Uploaded" folder, keeping the file name exactly the same. '
            "It'll be picked up and reprocessed automatically.<br>"
            "3. Not sure what's wrong, or don't have access to fix it? Forward "
            "this email to peter@notpla.com and leave the file where it is."
        )
        body += email_style.muted_note(
            "These were routed to you because each file's User Initials matched yours."
        )

    if escalations:
        body += email_style.divider()
        body += email_style.section_header("Needs a closer look")
        body += email_style.paragraph(
            "These failed, were apparently reprocessed under the same name, and "
            "failed again - the usual self-serve fix didn't take."
        )
        body += email_style.data_table(
            ["File", "Previous failure", "Previous error"],
            [
                [
                    f"<code>{escape(it['filename'])}</code>",
                    escape(it["previous_failed_at"] or ""),
                    escape(it["previous_error"] or ""),
                ]
                for it in escalations
            ],
        )

    body += email_style.cta_link("View pipeline logs", LOGS_URL)

    html = email_style.wrap_email("Hello,", body)
    subject = f"[Alert] {count} file{'s' if count != 1 else ''} failed processing"
    return subject, html


def build_escalation_bundle_email(escalation_items):
    """Repeat failures always reach Peter directly, regardless of which
    route (Katie/Emily/default) their primary bundled alert went to -
    same guarantee the old per-file build_escalation_email gave, just
    bundled across everything that escalated this run instead of one
    email per file."""
    count = len(escalation_items)
    body = email_style.section_header("What happened?")
    body += email_style.paragraph(
        f"{count} file{'s' if count != 1 else ''} failed automated processing, "
        f"appear{'s' if count == 1 else ''} to have been reprocessed under the "
        f"same name, and failed again. The usual self-serve fix didn't take; "
        f"these need a human to look directly."
    )
    body += email_style.data_table(
        ["Pipeline", "File", "Latest failure", "Latest error", "Previous failure", "Previous error"],
        [
            [
                escape(it["pipeline_readable"]),
                f"<code>{escape(it['filename'])}</code>",
                escape(it["failed_at"]),
                escape(it["error_message"]),
                escape(it["previous_failed_at"] or ""),
                escape(it["previous_error"] or ""),
            ]
            for it in escalation_items
        ],
    )
    body += email_style.cta_link("View pipeline logs", LOGS_URL)

    html = email_style.wrap_email("Hello,", body)
    subject = f"[Alert] Repeat failure needs attention: {count} file{'s' if count != 1 else ''}"
    return subject, html


@functions_framework.http
def check_and_alert(request):
    bq = bigquery.Client(project=PROJECT_ID)

    failures = find_new_failures(bq)
    directory = load_directory(bq)

    # Resolve route/escalation per file first (needs per-file queries), but
    # group into one bundle per route so an upload batch (raw + summary
    # files landing together, per Peter) becomes one email per recipient
    # instead of one email per file. Escalations still always go to Peter
    # directly too, regardless of which route the primary bundle went to -
    # same guarantee the old per-file design gave.
    grouped_by_route = {}
    escalation_items = []

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

        previous = find_previous_failure(bq, pipeline, filename, row["processed_at"])
        is_escalation = previous is not None and previous["status"] == "failed"
        previous_failed_at = (
            (previous["processed_at"].strftime("%d %b %Y %H:%M UTC") if previous["processed_at"] else "unknown")
            if is_escalation else None
        )
        previous_error = (previous["error_message"] or "") if is_escalation else None

        item = {
            "pipeline": pipeline, "pipeline_readable": pipeline_readable,
            "source_file": source_file, "checksum": checksum, "filename": filename,
            "failed_at": failed_at, "error_message": error_message,
            "route_reason": route_reason,
            "is_escalation": is_escalation,
            "previous_failed_at": previous_failed_at, "previous_error": previous_error,
        }
        grouped_by_route.setdefault(route, []).append(item)
        if is_escalation:
            escalation_items.append(item)

    alerted = 0
    dedup_write_failed = 0

    for route, items in grouped_by_route.items():
        recipient = ROUTE_EMAILS.get(route)
        if recipient is None:
            print(f"PIPELINE_FAILURE_UNKNOWN_ROUTE route={route}, falling back to default")
            recipient = ROUTE_EMAILS["default"]

        subject, html = build_failure_bundle_email(route, items)
        try:
            send_result = gmail_sender.send_html_email(PROJECT_ID, recipient, subject, html)
            print(
                f"PIPELINE_FAILURE_BUNDLE_SENT route={route} count={len(items)} "
                f"message_id={send_result.get('id')}"
            )
        except Exception as exc:
            # Not marked as sent below, so every file in this bundle is
            # retried next hour rather than silently never alerting.
            print(f"PIPELINE_FAILURE_BUNDLE_SEND_FAILED route={route} count={len(items)} error={exc}")
            dedup_write_failed += len(items)
            continue

        try:
            now = datetime.now(timezone.utc).isoformat()
            errors = bq.insert_rows_json(
                ALERTS_SENT_TABLE,
                [{
                    "pipeline": it["pipeline"], "source_file": it["source_file"],
                    "checksum": it["checksum"], "route": route,
                    "route_reason": it["route_reason"], "sent_at": now,
                } for it in items],
            )
            if errors:
                print(f"ALERT_DEDUP_WRITE_FAILURE route={route} errors={errors}")
                dedup_write_failed += len(items)
                continue
        except Exception as exc:
            # Not marked as sent, so this bundle's files are retried next
            # hour instead of silently never being recorded.
            print(f"ALERT_DEDUP_WRITE_FAILURE route={route} error={exc}")
            dedup_write_failed += len(items)
            continue

        alerted += len(items)

    if escalation_items:
        esc_subject, esc_html = build_escalation_bundle_email(escalation_items)
        try:
            esc_result = gmail_sender.send_html_email(PROJECT_ID, "peter@notpla.com", esc_subject, esc_html)
            print(
                f"PIPELINE_FAILURE_ESCALATION_BUNDLE_SENT count={len(escalation_items)} "
                f"message_id={esc_result.get('id')}"
            )
        except Exception as exc:
            # Best-effort: the primary bundles above already went out (or
            # were retried above independently), so a failed escalation
            # send doesn't block anything else in this run.
            print(f"PIPELINE_FAILURE_ESCALATION_BUNDLE_SEND_FAILED count={len(escalation_items)} error={exc}")

    flagged_inserted = find_new_flagged_rows(bq)
    if flagged_inserted:
        print(f"FLAGGED_ROWS_INSERTED count={flagged_inserted}")

    row_issues = find_unalerted_row_issues(bq)
    row_alerted = 0
    if row_issues:
        subject, html = build_row_issue_bundle_email(row_issues)
        try:
            send_result = gmail_sender.send_html_email(PROJECT_ID, "peter@notpla.com", subject, html)
            print(f"ROW_ISSUE_BUNDLE_SENT count={len(row_issues)} message_id={send_result.get('id')}")
        except Exception as exc:
            # Not marked alerted below, so every row here is retried next
            # hour instead of silently never alerting because the send
            # failed.
            print(f"ROW_ISSUE_BUNDLE_SEND_FAILED count={len(row_issues)} error={exc}")
        else:
            try:
                now = datetime.now(timezone.utc).isoformat()
                errors = bq.insert_rows_json(
                    ROW_ISSUE_ALERTS_SENT_TABLE,
                    [{"row_error_id": issue["row_error_id"], "sent_at": now} for issue in row_issues],
                )
                if errors:
                    print(f"ROW_ISSUE_ALERTS_SENT_WRITE_FAILED errors={errors}")
                else:
                    row_alerted = len(row_issues)
            except Exception as exc:
                # Alert already sent but not marked - worst case this batch
                # gets a duplicate email next hour, far safer than silently
                # never being alerted on again.
                print(f"ROW_ISSUE_ALERTS_SENT_WRITE_FAILED error={exc}")

    summary = {
        "checked": len(failures), "alerted": alerted, "dedup_write_failed": dedup_write_failed,
        "escalations_bundled": len(escalation_items),
        "flagged_rows_inserted": flagged_inserted,
        "row_issues_checked": len(row_issues), "row_issues_alerted": row_alerted,
    }
    print(f"ALERT_RUN_SUMMARY {json.dumps(summary)}")
    return summary
