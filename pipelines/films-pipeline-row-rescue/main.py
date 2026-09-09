"""Row rescue: the human side of the per-row flag system (8 September
2026). films-pipeline-failure-alerter emails a link for every row in
films_pipeline_row_errors; this function serves what's behind that link -
view the row, edit it or discard it, and (if edited) resubmit it through
the exact same validation the real pipeline uses.

Access control is deliberately not implemented here: this function is
meant to sit behind Identity-Aware Proxy, restricted to peter@notpla.com,
which supplies the caller's verified identity via
X-Goog-Authenticated-User-Email. See pipeline-roadmap.md's 8 September
"per-row flag + rescue" entry for the IAP setup steps and why they're a
one-time manual step rather than something this code does.

Landing a correction differs by pipeline - see handle_extrusion_flagged
below for why extrusion (no revision_handling.py) is the one real
exception to "just drop a corrected CSV in the watch folder."
"""

import json
import os
from html import escape

import functions_framework
from google.cloud import bigquery, storage

from shared.extrusion_parser import TABLE_COLUMNS as EXTRUSION_TABLE_COLUMNS
from shared.id_validation import validation_status
from shared.row_rescue import revalidate_and_rebuild, strip_derived_fields

PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BUCKET = os.environ.get("BUCKET", "notpla-machine-data")
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"
EXTRUSION_TABLE = f"{PROJECT_ID}.machine_collin_e25e.raw_films_extrusion"

# Lifted from each pipeline's deployed env vars - same pattern as
# films-pipeline-failure-alerter's FAILED_PREFIXES. Only tensile/friction
# actually get used (both have shared/revision_handling.py, so dropping a
# corrected CSV into the real watch folder is correct for both a rejected
# row and a flagged one - it either inserts for the first time or
# supersedes the flagged row as a new revision). Extrusion's rejected rows
# also use this; its flagged rows do not - see handle_extrusion_flagged.
WATCH_PREFIXES = {
    "tensile": "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/tensiletester-films-tensile-summary-tables/",
    "friction": "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/tensiletester-films-friction-to-be-processed/tensiletester-films-friction-to-be-processed-summary/",
    "extrusion": "machine-collin-e25e/machine-collin-e25e-to-be-processed/",
}

# Fields that identify a specimen for tensile/friction's revision_handling
# (specimen_key = machine|pipeline|timestamp_minute|sample). Locked
# read-only on a FLAGGED-row rescue form: editing either would build a
# *different* specimen_key, so the corrected row would land as a brand new
# specimen instead of superseding the flagged one - the live bad row would
# stay live, silently, alongside a new correct-looking one. A REJECTED row
# has no live specimen to supersede, so no such risk - not locked there.
IDENTITY_LOCKED_FIELDS = {"sample", "timestamp_start"}

# Only these two are ever corrected via the extrusion-flagged direct-UPDATE
# path (see handle_extrusion_flagged) - deliberately narrow, matching the
# actual motivating case (a typo'd ID), not a general-purpose field editor
# against a table with no revision history to fall back on.
EXTRUSION_FLAGGED_EDITABLE = {"pellet_id", "extrusion_id"}

bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)


def get_actor_email(request):
    """IAP injects the verified caller's identity in this header once
    enabled in front of this function - 'accounts.google.com:email'.
    Falls back to 'unknown' rather than failing outright so a rescue made
    before IAP is fully wired up is still attributed to *something*."""
    header = request.headers.get("X-Goog-Authenticated-User-Email", "")
    return header.split(":", 1)[-1] if header else "unknown"


def lookup_row_error(token):
    query = f"SELECT * FROM `{ROW_ERRORS_TABLE}` WHERE row_error_id = @token"
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("token", "STRING", token),
    ])
    rows = list(bq_client.query(query, job_config=job_config).result())
    return dict(rows[0].items()) if rows else None


def mark_resolved(token, status, actor, rescued_source_file):
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("token", "STRING", token),
        bigquery.ScalarQueryParameter("status", "STRING", status),
        bigquery.ScalarQueryParameter("actor", "STRING", actor),
        bigquery.ScalarQueryParameter("rescued_source_file", "STRING", rescued_source_file),
    ])
    bq_client.query(
        f"""
        UPDATE `{ROW_ERRORS_TABLE}`
        SET status = @status, rescued_at = CURRENT_TIMESTAMP(), rescued_by = @actor,
            rescued_source_file = @rescued_source_file
        WHERE row_error_id = @token
        """,
        job_config=job_config,
    ).result()


def page(title, body_html):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 640px; margin: 40px auto; color: #222; padding: 0 16px; }}
  h1 {{ color: #E8623A; font-size: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
  td.label {{ font-weight: bold; width: 40%; color: #555; }}
  input[type=text] {{ width: 100%; padding: 4px 6px; box-sizing: border-box; }}
  input[readonly] {{ background: #f5f5f5; color: #888; }}
  button {{ padding: 8px 16px; margin-right: 8px; cursor: pointer; font-size: 14px; }}
  .resubmit {{ background: #E8623A; color: white; border: none; border-radius: 4px; }}
  .discard {{ background: #eee; border: 1px solid #ccc; border-radius: 4px; }}
  .error {{ background: #fdecea; border: 1px solid #f5c6cb; padding: 10px; border-radius: 4px; margin-bottom: 16px; }}
  .note {{ color: #777; font-size: 13px; }}
</style></head>
<body><h1>{escape(title)}</h1>{body_html}</body></html>"""


def render_message(title, message):
    return page(title, f"<p>{escape(message)}</p>")


def render_form(row, form_error=None, submitted_values=None):
    raw_row = json.loads(row["raw_row"]) if row["raw_row"] else {}
    fields = strip_derived_fields(raw_row)
    values = submitted_values if submitted_values is not None else {
        k: ("" if v is None else str(v)) for k, v in fields.items()
    }

    locked = row["category"] == "flagged" and row["pipeline"] in ("tensile", "friction")

    def field_row(k):
        val = values.get(k, "")
        if locked and k in IDENTITY_LOCKED_FIELDS:
            return (
                f'<tr><td class="label">{escape(k)}</td>'
                f'<td><input type="text" name="field__{escape(k)}" value="{escape(val)}" readonly>'
                f'<div class="note">locked - identifies this specimen; edit this and the fix '
                f'will create a new specimen instead of correcting this one</div></td></tr>'
            )
        return (
            f'<tr><td class="label">{escape(k)}</td>'
            f'<td><input type="text" name="field__{escape(k)}" value="{escape(val)}"></td></tr>'
        )

    rows_html = "".join(field_row(k) for k in fields.keys())
    error_html = f'<div class="error">{escape(form_error)}</div>' if form_error else ""

    extrusion_flagged_note = ""
    if row["pipeline"] == "extrusion" and row["category"] == "flagged":
        extrusion_flagged_note = (
            '<p class="note">This pipeline has no revision history, so only '
            '<b>Pellet ID</b> and <b>Extrusion ID</b> can be corrected here - '
            'other fields are shown for context only and will not be changed.</p>'
        )

    body = f"""
    {error_html}
    <p><b>Pipeline:</b> {escape(row['pipeline'])}<br>
       <b>Category:</b> {escape(row['category'])}<br>
       <b>Why:</b> {escape(row['reason'] or '')}<br>
       <b>File:</b> <code>{escape((row['source_file'] or '').split('/')[-1])}</code></p>
    {extrusion_flagged_note}
    <form method="post">
      <input type="hidden" name="token" value="{escape(row['row_error_id'])}">
      <table>{rows_html}</table>
      <button class="resubmit" name="action" value="resubmit" type="submit">Resubmit corrected row</button>
      <button class="discard" name="action" value="discard" type="submit">Discard this row</button>
    </form>
    <p class="note">Resubmitting re-runs this row through the exact same
    validation the real pipeline uses - an invalid fix is rejected right
    here, nothing gets written.</p>
    """
    return page(f"Rescue row - {row['pipeline']}", body)


def handle_extrusion_flagged_resubmit(row, original_fields, edited, actor):
    pellet_id = edited.get("pellet_id", original_fields.get("pellet_id"))
    extrusion_id = edited.get("extrusion_id", original_fields.get("extrusion_id"))
    new_status = validation_status(pellet_id, extrusion_id)
    if new_status != "valid":
        return render_form(
            row,
            form_error=f"Pellet/Extrusion ID still doesn't match the required format ({new_status}).",
            submitted_values=edited,
        ), 200

    identity_key = row["identity_key"]
    if not identity_key:
        return render_form(
            row,
            form_error="This row has no stable key (extrusion's 'Key' column was blank) - cannot update it safely.",
            submitted_values=edited,
        ), 200

    changed = {}
    for k in EXTRUSION_FLAGGED_EDITABLE:
        original_val = original_fields.get(k)
        original_str = "" if original_val is None else str(original_val)
        if k in edited and edited[k] != original_str:
            changed[k] = edited[k]

    if not changed:
        return render_form(row, form_error="Pellet ID / Extrusion ID were not changed.", submitted_values=edited), 200

    set_clause = ", ".join(f"{col} = @val_{i}" for i, col in enumerate(changed.keys()))
    params = [
        bigquery.ScalarQueryParameter(f"val_{i}", "STRING", val)
        for i, val in enumerate(changed.values())
    ]
    params.append(bigquery.ScalarQueryParameter("key", "STRING", identity_key))
    params.append(bigquery.ScalarQueryParameter("source_file", "STRING", row["source_file"]))

    job = bq_client.query(
        f"UPDATE `{EXTRUSION_TABLE}` SET {set_clause} WHERE key = @key AND source_file = @source_file",
        job_config=bigquery.QueryJobConfig(query_parameters=params),
    )
    job.result()
    if job.num_dml_affected_rows != 1:
        return render_form(
            row,
            form_error=(
                f"Update matched {job.num_dml_affected_rows} row(s), expected exactly 1 - "
                f"nothing was changed. Check manually before retrying."
            ),
            submitted_values=edited,
        ), 200

    mark_resolved(
        token=row["row_error_id"], status="rescued", actor=actor,
        rescued_source_file=f"direct BigQuery UPDATE: {EXTRUSION_TABLE}",
    )
    return render_message(
        "Updated",
        "The live extrusion row was corrected directly - this pipeline has "
        "no revision history to supersede instead.",
    ), 200


@functions_framework.http
def rescue(request):
    token = request.values.get("token")
    if not token:
        return render_message("Missing token", "No token provided."), 400

    row = lookup_row_error(token)
    if row is None:
        return render_message("Not found", "No row found for this link. It may have already been cleared."), 404

    if row["status"] != "open":
        detail = f" (already marked '{row['status']}'"
        detail += f" by {row['rescued_by']})" if row.get("rescued_by") else ")"
        return render_message("Already handled", f"This row was already handled{detail}.")

    if request.method == "GET":
        return render_form(row)

    action = request.form.get("action")
    actor = get_actor_email(request)

    if action == "discard":
        mark_resolved(token, status="dismissed", actor=actor, rescued_source_file=None)
        return render_message("Discarded", "This row has been marked as discarded. No data was changed.")

    if action != "resubmit":
        return render_message("Unknown action", "Unrecognised form action."), 400

    raw_row = json.loads(row["raw_row"]) if row["raw_row"] else {}
    original_fields = strip_derived_fields(raw_row)
    edited = {k: request.form.get(f"field__{k}", "") for k in original_fields.keys()}

    pipeline = row["pipeline"]
    category = row["category"]

    if pipeline == "extrusion" and category == "flagged":
        return handle_extrusion_flagged_resubmit(row, original_fields, edited, actor)

    ok, error, csv_bytes, new_validation_status = revalidate_and_rebuild(pipeline, edited)
    if not ok:
        return render_form(row, form_error=f"Not resubmitted: {error}", submitted_values=edited)

    if category == "flagged" and new_validation_status != "valid":
        return render_form(
            row,
            form_error=(
                f"Row now parses, but is still flagged ({new_validation_status}) - "
                f"fix the ID fully before resubmitting, or discard instead."
            ),
            submitted_values=edited,
        )

    watch_prefix = WATCH_PREFIXES[pipeline]
    original_filename = (row["source_file"] or "row").split("/")[-1]
    dest_name = f"{watch_prefix}rescued-{row['row_error_id']}-{original_filename}"
    storage_client.bucket(BUCKET).blob(dest_name).upload_from_string(csv_bytes, content_type="text/csv")

    mark_resolved(token, status="rescued", actor=actor, rescued_source_file=f"gs://{BUCKET}/{dest_name}")
    return render_message(
        "Resubmitted",
        f"Corrected row uploaded to the real pipeline as gs://{BUCKET}/{dest_name} - "
        f"it will be processed on the normal trigger, same as any new file.",
    )
