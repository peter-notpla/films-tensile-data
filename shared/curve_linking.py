"""Best-effort linking of a raw curve file back to the specimen it belongs
to. Not pure (queries BigQuery live), same reasoning as
shared/revision_handling.py for keeping this out of curve_parser.py.

The raw curve file carries no pellet ID, extrusion ID, or real test
timestamp - only a bare sample number in its filename, which CLAUDE.md's
"Sample numbers are not stable identifiers" section already establishes
can't be trusted as a join key (confirmed again here: real raw files exist
with sample numbers that have no matching row in the summary table at all).
So linking goes by time proximity instead: the raw file's GCS creation
time as a proxy for when the test happened, matched against the nearest
`timestamp_start` in the summary table, within a generous window. Never
blocking: if nothing is found within the window, the caller stores NULL
rather than failing the file.

**template_name is now a required match, added 4 September 2026.** The
curve filename and the results row both carry the VectorPro template name
(`raw-<template>-sample-<n>.csv` vs. row 1 of the results CSV), and they
match cleanly on every real file checked. Requiring it alongside nearest-
time is a pure precision improvement for live traffic (rules out matching
a `TensileTest-Films[WIP](V1)` curve to a same-minute `TensileTest-
Films(V1)` specimen) with no coverage cost, since a live-triggered file's
GCS creation time already closely tracks its real test time.

**Does not help the historical backfill, and cannot be made to**: the
per-file GCS creation timestamp this function needs is only meaningful for
a file still sitting at its original upload path. `move_blob` (in each
raw-processor's main.py) moves a successfully-processed file via
`copy_blob` + `delete`, which gives the copy a new object generation and
therefore a new `time_created` reflecting the move, not the original
upload - confirmed on a real example 4 September 2026 (`sample-30.csv`:
`link_time_delta_seconds` recorded 69 seconds at ingest time; the same
file's current blob metadata, in the processed folder, implies a nearest
same-template candidate over 4,000 minutes away). No audit-log trail of
the original creation event exists either (checked: no matching
`storage.objects.create` entries for real filenames). The original signal
this function needs is gone for any file already moved, so widening the
window cannot recover historical coverage - only going-forward accuracy.
"""

from google.cloud import bigquery

DEFAULT_WINDOW_MINUTES = 30


def find_specimen_link(bq_client, table_id, gcs_created_at, template_name, window_minutes=DEFAULT_WINDOW_MINUTES):
    """table_id: fully-qualified `project.dataset.table` for the pipeline's
    own *_all_revisions table. Only matches against row_state = 'current'
    rows, so a curve point links to the authoritative specimen, not an
    archived/superseded revision (see CLAUDE.md's "row_state"/
    "database_revision" section).

    template_name: the curve file's own template (from
    shared/curve_parser.parse_filename), matched case-insensitively against
    the results table's template_name. A curve file with no comparable
    template match simply won't link, same as one outside the time window.

    Returns (specimen_key, time_delta_seconds) - both None if nothing is
    within the window. time_delta_seconds is always non-negative and is
    returned even on a match, so confidence is judgeable later rather than
    collapsed into a single boolean (a 12-second delta and a 28-minute
    delta are not the same confidence).
    """
    window_seconds = window_minutes * 60
    query = f"""
        SELECT
            specimen_key,
            ABS(TIMESTAMP_DIFF(timestamp_start, @gcs_created_at, SECOND)) AS delta_seconds
        FROM `{table_id}`
        WHERE row_state = 'current'
          AND LOWER(TRIM(template_name)) = LOWER(TRIM(@template_name))
          AND ABS(TIMESTAMP_DIFF(timestamp_start, @gcs_created_at, SECOND)) <= @window_seconds
        ORDER BY delta_seconds ASC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("gcs_created_at", "TIMESTAMP", gcs_created_at),
        bigquery.ScalarQueryParameter("template_name", "STRING", template_name),
        bigquery.ScalarQueryParameter("window_seconds", "INT64", window_seconds),
    ])
    rows = list(bq_client.query(query, job_config=job_config).result())
    if not rows:
        return None, None
    return rows[0]["specimen_key"], rows[0]["delta_seconds"]
