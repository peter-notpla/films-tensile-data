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
"""

from google.cloud import bigquery

DEFAULT_WINDOW_MINUTES = 30


def find_specimen_link(bq_client, table_id, gcs_created_at, window_minutes=DEFAULT_WINDOW_MINUTES):
    """table_id: fully-qualified `project.dataset.table` for the pipeline's
    own *_all_revisions table. Only matches against row_state = 'current'
    rows, so a curve point links to the authoritative specimen, not an
    archived/superseded revision (see CLAUDE.md's "row_state"/
    "database_revision" section).

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
          AND ABS(TIMESTAMP_DIFF(timestamp_start, @gcs_created_at, SECOND)) <= @window_seconds
        ORDER BY delta_seconds ASC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("gcs_created_at", "TIMESTAMP", gcs_created_at),
        bigquery.ScalarQueryParameter("window_seconds", "INT64", window_seconds),
    ])
    rows = list(bq_client.query(query, job_config=job_config).result())
    if not rows:
        return None, None
    return rows[0]["specimen_key"], rows[0]["delta_seconds"]
