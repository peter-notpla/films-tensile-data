import os
from datetime import datetime, timedelta, timezone

import functions_framework
from google.cloud import bigquery

PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"

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


def most_recent_test(bq, pipeline):
    """MAX() of the real test-time column in that pipeline's results table.
    Not windowed: this should reflect the true most recent test regardless
    of when the digest last ran, so a stale value stays visibly stale.
    Formatted as a single space-free token (ISO-ish) so the log-based
    metric's label extractors can bound it with [^ ]+ like every other
    field, rather than needing a lookahead to a following key name."""
    config = RESULTS_TABLES[pipeline]
    query = f"SELECT MAX({config['date_column']}) AS most_recent FROM `{config['table']}`"
    rows = list(bq.query(query).result())
    value = rows[0]["most_recent"] if rows else None
    if value is None:
        return "none"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%dT%H:%MZ")
    return value.strftime("%Y-%m-%d")


@functions_framework.http
def send_weekly_digest(request):
    bq = bigquery.Client(project=PROJECT_ID)

    window_end = datetime.now(timezone.utc)
    window_start = window_end - timedelta(days=7)
    counts = manifest_counts(bq, window_start)

    summary = {}
    for pipeline, pipeline_readable in PIPELINE_READABLE.items():
        row = counts.get(pipeline)
        files_processed = row["files_processed"] if row else 0
        files_failed = row["files_failed"] if row else 0
        specimens_ingested = (row["specimens_ingested"] or 0) if row else 0
        recent_test = most_recent_test(bq, pipeline)

        # pipeline_readable is the only multi-word field here and is bounded
        # by a lookahead to the next key, same convention as
        # PIPELINE_FAILURE_ALERT in films-pipeline-failure-alerter. Every
        # other field is a single space-free token so the metric's label
        # extractors can all use the simpler [^ ]+ form.
        print(
            f"PIPELINE_WEEKLY_DIGEST pipeline={pipeline} "
            f"pipeline_readable={pipeline_readable} "
            f"files_processed={files_processed} files_failed={files_failed} "
            f"specimens_ingested={specimens_ingested} "
            f"window_start={window_start.strftime('%Y-%m-%d')} "
            f"window_end={window_end.strftime('%Y-%m-%d')} "
            f"most_recent_test={recent_test}"
        )
        summary[pipeline] = {
            "files_processed": files_processed,
            "files_failed": files_failed,
            "specimens_ingested": specimens_ingested,
            "most_recent_test": recent_test,
        }

    print(f"DIGEST_RUN_SUMMARY {summary}")
    return summary
