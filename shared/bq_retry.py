"""Wraps a BigQuery load job with retry-with-backoff for BigQuery's rate
limits - both the per-table `429 rateLimitExceeded: too many table update
operations for this table` (roughly 5 table-update ops/10s per table) and
the per-user `403 ... too many api requests per user per method for this
user_method (JobService.insertJob)` variant, which BigQuery also reports
as a plain 403 Forbidden rather than 429. A purely serial, one-file-at-a-
time caller can still trip either of these if individual load jobs
complete faster than that spacing, which is exactly what happened
backfilling films_tensile_curve_points once its rows shrank ~11x from
min/max downsampling (30 August 2026) - small dataframes load fast enough
to burst past the quota even with no concurrency at all. Both error
flavors showed up in that incident's manifest rows. See
pipeline-roadmap.md's Phase 5 entry.
"""

import time

from google.api_core.exceptions import Forbidden, TooManyRequests

MIN_SECONDS_BETWEEN_LOADS = 2.5


def _is_rate_limit(exc):
    if isinstance(exc, TooManyRequests):
        return True
    return isinstance(exc, Forbidden) and "rate limit" in str(exc).lower()


def load_dataframe_with_retry(bq_client, df, table_id, max_retries=5):
    """Blocks until the load job completes or genuinely fails. Retries only
    on a rate-limit error (429, or the 403-flavored rate limit BigQuery
    also uses), with exponential backoff; any other exception - including a
    genuine 403 permissions error - propagates immediately to the caller's
    existing per-file error handling.
    """
    attempt = 0
    while True:
        try:
            job = bq_client.load_table_from_dataframe(df, table_id)
            job.result()
            time.sleep(MIN_SECONDS_BETWEEN_LOADS)
            return
        except Exception as exc:
            if not _is_rate_limit(exc):
                raise
            attempt += 1
            if attempt > max_retries:
                raise
            backoff = MIN_SECONDS_BETWEEN_LOADS * (2 ** attempt)
            time.sleep(backoff)
