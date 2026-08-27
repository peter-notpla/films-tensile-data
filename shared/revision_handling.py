"""Metadata revision handling (Phase 2.5): re-ingesting a file that contains
a specimen already in the table supersedes the prior row instead of leaving
it as an undetected duplicate under plain WRITE_APPEND.

Column shape aligned to Callum's existing pattern on
Rigid_Tensile_euw2.tensile_v21_results (row_state, database_revision,
archived_at, archived_by, revised_at, revised_by). The VALUE semantics
below are this repo's own interpretation of those column names, not a
copy of his code (not available to read here) - see pipeline-roadmap.md
item 2.5 for why this was built ahead of that conversation, and reconcile
if his conventions differ once it happens.

The historical backfill that established this same shape for existing rows
(pipeline-roadmap.md, Checkpoint I) found that most pre-existing "duplicate"
specimen_keys were not corrections at all - VectorPro re-exports a
cumulative results table repeatedly during a test session, so an early
specimen reappears, byte-identical, in every later export of the same
session. This module treats that the same way as a genuine correction
(archive the old row, the new one becomes current): the two cases are
indistinguishable from a single file's data alone, and archiving a
byte-identical row is harmless, just not very meaningful as a revision
count. `database_revision` should be read as "how many times this
specimen's row has been (re)written", not "how many times it was
corrected".

No MERGE precedent existed anywhere in this codebase before this.
Deliberately not a single MERGE statement: inserting the new row must
always happen, archiving an old one only conditionally, which isn't what
MERGE's matched/not-matched semantics express cleanly for a batch of
several rows at once. Two steps instead: a live BigQuery UPDATE to archive
superseded rows, then the existing load_table_from_dataframe append for
the new ones (unchanged call site in each main.py).
"""

import json
from datetime import datetime, timezone

from google.cloud import bigquery


def dedupe_within_file(df, row_errors):
    """If a single file contains the same specimen_key more than once, keep
    only the last occurrence and route the earlier one(s) to row_errors.
    Two rows both claiming row_state='current' for the same key in one
    batch would defeat the point of this whole mechanism, so this must run
    before apply_revision_handling, not after."""
    dup_mask = df["specimen_key"].duplicated(keep="last")
    if not dup_mask.any():
        return df, row_errors

    for idx, row in df.loc[dup_mask].iterrows():
        row_errors.append({
            "row_number": int(idx) + 1,
            "reason": f"duplicate specimen_key within this file, superseded by a later row in the same file: {row['specimen_key']!r}",
            "raw_row": json.dumps(row.to_dict(), default=str),
        })

    return df[~dup_mask].copy(), row_errors


def apply_revision_handling(df, bq_client: bigquery.Client, table_id: str, source_file: str):
    """df must already be deduplicated within-file (see dedupe_within_file)
    and have a specimen_key column. Not pure: queries and, if any matches
    are found, UPDATEs the live table to archive superseded rows, before
    returning df annotated with row_state/database_revision/revised_at/
    revised_by ready to load as the new 'current' rows."""
    keys = df["specimen_key"].tolist()
    df = df.copy()

    if not keys:
        return df

    lookup_query = f"""
        SELECT specimen_key, database_revision
        FROM `{table_id}`
        WHERE specimen_key IN UNNEST(@keys) AND row_state = "current"
    """
    lookup_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("keys", "STRING", keys),
    ])
    existing = {
        row.specimen_key: row.database_revision
        for row in bq_client.query(lookup_query, job_config=lookup_config).result()
    }

    if existing:
        archive_query = f"""
            UPDATE `{table_id}`
            SET row_state = "archived", archived_at = CURRENT_TIMESTAMP(), archived_by = @source_file
            WHERE specimen_key IN UNNEST(@keys) AND row_state = "current"
        """
        archive_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("keys", "STRING", list(existing.keys())),
            bigquery.ScalarQueryParameter("source_file", "STRING", source_file),
        ])
        bq_client.query(archive_query, job_config=archive_config).result()

    now = datetime.now(timezone.utc)
    df["row_state"] = "current"
    df["database_revision"] = df["specimen_key"].map(lambda k: existing.get(k, 0) + 1)
    df["archived_at"] = None
    df["archived_by"] = None
    df["revised_at"] = now
    df["revised_by"] = source_file

    return df
