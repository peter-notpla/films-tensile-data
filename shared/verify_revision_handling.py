"""Dry-run check for shared/revision_handling.py against real processed
files: reports what WOULD be archived and what revision numbers WOULD be
assigned, without ever calling apply_revision_handling's live UPDATE.

This is the verification pipeline-roadmap.md's Checkpoint I design calls
for before the real (writing) function is ever wired into a main.py.
Read-only: downloads files and queries BigQuery, writes nothing.

Usage: python3 shared/dry_run_revision_handling.py
"""

import sys
from pathlib import Path

from google.cloud import bigquery, storage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.tensile_parser import extract_relevant_dataframe
from shared.friction_parser import extract_friction_dataframe
from shared.revision_handling import dedupe_within_file

PROJECT_ID = "notpla-machine-data"


def dry_run(label, prefix, extract_fn, table_id, n_files=15):
    print(f"--- {label} ---")
    sc = storage.Client(project=PROJECT_ID)
    bq = bigquery.Client(project=PROJECT_ID)

    blobs = list(sc.list_blobs("notpla-machine-data", prefix=prefix, max_results=n_files * 2))
    blobs = [b for b in blobs if b.name.lower().endswith(".csv")][:n_files]

    total_would_archive = 0
    total_new_current = 0
    total_within_file_dupes = 0

    for b in blobs:
        content = b.download_as_bytes()
        df, rows_dropped, row_errors = extract_fn(content, source_file=b.name)
        before = len(df)
        df, row_errors = dedupe_within_file(df, row_errors)
        within_file_dupes = before - len(df)
        total_within_file_dupes += within_file_dupes

        keys = df["specimen_key"].tolist()
        if not keys:
            continue

        lookup_query = f"""
            SELECT specimen_key, database_revision
            FROM `{table_id}`
            WHERE specimen_key IN UNNEST(@keys) AND row_state = "current"
        """
        job_config = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("keys", "STRING", keys),
        ])
        existing = {
            row.specimen_key: row.database_revision
            for row in bq.query(lookup_query, job_config=job_config).result()
        }

        would_archive = len(existing)
        total_would_archive += would_archive
        total_new_current += len(df)

        filename = b.name.split("/")[-1]
        print(f"  {filename}: {len(df)} specimen rows, "
              f"{within_file_dupes} within-file duplicate(s) dropped, "
              f"{would_archive} existing current row(s) would be archived")

    print(f"  TOTAL across {len(blobs)} files: {total_new_current} rows would become current, "
          f"{total_would_archive} existing current rows would be archived, "
          f"{total_within_file_dupes} within-file duplicates dropped")
    print()


def main():
    dry_run(
        "tensile",
        "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/tensiletester-films-tensile-processed/",
        extract_relevant_dataframe,
        f"{PROJECT_ID}.films_tensile_london.films_tensile_results",
    )
    dry_run(
        "friction",
        "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/tensiletester-films-friction-processed/",
        extract_friction_dataframe,
        f"{PROJECT_ID}.machine_data.films_friction_raw",
    )


if __name__ == "__main__":
    main()
