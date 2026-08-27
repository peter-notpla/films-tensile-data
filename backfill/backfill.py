import logging
import sys
from pathlib import Path

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.tensile_parser import extract_relevant_dataframe

# --------- Config ----------
PROJECT_ID = "notpla-machine-data"
BUCKET = "notpla-machine-data"

WATCH_PREFIX = (
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-summary-tables/"
)
PROCESSED_PREFIX = (
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-processed/"
)
FAILED_PREFIX = (
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-failed-processing/"
)

BQ_DATASET = "films_tensile_london"
# films_tensile_results is a view onto this table (WHERE row_state =
# "current"), not the underlying table, as of 27 August 2026 - see
# CLAUDE.md's "Table naming" section. Loads must target the real table.
BQ_TABLE = "films_tensile_results_all_revisions"

# Columns this legacy script backfills. Deliberately narrower than
# shared.tensile_parser.TABLE_COLUMNS: row_state, database_revision,
# archived_at, archived_by, revised_at, revised_by (Phase 2.5) are left
# out because this script doesn't call shared.revision_handling - it
# predates that model and is a one-off recovery tool, not part of the live
# event-triggered pipeline. Any rows it inserts get those six columns
# NULL and won't participate in the current/archived model until
# reconciled by hand. Wiring in full revision handling here is a separate
# task, not part of Phase 0.3 (the date-parsing fix this file addresses).
BACKFILL_COLUMNS = [
    "sample",
    "youngs_modulus_mpa",
    "offset_yield_mpa",
    "max_load_n",
    "max_stress_mpa",
    "break_pct",
    "toughness_mpa",
    "timestamp_start",
    "pellet_id",
    "extrusion_id",
    "test_direction",
    "sample_number",
    "sample_thickness_mm",
    "relative_humidity_pct",
    "notes",
    "user_initials",
    "source_file",
    "processed_at",
    "template_name",
    "timestamp_minute",
    "specimen_key",
    "validation_status",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


def load_to_bigquery(df: pd.DataFrame, bq: bigquery.Client) -> int:
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
    df = df[BACKFILL_COLUMNS].copy()

    job = bq.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND),
    )
    job.result()
    return len(df)


def move_blob(gcs: storage.Client, source_name: str, dest_name: str) -> None:
    bucket = gcs.bucket(BUCKET)
    blob = bucket.blob(source_name)
    bucket.copy_blob(blob, bucket, new_name=dest_name)
    blob.delete()


def main():
    gcs = storage.Client(project=PROJECT_ID)
    bq = bigquery.Client(project=PROJECT_ID)
    bucket = gcs.bucket(BUCKET)

    blobs = list(gcs.list_blobs(BUCKET, prefix=WATCH_PREFIX))
    csv_blobs = [b for b in blobs if b.name.lower().endswith(".csv")]

    logger.info("Found %d CSVs to backfill", len(csv_blobs))

    processed_files = 0
    failed_files = 0
    inserted_rows = 0
    dropped_rows = 0

    for i, blob in enumerate(csv_blobs, start=1):
        name = blob.name
        filename = name.split("/")[-1]
        try:
            csv_bytes = bucket.blob(name).download_as_bytes()
            df, rows_dropped, _row_errors = extract_relevant_dataframe(csv_bytes, source_file=name)

            rows = load_to_bigquery(df, bq)
            inserted_rows += rows
            dropped_rows += rows_dropped

            dest = f"{PROCESSED_PREFIX}{filename}"
            move_blob(gcs, name, dest)
            processed_files += 1

            logger.info("[%d/%d] OK: %s inserted=%d dropped=%d", i, len(csv_blobs), filename, rows, rows_dropped)

        except Exception:
            logger.exception("[%d/%d] FAIL: %s", i, len(csv_blobs), filename)
            try:
                dest = f"{FAILED_PREFIX}{filename}"
                move_blob(gcs, name, dest)
            except Exception:
                logger.exception("Also failed to move %s to failed-processing", filename)
            failed_files += 1

    logger.info(
        "DONE. processed_files=%d failed_files=%d inserted_rows=%d dropped_rows=%d",
        processed_files, failed_files, inserted_rows, dropped_rows,
    )


if __name__ == "__main__":
    main()
