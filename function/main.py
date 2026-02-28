import os
import io
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery
import logging

# ---------- Config (env vars set at deploy time) ----------
PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BQ_DATASET = os.environ.get("BQ_DATASET", "films_tensile_london")
BQ_TABLE = os.environ.get("BQ_TABLE", "films_tensile_results")

# Folder prefixes inside the bucket
WATCH_PREFIX = os.environ.get(
    "WATCH_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-summary-tables/"
)
PROCESSED_PREFIX = os.environ.get(
    "PROCESSED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-processed/"
)
FAILED_PREFIX = os.environ.get(
    "FAILED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/"
    "tensiletester-films-tensile-failed-processing/"
)

# ---------- Logging (structured-ish) ----------
logger = logging.getLogger("tensile_processor")
logger.setLevel(logging.INFO)


FOOTER_LABELS = {"mean", "sd", "min", "max"}


def _is_footer_row(first_cell: str) -> bool:
    if first_cell is None:
        return False
    return str(first_cell).strip().lower() in FOOTER_LABELS


def extract_relevant_dataframe(csv_bytes: bytes, source_file: str) -> pd.DataFrame:
    """
    Rules:
    - Row 1 irrelevant
    - Row 2 contains headers
    - Four footer rows starting with Mean/SD/Min/Max are irrelevant
    - Valuable data is between headers and footer block
    """
    text = csv_bytes.decode("utf-8", errors="replace")

    # Read as raw rows (no header), preserve empty cells
    raw = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False)

    if raw.shape[0] < 3:
        raise ValueError("CSV too short to contain headers + data")

    # Row 0 irrelevant; headers on row 1
    headers = raw.iloc[1].tolist()
    headers = [str(h).strip() if str(h).strip() != "" else f"col_{i}" for i, h in enumerate(headers)]

    data = raw.iloc[2:].copy()  # rows after header row
    data.columns = headers

    # Find footer start: first row where first column cell is Mean/SD/Min/Max
    first_col_name = headers[0]
    footer_idx = None
    for i in range(len(data)):
        if _is_footer_row(data.iloc[i][first_col_name]):
            footer_idx = i
            break

    if footer_idx is None:
        raise ValueError("Could not find footer block (Mean/SD/Min/Max)")

    data = data.iloc[:footer_idx].copy()

    # Drop any fully empty rows
    data = data.replace(r"^\s*$", "", regex=True)
    data = data[~(data == "").all(axis=1)]

    if data.shape[0] == 0:
        raise ValueError("No data rows found between header and footer")

    # Add required metadata
    data["source_file"] = source_file
    data["processed_at"] = datetime.now(timezone.utc)

    return data


def load_to_bigquery(df: pd.DataFrame) -> int:
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

    # Append only
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=False,
    )

    # Let BigQuery coerce where it can; keep strings where mismatch occurs.
    # (We’ll tighten schema mapping once we confirm the real column names in your sample CSVs.)
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()

    return df.shape[0]


def move_blob(bucket_name: str, source_name: str, dest_name: str) -> None:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_name)

    bucket.copy_blob(blob, bucket, new_name=dest_name)
    blob.delete()


def should_process(object_name: str) -> bool:
    # Only process objects in the watched prefix (and not already in processed/failed)
    if not object_name.startswith(WATCH_PREFIX):
        return False
    if object_name.startswith(PROCESSED_PREFIX) or object_name.startswith(FAILED_PREFIX):
        return False
    # Only CSV
    return object_name.lower().endswith(".csv")


def process_gcs_event(event, context=None):
    """
    Cloud Functions (2nd gen) background event handler.
    Event structure varies slightly; we’ll rely on bucket/name keys.
    """
    bucket = event.get("bucket") if isinstance(event, dict) else None
    name = event.get("name") if isinstance(event, dict) else None

    if not bucket or not name:
        logger.error("Missing bucket/name in event", extra={"event": str(event)})
        return

    logger.info("Event received", extra={"bucket": bucket, "name": name})

    if not should_process(name):
        logger.info("Skipping (not in watch scope)", extra={"bucket": bucket, "name": name})
        return

    storage_client = storage.Client(project=PROJECT_ID)
    blob = storage_client.bucket(bucket).blob(name)

    try:
        csv_bytes = blob.download_as_bytes()
        df = extract_relevant_dataframe(csv_bytes, source_file=name)

        rows = load_to_bigquery(df)

        # Move to processed folder, keeping filename
        filename = name.split("/")[-1]
        dest = f"{PROCESSED_PREFIX}{filename}"
        move_blob(bucket, name, dest)

        logger.info(
            "Processed OK",
            extra={"bucket": bucket, "name": name, "rows": rows, "moved_to": dest},
        )

    except Exception as e:
        # Move to failed folder, keeping filename
        try:
            filename = name.split("/")[-1]
            dest = f"{FAILED_PREFIX}{filename}"
            move_blob(bucket, name, dest)
            logger.exception(
                "Processing failed; moved to failed-processing",
                extra={"bucket": bucket, "name": name, "moved_to": dest},
            )
        except Exception:
            logger.exception(
                "Processing failed; additionally failed to move to failed-processing",
                extra={"bucket": bucket, "name": name},
            )
        # Re-raise so the platform records the failure clearly
        raise
