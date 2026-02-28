import os
import io
import logging
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery


# ---------------- Config ----------------
PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BQ_DATASET = os.environ.get("BQ_DATASET", "films_tensile_london")
BQ_TABLE = os.environ.get("BQ_TABLE", "films_tensile_results")

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

logger = logging.getLogger("tensile_processor")
logger.setLevel(logging.INFO)

FOOTER_LABELS = {"mean", "sd", "min", "max"}


def _is_footer_row(first_cell: str) -> bool:
    if first_cell is None:
        return False
    return str(first_cell).strip().lower() in FOOTER_LABELS


def should_process(object_name: str) -> bool:
    # Only process CSVs landing under WATCH_PREFIX
    if not object_name.startswith(WATCH_PREFIX):
        return False
    # Never process anything already moved
    if object_name.startswith(PROCESSED_PREFIX) or object_name.startswith(FAILED_PREFIX):
        return False
    return object_name.lower().endswith(".csv")


def extract_relevant_dataframe(csv_bytes: bytes, source_file: str) -> pd.DataFrame:
    """
    Rules from you:
      - First row irrelevant (title line) -> drop it
      - Second row contains headers
      - Four footer rows start with Mean/SD/Min/Max in first column -> ignore them
      - Valuable rows are between header and footer
    """
    text = csv_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("CSV too short (needs title + header + data)")

    # Drop first line (title)
    trimmed = "\n".join(lines[1:])

    # Parse CSV where first line of trimmed is the header
    df_raw = pd.read_csv(io.StringIO(trimmed), dtype=str, keep_default_na=False)

    if "Sample" not in df_raw.columns:
        raise ValueError(f"Expected 'Sample' column not found. Columns: {list(df_raw.columns)}")

    # Find footer start
    footer_pos = None
    for i in range(len(df_raw)):
        if _is_footer_row(df_raw.iloc[i]["Sample"]):
            footer_pos = i
            break
    if footer_pos is None:
        raise ValueError("Footer block (Mean/SD/Min/Max) not found")

    df = df_raw.iloc[:footer_pos].copy()

    # Drop fully blank rows
    df = df.replace(r"^\s*$", "", regex=True)
    df = df[~(df == "").all(axis=1)]
    if len(df) == 0:
        raise ValueError("No data rows found between header and footer")

    out = pd.DataFrame()

    out["sample"] = pd.to_numeric(df["Sample"], errors="coerce").astype("Int64")
    out["youngs_modulus_mpa"] = pd.to_numeric(df.get("Young's Modulus (MPa)", ""), errors="coerce")
    out["offset_yield_mpa"] = pd.to_numeric(df.get("Offset Yield (MPa)", ""), errors="coerce")
    out["max_load_n"] = pd.to_numeric(df.get("Max Load (N) (N)", ""), errors="coerce")
    out["max_stress_mpa"] = pd.to_numeric(df.get("Max Stress (MPa) (MPa)", ""), errors="coerce")
    out["break_pct"] = pd.to_numeric(df.get("Break (%)", ""), errors="coerce")
    out["toughness_mpa"] = pd.to_numeric(df.get("Toughness (MPa)", ""), errors="coerce")

    out["timestamp_start"] = pd.to_datetime(df.get("Timestamp - Start ", ""), errors="coerce")

    out["pellet_id"] = df.get("Pellet ID (Prompt For Value - Before Test)", "").astype(str)
    out["extrusion_id"] = df.get("Extrusion ID (Prompt For Value - Before Test)", "").astype(str)
    out["test_direction"] = df.get("Test Direction (Prompt For Value - Before Test)", "").astype(str)
    out["sample_number"] = df.get("Sample Number  (Prompt For Value - Before Test)", "").astype(str)
    out["sample_thickness_mm"] = pd.to_numeric(
        df.get("Sample Thickness (mm) (Prompt For Value - Before Test)", ""), errors="coerce"
    )
    out["relative_humidity_pct"] = pd.to_numeric(
        df.get("Relative Humidity (%) (Prompt For Value - Before Test)", ""), errors="coerce"
    )
    out["notes"] = df.get("Notes (Prompt For Value - After Test)", "").astype(str)
    out["user_initials"] = df.get("User Initials (Prompt For Value - After Test)", "").astype(str)

    # Ingestion metadata
    out["source_file"] = source_file
    out["processed_at"] = datetime.now(timezone.utc)

    # Require sample present
    out = out[~out["sample"].isna()]
    if len(out) == 0:
        raise ValueError("No valid specimen rows (sample column empty after cleaning)")

    return out


def load_to_bigquery(df: pd.DataFrame) -> int:
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

    # Ensure columns match BigQuery schema order
    df = df[
        [
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
        ]
    ].copy()

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND),
    )
    job.result()
    return len(df)


def move_blob(bucket_name: str, source_name: str, dest_name: str) -> None:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_name)

    bucket.copy_blob(blob, bucket, new_name=dest_name)
    blob.delete()


def process_gcs_event(event, context=None):
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

        filename = name.split("/")[-1]
        dest = f"{PROCESSED_PREFIX}{filename}"
        move_blob(bucket, name, dest)

        logger.info("Processed OK", extra={"source_file": name, "rows_inserted": rows, "moved_to": dest})

    except Exception:
        try:
            filename = name.split("/")[-1]
            dest = f"{FAILED_PREFIX}{filename}"
            move_blob(bucket, name, dest)
            logger.exception("Processing failed; moved to failed-processing", extra={"source_file": name, "moved_to": dest})
        except Exception:
            logger.exception("Processing failed; also failed to move to failed-processing", extra={"source_file": name})
        raise
