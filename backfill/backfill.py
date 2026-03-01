import io
import logging
from datetime import datetime, timezone

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery

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
BQ_TABLE = "films_tensile_results"

FOOTER_LABELS = {"mean", "sd", "min", "max"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill")


def _is_footer_row(first_cell: str) -> bool:
    if first_cell is None:
        return False
    return str(first_cell).strip().lower() in FOOTER_LABELS


def extract_relevant_dataframe(csv_bytes: bytes, source_file: str) -> pd.DataFrame:
    text = csv_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("CSV too short")

    trimmed = "\n".join(lines[1:])  # drop first title line
    df_raw = pd.read_csv(io.StringIO(trimmed), dtype=str, keep_default_na=False)

    if "Sample" not in df_raw.columns:
        raise ValueError(f"Expected 'Sample' column not found. Columns: {list(df_raw.columns)}")

    footer_pos = None
    for i in range(len(df_raw)):
        if _is_footer_row(df_raw.iloc[i]["Sample"]):
            footer_pos = i
            break
    if footer_pos is None:
        raise ValueError("Footer (Mean/SD/Min/Max) not found")

    df = df_raw.iloc[:footer_pos].copy()
    df = df.replace(r"^\s*$", "", regex=True)
    df = df[~(df == "").all(axis=1)]
    if len(df) == 0:
        raise ValueError("No specimen rows found")

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

    out["source_file"] = source_file
    out["processed_at"] = datetime.now(timezone.utc)

    out = out[~out["sample"].isna()]
    if len(out) == 0:
        raise ValueError("No valid specimen rows")

    return out


def load_to_bigquery(df: pd.DataFrame, bq: bigquery.Client) -> int:
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
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

    for i, blob in enumerate(csv_blobs, start=1):
        name = blob.name
        filename = name.split("/")[-1]
        try:
            csv_bytes = bucket.blob(name).download_as_bytes()
            df = extract_relevant_dataframe(csv_bytes, source_file=name)

            rows = load_to_bigquery(df, bq)
            inserted_rows += rows

            dest = f"{PROCESSED_PREFIX}{filename}"
            move_blob(gcs, name, dest)
            processed_files += 1

            logger.info("[%d/%d] OK: %s inserted=%d", i, len(csv_blobs), filename, rows)

        except Exception:
            logger.exception("[%d/%d] FAIL: %s", i, len(csv_blobs), filename)
            try:
                dest = f"{FAILED_PREFIX}{filename}"
                move_blob(gcs, name, dest)
            except Exception:
                logger.exception("Also failed to move %s to failed-processing", filename)
            failed_files += 1

    logger.info("DONE. processed_files=%d failed_files=%d inserted_rows=%d", processed_files, failed_files, inserted_rows)


if __name__ == "__main__":
    main()
