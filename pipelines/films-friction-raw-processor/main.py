import io
import os
import re
from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery
from google.cloud import storage


PROJECT_ID = os.environ.get("PROJECT_ID", "notpla-machine-data")
BQ_DATASET = os.environ.get("BQ_DATASET", "films_tensile_london")
BQ_TABLE = os.environ.get("BQ_TABLE", "films_friction_curve_points")

WATCH_PREFIX = os.environ.get(
    "WATCH_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/"
    "tensiletester-films-friction-to-be-processed/"
    "tensiletester-films-friction-to-be-processed-raw/"
)
PROCESSED_PREFIX = os.environ.get(
    "PROCESSED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/"
    "tensiletester-films-friction-processed/"
    "tensiletester-films-friction-processed-raw/"
)
FAILED_PREFIX = os.environ.get(
    "FAILED_PREFIX",
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/"
    "tensiletester-films-friction-failed-processing/"
    "tensiletester-films-friction-failed-processing-raw/"
)


def should_process(object_name: str) -> bool:
    if not object_name.startswith(WATCH_PREFIX):
        return False
    if object_name.startswith(PROCESSED_PREFIX) or object_name.startswith(FAILED_PREFIX):
        return False
    return object_name.lower().endswith(".csv")


def extract_sample_from_filename(object_name: str) -> int:
    filename = object_name.split("/")[-1]
    match = re.search(r"sample-(\d+)", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract sample number from filename: {filename}")
    return int(match.group(1))


def parse_curve_dataframe(csv_bytes: bytes, source_file: str) -> pd.DataFrame:
    text = csv_bytes.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    expected = [
        "Time (s)",
        "Load (N)",
        "Displacement (mm)",
        "Stress (MPa)",
        "Strain (%)",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["sample"] = extract_sample_from_filename(source_file)
    out["time_s"] = pd.to_numeric(df["Time (s)"], errors="coerce")
    out["load_n"] = pd.to_numeric(df["Load (N)"], errors="coerce")
    out["displacement_mm"] = pd.to_numeric(df["Displacement (mm)"], errors="coerce")
    out["stress_mpa"] = pd.to_numeric(df["Stress (MPa)"], errors="coerce")
    out["strain_pct"] = pd.to_numeric(df["Strain (%)"], errors="coerce")
    out["source_file"] = source_file
    out["processed_at"] = datetime.now(timezone.utc)

    out = out.dropna(subset=["time_s", "load_n", "displacement_mm", "stress_mpa", "strain_pct"], how="all")
    if out.empty:
        raise ValueError("No numeric curve rows found after parsing")

    return out


def load_to_bigquery(df: pd.DataFrame) -> int:
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

    df = df[
        [
            "sample",
            "time_s",
            "load_n",
            "displacement_mm",
            "stress_mpa",
            "strain_pct",
            "source_file",
            "processed_at",
        ]
    ].copy()

    job = client.load_table_from_dataframe(
        df,
        table_id,
        job_config=bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND
        ),
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
        print(f"Missing bucket/name in event: {event}")
        return

    print(f"Event received for {name}")

    if not should_process(name):
        print(f"Skipping (not in watch scope): {name}")
        return

    storage_client = storage.Client(project=PROJECT_ID)
    blob = storage_client.bucket(bucket).blob(name)

    try:
        csv_bytes = blob.download_as_bytes()
        df = parse_curve_dataframe(csv_bytes, source_file=name)
        rows = load_to_bigquery(df)

        filename = name.split("/")[-1]
        dest = f"{PROCESSED_PREFIX}{filename}"
        move_blob(bucket, name, dest)

        print(f"Processed OK: {name}; rows_inserted={rows}; moved_to={dest}")

    except Exception as exc:
        try:
            filename = name.split("/")[-1]
            dest = f"{FAILED_PREFIX}{filename}"
            move_blob(bucket, name, dest)
            print(f"Processing failed for {name}; moved_to={dest}; error={exc}")
        except Exception as move_exc:
            print(f"Processing failed for {name}; also failed to move file; error={exc}; move_error={move_exc}")
        raise
