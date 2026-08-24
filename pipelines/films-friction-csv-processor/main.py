import os
import re
import json
import hashlib
import tempfile
import traceback
from datetime import datetime, timezone

import pandas as pd
from google.cloud import bigquery
from google.cloud import storage


PIPELINE_NAME = "friction"
MACHINE_ID = "tensiletester-1"


def write_manifest(project_id, source_file, checksum, status, rows_total,
                    rows_inserted, rows_rejected, error_message):
    """Best-effort manifest row. Never allowed to fail the pipeline run."""
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.films_pipeline_ops.films_pipeline_manifest"
        row = {
            "pipeline": PIPELINE_NAME,
            "source_file": source_file,
            "checksum": checksum,
            "machine_id": MACHINE_ID,
            "status": status,
            "rows_total": rows_total,
            "rows_inserted": rows_inserted,
            "rows_rejected": rows_rejected,
            "error_message": error_message,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        errors = client.insert_rows_json(table_id, [row])
        if errors:
            print(f"Manifest insert returned errors: {errors}")
    except Exception as manifest_exc:
        print(f"Manifest insert failed (pipeline result unaffected): {manifest_exc}")


def write_row_errors(project_id, row_errors, source_file, checksum):
    """Best-effort row-errors write. Never allowed to fail the pipeline run."""
    if not row_errors:
        return
    try:
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.films_pipeline_ops.films_pipeline_row_errors"
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            {
                "pipeline": PIPELINE_NAME,
                "source_file": source_file,
                "checksum": checksum,
                "row_number": e["row_number"],
                "reason": e["reason"],
                "raw_row": e["raw_row"],
                "processed_at": now,
            }
            for e in row_errors
        ]
        errors = client.insert_rows_json(table_id, rows)
        if errors:
            print(f"Row-errors insert returned errors: {errors}")
    except Exception as row_errors_exc:
        print(f"Row-errors insert failed (pipeline result unaffected): {row_errors_exc}")


def gcs_csv_to_bigquery(data, context):
    PROJECT_ID = os.environ["PROJECT_ID"]
    BQ_DATASET = os.environ["BQ_DATASET"]
    BQ_TABLE = os.environ["BQ_TABLE"]
    WATCH_PREFIX = os.environ["WATCH_PREFIX"].rstrip("/") + "/"
    PROCESSED_PREFIX = os.environ["PROCESSED_PREFIX"].rstrip("/") + "/"
    FAILED_PREFIX = os.environ["FAILED_PREFIX"].rstrip("/") + "/"

    TIMESTAMP_FORMATS = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
    ]

    FOOTER_MARKERS = {"mean", "sd", "min", "max"}

    def normalize(name):
        name = (name or "").strip().lower()
        name = name.replace("%", "pct")
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return re.sub(r"_+", "_", name).strip("_")

    def parse_ts(value):
        if pd.isna(value) or str(value).strip() == "":
            return None
        text = str(value).strip()
        for fmt in TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    bucket_name = data["bucket"]
    blob_name = data["name"]

    if not blob_name.startswith(WATCH_PREFIX) or blob_name.endswith("/"):
        print(f"Skipping: {blob_name}")
        return

    storage_client = storage.Client()
    bq_client = bigquery.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)

    filename = blob_name.split("/")[-1]
    processed_path = f"{PROCESSED_PREFIX}{filename}"
    failed_path = f"{FAILED_PREFIX}{filename}"
    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    checksum = None

    try:
        print(f"Processing gs://{bucket_name}/{blob_name}")

        blob = bucket.blob(blob_name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            blob.download_to_filename(tmp.name)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            checksum = hashlib.md5(f.read()).hexdigest()

        try:
            df = pd.read_csv(tmp_path, header=1, dtype=str, encoding="utf-8", keep_default_na=False)
        except UnicodeDecodeError:
            df = pd.read_csv(tmp_path, header=1, dtype=str, encoding="latin1", keep_default_na=False)

        os.remove(tmp_path)

        cols = [c for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed:")]
        df = df[cols]

        first_col = df.columns[0]
        df = df[~df[first_col].astype(str).str.lower().isin(FOOTER_MARKERS)]

        # A fully blank row (every column empty, e.g. Excel's trailing
        # padding) carries nothing and is dropped silently here. A row
        # that's blank in some columns but not others is real data with a
        # problem and is handled below instead, not swallowed here. This
        # used to check only the first column, which meant a row with a
        # blank Sample but real measurements in every other column was
        # dropped the same way as genuine padding, with no trace at all.
        blanked = df.replace(r"^\s*$", "", regex=True)
        df = df[~(blanked == "").all(axis=1)]

        if df.empty:
            raise ValueError("No data rows")

        df.columns = [normalize(c) for c in df.columns]

        if "sample" not in df.columns:
            raise ValueError("Missing sample column")

        if "timestamp_start" not in df.columns:
            raise ValueError("Missing timestamp column")

        # Row_number below is a stable 1-based position within the data
        # block, and the reset lets the raw-value snapshot and the parsed
        # masks below line up by index.
        df = df.reset_index(drop=True)
        raw_values = df.copy()

        # A row needs both a sample number and a parseable timestamp to be
        # usable. Previously a single blank sample or bad timestamp raised
        # and killed the entire file. Bad rows are now dropped individually
        # and routed to the row-errors table with the raw values and reason;
        # the rest of the file still loads.
        bad_sample_mask = df["sample"].astype(str).str.strip() == ""
        parsed_ts = df["timestamp_start"].apply(parse_ts)
        bad_ts_mask = parsed_ts.isna()
        bad_mask = bad_sample_mask | bad_ts_mask

        row_errors = []
        for idx, raw_row in raw_values.loc[bad_mask].iterrows():
            reasons = []
            if bad_sample_mask.loc[idx]:
                reasons.append("blank sample")
            if bad_ts_mask.loc[idx]:
                reasons.append(f"unparseable timestamp: {raw_values.loc[idx, 'timestamp_start']!r}")
            row_errors.append({
                "row_number": int(idx) + 1,
                "reason": "; ".join(reasons),
                "raw_row": json.dumps(raw_row.to_dict()),
            })

        rows_total = len(df)
        df = df[~bad_mask].copy()
        rows_dropped = len(row_errors)
        if df.empty:
            raise ValueError("No valid rows remain after dropping blank samples / unparseable timestamps")

        df["timestamp_start"] = parsed_ts[~bad_mask]
        df["timestamp_start"] = pd.to_datetime(df["timestamp_start"], utc=True)

        df["source_file"] = f"gs://{bucket_name}/{blob_name}"
        df["processed_at"] = pd.Timestamp.now(tz="UTC")

        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()

        bucket.copy_blob(blob, bucket, processed_path)
        blob.delete()

        print(f"SUCCESS -> {processed_path}")

        write_row_errors(PROJECT_ID, row_errors, source_file=gcs_uri, checksum=checksum)

        write_manifest(
            project_id=PROJECT_ID,
            source_file=gcs_uri,
            checksum=checksum,
            status="success",
            rows_total=rows_total,
            rows_inserted=len(df),
            rows_rejected=rows_dropped,
            error_message=None,
        )

    except Exception as e:
        print(f"FAILED: {e}")
        print(traceback.format_exc())

        try:
            blob = bucket.blob(blob_name)
            if blob.exists():
                bucket.copy_blob(blob, bucket, failed_path)
                blob.delete()
                print(f"Moved to failed: {failed_path}")
        except Exception as move_err:
            print(f"Move failed: {move_err}")

        write_manifest(
            project_id=PROJECT_ID,
            source_file=gcs_uri,
            checksum=checksum,
            status="failed",
            rows_total=None,
            rows_inserted=0,
            rows_rejected=0,
            error_message=str(e)[:1500],
        )

        raise
