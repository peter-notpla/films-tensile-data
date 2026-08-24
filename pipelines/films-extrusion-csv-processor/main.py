import os
import json
import hashlib
import traceback
from datetime import datetime, timezone
from io import BytesIO

import functions_framework
import pandas as pd
from google.cloud import bigquery, storage


PROJECT_ID = os.environ["PROJECT_ID"]
BQ_DATASET = os.environ["BQ_DATASET"]
BQ_TABLE = os.environ["BQ_TABLE"]
WATCH_PREFIX = os.environ["WATCH_PREFIX"].strip("/")
PROCESSED_PREFIX = os.environ["PROCESSED_PREFIX"].strip("/")
FAILED_PREFIX = os.environ["FAILED_PREFIX"].strip("/")

MANIFEST_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_manifest"
ROW_ERRORS_TABLE = f"{PROJECT_ID}.films_pipeline_ops.films_pipeline_row_errors"
PIPELINE_NAME = "extrusion"
MACHINE_ID = "collin-e25e"

storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)


TABLE_COLUMNS = [
    "trial_code", "date", "ingredients", "proportion", "batches", "pellet_id",
    "extrusion_id", "time", "zone_1", "zone_2", "zone_3", "zone_4", "zone_5",
    "zone_7", "zone_8", "zone_9", "zone_10", "zone_11", "screw_speed_rpm",
    "amp_a", "torque_percent", "die_pressure_bar", "melt_temp_c",
    "calender_set_temp_c", "calender_actual_temp_c", "calender_speed_m_min",
    "middle_roller_speed_percent", "spooling_reel_torque_nm",
    "left_film_thickness_1_mm", "film_thickness_2_mm", "film_thickness_3_mm",
    "film_thickness_4_mm", "film_thickness_5_mm", "film_thickness_6_mm",
    "film_thickness_7_mm", "film_thickness_8_mm", "film_thickness_9_mm",
    "right_film_thickness_10_mm", "average_thickness_mm",
    "sd_percent_variation", "left_film_thickness_1_end_mm",
    "film_thickness_2_end_mm", "film_thickness_3_end_mm",
    "film_thickness_4_end_mm", "film_thickness_5_end_mm",
    "film_thickness_6_end_mm", "film_thickness_7_end_mm",
    "film_thickness_8_end_mm", "film_thickness_9_end_mm",
    "right_film_thickness_10_end_mm", "average_thickness_end_mm",
    "sd_end", "percent_variation_end", "width_mm", "length_m",
    "pellets_moisture_content_percent", "relative_humidity_percent",
    "temperature_c", "comments", "key", "source_file", "processed_at",
    "sd", "variation", "variation_end"
]


HEADER_MAP = {
    "Trial Code": "trial_code",
    "Date": "date",
    "Ingredients": "ingredients",
    "Proportion": "proportion",
    "Batches": "batches",
    "Pellet ID": "pellet_id",
    "Extrusion ID": "extrusion_id",
    "Time": "time",
    "Zone 1": "zone_1",
    "Zone 2": "zone_2",
    "Zone 3": "zone_3",
    "Zone 4": "zone_4",
    "Zone 5": "zone_5",
    "Zone 7": "zone_7",
    "Zone 8": "zone_8",
    "Zone 9": "zone_9",
    "Zone 10": "zone_10",
    "Zone 11": "zone_11",
    "Screw speed (rpm)": "screw_speed_rpm",
    "Amp (A)": "amp_a",
    "Torque (%)": "torque_percent",
    "Die pressure (bar)": "die_pressure_bar",
    "Melt temp (°C)": "melt_temp_c",
    "Calender Set Temp (°C)": "calender_set_temp_c",
    "Calender Actual Temp (°C)": "calender_actual_temp_c",
    "Calender Speed (m/min)": "calender_speed_m_min",
    "Middle Roller Speed (%)": "middle_roller_speed_percent",
    "Spooling Reel Torque (Nm)": "spooling_reel_torque_nm",
    "Left Film Thickness 1 (mm)": "left_film_thickness_1_mm",
    "Film Thickness 2 (mm)": "film_thickness_2_mm",
    "Film Thickness 3 (mm)": "film_thickness_3_mm",
    "Film Thickness 4 (mm)": "film_thickness_4_mm",
    "Film Thickness 5 (mm)": "film_thickness_5_mm",
    "Film Thickness 6 (mm)": "film_thickness_6_mm",
    "Film Thickness 7 (mm)": "film_thickness_7_mm",
    "Film Thickness 8 (mm)": "film_thickness_8_mm",
    "Film Thickness 9 (mm)": "film_thickness_9_mm",
    "Right Film Thickness 10 (mm)": "right_film_thickness_10_mm",
    "Average Thickness (mm)": "average_thickness_mm",
    "SD": "sd",
    "% Variation": "variation",
    "Variation": "variation",
    "Left Film Thickness 1 End (mm)": "left_film_thickness_1_end_mm",
    "Film Thickness 2 End (mm)": "film_thickness_2_end_mm",
    "Film Thickness 3 End (mm)": "film_thickness_3_end_mm",
    "Film Thickness 4 End (mm)": "film_thickness_4_end_mm",
    "Film Thickness 5 End (mm)": "film_thickness_5_end_mm",
    "Film Thickness 6 End (mm)": "film_thickness_6_end_mm",
    "Film Thickness 7 End (mm)": "film_thickness_7_end_mm",
    "Film Thickness 8 End (mm)": "film_thickness_8_end_mm",
    "Film Thickness 9 End (mm)": "film_thickness_9_end_mm",
    "Right Film Thickness 10 End (mm)": "right_film_thickness_10_end_mm",
    "Average Thickness End (mm)": "average_thickness_end_mm",
    "SD End": "sd_end",
    "% Variation End": "variation_end",
    "Variation End": "variation_end",
    "Width (mm)": "width_mm",
    "Length (m)": "length_m",
    "Pellets Moisture Content (%)": "pellets_moisture_content_percent",
    "Relative Humidity (%)": "relative_humidity_percent",
    "Temperature (°C)": "temperature_c",
    "Comments": "comments",
    "Key": "key",
}

# ---------------------------------------------------------------
# ADDED 21 Aug 2026. Guards against ingesting files that are not
# extrusion exports. See the checks in process_file().
# ---------------------------------------------------------------
IDENTITY_COLUMNS = ["trial_code", "pellet_id", "extrusion_id"]

# The four real extrusion files map well over 40 columns each, so 10 is a
# generous floor that admits a trimmed-down export while rejecting noise.
MIN_MAPPED_COLUMNS = 10


FLOAT_COLUMNS = [
    c for c in TABLE_COLUMNS
    if c not in {"trial_code", "date", "ingredients", "proportion", "batches",
                 "pellet_id", "extrusion_id", "time", "comments", "key",
                 "source_file", "processed_at"}
]


def normalize_header(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def parse_date(value):
    if pd.isna(value) or str(value).strip() == "":
        return None

    text = str(value).strip()

    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%-d/%-m/%Y", "%-d/%-m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def move_blob(bucket_name, blob_name, new_prefix):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    new_name = f"{new_prefix}/{blob_name.split('/')[-1]}"
    bucket.copy_blob(blob, bucket, new_name)
    blob.delete()


def delete_stale_failed_copy(bucket_name, blob_name):
    """Best-effort: once this filename has successfully reprocessed, remove
    any earlier failed copy sitting under FAILED_PREFIX, so a fixed file
    doesn't leave what looks like an unresolved failure there forever."""
    try:
        filename = blob_name.split("/")[-1]
        failed_path = f"{FAILED_PREFIX}/{filename}"
        blob = storage_client.bucket(bucket_name).blob(failed_path)
        if blob.exists():
            blob.delete()
            print(f"Deleted stale failed-processing copy: {failed_path}")
    except Exception as cleanup_exc:
        print(f"Failed to delete stale failed-processing copy (pipeline result unaffected): {cleanup_exc}")


def write_manifest(source_file, checksum, status, rows_total, rows_inserted,
                    rows_rejected, error_message):
    """Best-effort manifest row. Never allowed to fail the pipeline run."""
    try:
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
        errors = bq_client.insert_rows_json(MANIFEST_TABLE, [row])
        if errors:
            print(f"Manifest insert returned errors: {errors}")
    except Exception as manifest_exc:
        print(f"Manifest insert failed (pipeline result unaffected): {manifest_exc}")


def write_row_errors(row_errors, source_file, checksum):
    """Best-effort row-errors write. Never allowed to fail the pipeline run."""
    if not row_errors:
        return
    try:
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
        errors = bq_client.insert_rows_json(ROW_ERRORS_TABLE, rows)
        if errors:
            print(f"Row-errors insert returned errors: {errors}")
    except Exception as row_errors_exc:
        print(f"Row-errors insert failed (pipeline result unaffected): {row_errors_exc}")


@functions_framework.cloud_event
def process_file(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    blob_name = data["name"]

    print(f"Received file: {blob_name}")

    if not blob_name.startswith(WATCH_PREFIX):
        print("Skipping: outside watch folder")
        return

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    checksum = None

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        checksum = hashlib.md5(content).hexdigest()

        try:
            df = pd.read_csv(BytesIO(content), header=1)
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), header=1, encoding="latin-1")

        # First processing step: drop columns whose row-2 header cell is empty.
        df = df.loc[:, [
            c is not None
            and str(c).strip() != ""
            and not str(c).startswith("Unnamed:")
            for c in df.columns
        ]]

        # Normalize remaining headers.
        df.columns = [normalize_header(c) for c in df.columns]

        # Second processing step: drop rows whose first cell / column A is empty.
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(str).str.strip()
        df = df[
            df[first_col].notna()
            & (df[first_col] != "")
            & (df[first_col].str.lower() != "nan")
        ]

        # Rename known headers to BigQuery column names.
        df = df.rename(columns=HEADER_MAP)

        # Snapshot for the row-errors table, before numeric/date coercion
        # turns unparseable values into NaN and loses the original content.
        # No rows are dropped between here and the identity check below, so
        # the index still lines up.
        raw_snapshot = df.copy()

        # ---------------------------------------------------------------
        # ADDED 21 Aug 2026: required-column guard.
        #
        # Before this existed the parser accepted any CSV at all. A test file
        # containing the literal text "NotAHeader,AlsoNot" was dropped to
        # zero usable columns, padded out to all 64 table columns as NULL,
        # loaded as one row, and logged as a success.
        #
        # Two checks, both cheap:
        #   1. enough recognised headers to plausibly be an extrusion export
        #   2. at least one identity column actually present in the source
        #
        # Raising here is what puts the file in the failed folder and sends
        # the alert, which is the whole point of the restored failure path.
        # ---------------------------------------------------------------
        mapped = [c for c in df.columns if c in TABLE_COLUMNS]
        if len(mapped) < MIN_MAPPED_COLUMNS:
            raise ValueError(
                f"Only {len(mapped)} recognised columns "
                f"(need at least {MIN_MAPPED_COLUMNS}). "
                f"This does not look like an extrusion export. "
                f"Headers found: {list(df.columns)[:10]}"
            )

        identity_present = [c for c in IDENTITY_COLUMNS if c in df.columns]
        if not identity_present:
            raise ValueError(
                f"None of the identity columns {IDENTITY_COLUMNS} are present. "
                f"Recognised columns: {mapped[:10]}"
            )

        # Drop ignored/unmapped columns, e.g. Pellets sample taken?
        extra_columns = [c for c in df.columns if c not in TABLE_COLUMNS]
        if extra_columns:
            print(f"Dropping unmapped columns: {extra_columns}")
        df = df[[c for c in df.columns if c in TABLE_COLUMNS]]

        # Ensure all BigQuery table columns exist.
        for col in TABLE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Parse dates.
        df["date"] = df["date"].apply(parse_date)

        # Cast numeric fields safely.
        for col in FLOAT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # ---------------------------------------------------------------
        # ADDED 20 Aug 2026. Optional: remove this block if you would rather
        # keep the fix minimal. Leading and trailing spaces carry no
        # information and are invisible in every UI, but they silently break
        # joins and grouping. The extrusion table currently holds values like
        # " AO 260701 LR 1379" which fail an exact-match lookup.
        # ---------------------------------------------------------------
        for col in ("trial_code", "pellet_id", "extrusion_id", "ingredients",
                    "proportion", "batches", "time", "comments", "key"):
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda v: v.strip() if isinstance(v, str) else v
                )

        # Metadata.
        df["source_file"] = blob_name.split("/")[-1]
        df["processed_at"] = datetime.now(timezone.utc)

        # ---------------------------------------------------------------
        # ADDED 21 Aug 2026: drop rows with no identity at all. A row with no
        # trial code, no pellet id and no extrusion id cannot be joined to
        # anything and is not a usable record.
        # ---------------------------------------------------------------
        has_identity = False
        for col in IDENTITY_COLUMNS:
            col_filled = df[col].notna() & (df[col].astype(str).str.strip() != "")
            has_identity = col_filled if has_identity is False else (has_identity | col_filled)

        row_errors = []
        for idx in df.index[~has_identity]:
            row_errors.append({
                "row_number": int(idx) + 1,
                "reason": f"none of {IDENTITY_COLUMNS} present",
                "raw_row": json.dumps(raw_snapshot.loc[idx].to_dict(), default=str),
            })

        dropped = len(row_errors)
        if dropped:
            print(f"Dropping {dropped} row(s) with no trial code, pellet id or extrusion id")
        rows_total = len(df)
        df = df[has_identity]

        if df.empty:
            raise ValueError("No rows remain after removing rows with no identity")

        # Reorder exactly to BigQuery schema.
        df = df[TABLE_COLUMNS]

        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()

        print(f"Loaded {len(df)} rows to {table_id}")

        move_blob(bucket_name, blob_name, PROCESSED_PREFIX)

        delete_stale_failed_copy(bucket_name, blob_name)

        write_row_errors(row_errors, source_file=gcs_uri, checksum=checksum)

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="success",
            rows_total=rows_total,
            rows_inserted=len(df),
            rows_rejected=dropped,
            error_message=None,
        )

    except Exception as exc:
        # ---------------------------------------------------------------
        # RESTORED 20 Aug 2026.
        #
        # The deployed version of this block printed the traceback and then
        # returned normally. That meant a failed file was never moved to the
        # failed folder, the function reported success to Eventarc, and the
        # file stayed in to-be-processed where the uploader's blob.exists()
        # check treated every future copy as a duplicate. Failures were
        # completely invisible.
        #
        # The two behaviours below are what make a failure observable:
        #   1. move the file so its location reflects its state
        #   2. re-raise so Eventarc and Cloud Logging record an error
        #
        # Re-raising is safe here because the trigger uses
        # RETRY_POLICY_DO_NOT_RETRY, so this will not loop.
        # ---------------------------------------------------------------
        print(f"EXTRUSION_PIPELINE_FAILURE file={blob_name} error={exc}")
        print(traceback.format_exc())

        try:
            if storage_client.bucket(bucket_name).blob(blob_name).exists():
                move_blob(bucket_name, blob_name, FAILED_PREFIX)
                print(f"Moved failed file to {FAILED_PREFIX}/{blob_name.split('/')[-1]}")
            else:
                print("Blob no longer exists, nothing to move")
        except Exception as move_exc:
            print(f"EXTRUSION_PIPELINE_FAILURE could not move to failed prefix: {move_exc}")
            print(traceback.format_exc())

        write_manifest(
            source_file=gcs_uri,
            checksum=checksum,
            status="failed",
            rows_total=None,
            rows_inserted=0,
            rows_rejected=0,
            error_message=str(exc)[:1500],
        )

        raise
