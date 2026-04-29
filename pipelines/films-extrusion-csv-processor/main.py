import os
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

storage_client = storage.Client(project=PROJECT_ID)
bq_client = bigquery.Client(project=PROJECT_ID)

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
    "Amp \n(A)": "amp_a",
    "Torque \n(%)": "torque_percent",
    "Die pressure (bar)": "die_pressure_bar",
    "Melt temp \n(°C)": "melt_temp_c",
    "Calender Set Temp\n(°C)": "calender_set_temp_c",
    "Calender Actual Temp\n(°C)": "calender_actual_temp_c",
    "Calender Speed (m/min)": "calender_speed_m_min",
    "Middle Roller Speed\n(%)": "middle_roller_speed_percent",
    "Spooling Reel Torque\n(Nm)": "spooling_reel_torque_nm",

    "Left Film Thickness 1 (um)": "left_film_thickness_1_um",
    "Film Thickness 2 (um)": "film_thickness_2_um",
    "Film Thickness 3 (um)": "film_thickness_3_um",
    "Film Thickness 4 (um)": "film_thickness_4_um",
    "Film Thickness 5 (um)": "film_thickness_5_um",
    "Film Thickness 6 (um)": "film_thickness_6_um",
    "Film Thickness 7 (um)": "film_thickness_7_um",
    "Film Thickness 8 (um)": "film_thickness_8_um",
    "Film Thickness 9 (um)": "film_thickness_9_um",
    "Right Film Thickness 10 (um)": "right_film_thickness_10_um",
    "Average Thickness (um)": "average_thickness_um",
    "SD \n% Variation": "sd_percent_variation",
    "Left Film Thickness 1 End (um)": "left_film_thickness_1_end_um",
    "Film Thickness 2 End (um)": "film_thickness_2_end_um",
    "Film Thickness 3 End (um)": "film_thickness_3_end_um",
    "Film Thickness 4 End (um)": "film_thickness_4_end_um",
    "Film Thickness 5 End (um)": "film_thickness_5_end_um",
    "Film Thickness 6 End (um)": "film_thickness_6_end_um",
    "Film Thickness 7 End (um)": "film_thickness_7_end_um",
    "Film Thickness 8 End (um)": "film_thickness_8_end_um",
    "Film Thickness 9 End (um)": "film_thickness_9_end_um",
    "Right Film Thickness 10 End (um)": "right_film_thickness_10_end_um",
    "Average Thickness End (um)": "average_thickness_end_um",
    "SD End": "sd_end",
    "% Variation End": "percent_variation_end",
    "Width (mm)": "width_mm",
    "Length (m)": "length_m",
    "Pellets Moisture Content (%)": "pellets_moisture_content_percent",
    "Relative Humidity \n(%)": "relative_humidity_percent",
    "Temperature \n(°C)": "temperature_c",
    "Comments": "comments",
    "Key": "key",
}

FLOAT_COLUMNS = [
    "zone_1", "zone_2", "zone_3", "zone_4", "zone_5",
    "zone_7", "zone_8", "zone_9", "zone_10", "zone_11",
    "screw_speed_rpm", "amp_a", "torque_percent", "die_pressure_bar",
    "melt_temp_c", "calender_set_temp_c", "calender_actual_temp_c",
    "calender_speed_m_min", "middle_roller_speed_percent",
    "spooling_reel_torque_nm",
    "left_film_thickness_1_um", "film_thickness_2_um",
    "film_thickness_3_um", "film_thickness_4_um",
    "film_thickness_5_um", "film_thickness_6_um",
    "film_thickness_7_um", "film_thickness_8_um",
    "film_thickness_9_um", "right_film_thickness_10_um",
    "average_thickness_um", "sd_percent_variation",
    "left_film_thickness_1_end_um", "film_thickness_2_end_um",
    "film_thickness_3_end_um", "film_thickness_4_end_um",
    "film_thickness_5_end_um", "film_thickness_6_end_um",
    "film_thickness_7_end_um", "film_thickness_8_end_um",
    "film_thickness_9_end_um", "right_film_thickness_10_end_um",
    "average_thickness_end_um", "sd_end",
    "percent_variation_end",
    "width_mm", "length_m",
    "pellets_moisture_content_percent",
    "relative_humidity_percent", "temperature_c"
]

def clean_header(h):
    if pd.isna(h):
        return None
    return str(h).strip()


def parse_date(val):
    if pd.isna(val) or str(val).strip() == "":
        return None
    try:
        return datetime.strptime(str(val).strip(), "%d/%m/%y").date()
    except Exception:
        return None


def move_blob(bucket, blob_name, new_prefix):
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob_name)
    new_name = f"{new_prefix}/{blob_name.split('/')[-1]}"
    bucket.copy_blob(blob, bucket, new_name)
    blob.delete()


@functions_framework.cloud_event
def process_file(cloud_event):
    data = cloud_event.data
    bucket_name = data["bucket"]
    blob_name = data["name"]

    print(f"Received file: {blob_name}")

    if not blob_name.startswith(WATCH_PREFIX):
        print("Skipping: outside watch folder")
        return

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()

        try:
            df = pd.read_csv(BytesIO(content), header=1)
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), header=1, encoding="latin-1")

        # Clean headers
        df.columns = [clean_header(c) for c in df.columns]

        # Drop empty columns
        df = df.loc[:, df.columns.notnull()]

        # Keep rows where Trial Code exists
        df = df[df["Trial Code"].notna()]

        # Rename columns
        df = df.rename(columns=HEADER_MAP)

        # Parse date
        if "date" in df.columns:
            df["date"] = df["date"].apply(parse_date)

        # Cast floats safely
        for col in FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add metadata
        df["source_file"] = blob_name.split("/")[-1]
        df["processed_at"] = datetime.now(timezone.utc)

        # Load to BigQuery
        table_id = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
        job = bq_client.load_table_from_dataframe(df, table_id)
        job.result()

        print("Load complete")

        move_blob(bucket_name, blob_name, PROCESSED_PREFIX)

    except Exception as e:
        print("Processing failed")
        print(traceback.format_exc())

        move_blob(bucket_name, blob_name, FAILED_PREFIX)
        raise
