"""Pure extrusion CSV parser: bytes in, dataframe out, no cloud clients.

Extracted unchanged from films-extrusion-csv-processor/main.py on 27 August
2026 (Phase 2.1). Behaviour must stay byte-for-byte identical to that
version; see shared/verify_extrusion_parser.py for the replay check against
every real processed file, and pipeline-roadmap.md item 2.1 for why this
exists (the parser needed to be runnable without deploying it).
"""

import json
from datetime import datetime, timezone
from io import BytesIO

import pandas as pd

from shared.id_validation import validation_status

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
    "sd", "variation", "variation_end", "validation_status"
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
# extrusion exports. See the checks in extract_extrusion_dataframe().
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


def extract_extrusion_dataframe(csv_bytes: bytes, source_file: str):
    """
    Returns (df, rows_dropped, row_errors). rows_dropped is the count of
    rows removed for having none of trial_code / pellet_id / extrusion_id
    present, since a row with no identity cannot be joined to anything.
    row_errors is a list of dicts (row_number, reason, raw_row) for those
    same rows, one per row, for the row-errors table.

    Raises ValueError if the file doesn't look like an extrusion export at
    all (too few recognised columns, or no identity column present).
    """
    try:
        df = pd.read_csv(BytesIO(csv_bytes), header=1)
    except UnicodeDecodeError:
        df = pd.read_csv(BytesIO(csv_bytes), header=1, encoding="latin-1")

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
    df["source_file"] = source_file
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
    df = df[has_identity]

    if df.empty:
        raise ValueError("No rows remain after removing rows with no identity")

    # ID format validation (Phase 3.1). Flag, don't reject - see
    # shared/id_validation.py. Computed after the "ensure all table columns
    # exist" pass above overwrites pellet_id/extrusion_id with None where
    # absent, so a genuinely missing ID is correctly treated the same as an
    # empty one.
    df["validation_status"] = [
        validation_status(p, e) for p, e in zip(df["pellet_id"], df["extrusion_id"])
    ]

    # Reorder exactly to BigQuery schema.
    df = df[TABLE_COLUMNS]

    return df, dropped, row_errors
