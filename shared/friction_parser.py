"""Pure friction CSV parser: bytes in, dataframe out, no cloud clients.

Extracted unchanged from films-friction-csv-processor/main.py on 27 August
2026 (Phase 2.1). Behaviour must stay byte-for-byte identical to that
version; see shared/verify_friction_parser.py for the replay check against
every real processed file, and pipeline-roadmap.md item 2.1 for why this
exists (the parser needed to be runnable without deploying it).

Unlike the tensile parser, friction keeps every column that survives
cleaning rather than a fixed named set: `films_friction_raw`'s original
columns are all STRING (Phase 2.4 added typed "_num" siblings alongside
them; see NUMERIC_COLUMNS below).
"""

import io
import json
import re
from datetime import datetime, timezone

import pandas as pd

from shared.id_validation import validation_status

TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
]

FOOTER_MARKERS = {"mean", "sd", "min", "max"}

# Friction's ID columns, after normalize(), differ from tensile/extrusion's
# plain pellet_id/extrusion_id (Phase 3.1).
PELLET_ID_COLUMN = "pellet_id_prompt_for_value_before_test"
EXTRUSION_ID_COLUMN = "extrusion_code_prompt_for_value_before_test"

# Phase 2.4: films_friction_raw stores every measurement as STRING because
# this parser never called pd.to_numeric. Additive fix - a typed "_num"
# sibling is added for each of these (when the source column is present;
# older files lack some of them, e.g. backup_static_cof per CLAUDE.md), the
# original STRING columns are untouched so nothing downstream (Looker
# Studio) breaks. SAFE_CAST is mirrored in BigQuery for the historical
# backfill: blank/unparseable values become NULL, not an error, consistent
# with flag-don't-reject.
NUMERIC_COLUMNS = [
    "static_friction_force_magnitude_1_n",
    "backup_peak_n",
    "dynamic_friction_force_n",
    "static_coefficient_of_friction",
    "backup_static_cof",
    "dynamic_coefficientof_friction",
    "sample_number_prompt_for_value_before_test",
    "sample_repeat_number_prompt_for_value_before_test",
    "pctrh_prompt_for_value_before_test",
]

# Identity constants for the specimen key model (Phase 2.2). Match main.py's
# own MACHINE_ID/PIPELINE_NAME exactly; kept here too (a small, deliberate
# duplication of two fixed strings, not the column-list kind of drift risk
# Phase 2.3 is about) so the parser can build specimen_key without needing
# config passed in.
MACHINE_ID = "tensiletester-1"
PIPELINE_NAME = "friction"


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


def extract_friction_dataframe(csv_bytes: bytes, source_file: str):
    """
    Returns (df, rows_dropped, row_errors). rows_dropped is the count of
    rows removed for a blank sample or an unparseable timestamp_start, since
    both are required by the specimen key model. row_errors is a list of
    dicts (row_number, reason, raw_row) for those same rows, one per row,
    for the row-errors table.
    """
    # Row 1 (never loaded by the header=1 read below, since pandas skips it
    # entirely) is the VectorPro template name - provenance for the
    # specimen key model (Phase 2.2), same reasoning as the tensile parser.
    first_line = csv_bytes.decode("utf-8", errors="replace").splitlines()
    template_name = first_line[0].strip() if first_line else ""

    try:
        df = pd.read_csv(io.BytesIO(csv_bytes), header=1, dtype=str,
                          encoding="utf-8", keep_default_na=False)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(csv_bytes), header=1, dtype=str,
                          encoding="latin1", keep_default_na=False)

    cols = [c for c in df.columns if str(c).strip() and not str(c).startswith("Unnamed:")]
    df = df[cols]

    first_col = df.columns[0]
    df = df[~df[first_col].astype(str).str.lower().isin(FOOTER_MARKERS)]

    # A fully blank row (every column empty, e.g. Excel's trailing padding)
    # carries nothing and is dropped silently here. A row that's blank in
    # some columns but not others is real data with a problem and is
    # handled below instead, not swallowed here.
    blanked = df.replace(r"^\s*$", "", regex=True)
    df = df[~(blanked == "").all(axis=1)]

    if df.empty:
        raise ValueError("No data rows")

    df.columns = [normalize(c) for c in df.columns]

    if "sample" not in df.columns:
        raise ValueError("Missing sample column")

    if "timestamp_start" not in df.columns:
        raise ValueError("Missing timestamp column")

    # Row_number below is a stable 1-based position within the data block,
    # and the reset lets the raw-value snapshot and the parsed masks below
    # line up by index.
    df = df.reset_index(drop=True)
    raw_values = df.copy()

    # A row needs both a sample number and a parseable timestamp to be
    # usable. Bad rows are dropped individually and routed to the
    # row-errors table with the raw values and reason; the rest of the file
    # still loads.
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

    df = df[~bad_mask].copy()
    rows_dropped = len(row_errors)
    if df.empty:
        raise ValueError("No valid rows remain after dropping blank samples / unparseable timestamps")

    # Leading/trailing whitespace carries no information and is invisible in
    # every UI, but silently breaks exact-match lookups. Every surviving
    # column is still a string at this point (read with dtype=str above), so
    # a blanket strip is safe; timestamp_start is overwritten immediately
    # below regardless.
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    df["timestamp_start"] = parsed_ts[~bad_mask]
    df["timestamp_start"] = pd.to_datetime(df["timestamp_start"], utc=True)

    df["source_file"] = source_file
    df["processed_at"] = pd.Timestamp.now(tz="UTC")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[f"{col}_num"] = pd.to_numeric(df[col], errors="coerce")

    # Specimen key model (Phase 2.2). The %Y-%m-%dT%H:%M format is
    # deliberate and must stay in sync with the SQL backfill used for
    # historical rows (FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M', ...)) - a
    # mismatched format would give the same real specimen two different key
    # strings depending on which path produced its row.
    df["template_name"] = template_name
    df["timestamp_minute"] = df["timestamp_start"].dt.floor("min")
    minute_str = df["timestamp_minute"].dt.strftime("%Y-%m-%dT%H:%M")
    df["specimen_key"] = MACHINE_ID + "|" + PIPELINE_NAME + "|" + minute_str + "|" + df["sample"]

    # ID format validation (Phase 3.1). Flag, don't reject - see
    # shared/id_validation.py. Friction's ID columns are named differently
    # from tensile/extrusion's (pellet_id_prompt_for_value_before_test /
    # extrusion_code_prompt_for_value_before_test, not pellet_id /
    # extrusion_id), and, being dynamic-schema, may not exist in every file.
    pellet_vals = df.get(PELLET_ID_COLUMN, pd.Series([""] * len(df), index=df.index))
    extrusion_vals = df.get(EXTRUSION_ID_COLUMN, pd.Series([""] * len(df), index=df.index))
    df["validation_status"] = [
        validation_status(p, e) for p, e in zip(pellet_vals, extrusion_vals)
    ]

    return df, rows_dropped, row_errors
