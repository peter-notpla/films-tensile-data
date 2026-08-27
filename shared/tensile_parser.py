"""Pure tensile CSV parser: bytes in, dataframe out, no cloud clients.

Extracted unchanged from films-tensile-csv-processor/main.py on 27 August
2026 (Phase 2.1). Behaviour must stay byte-for-byte identical to that
version; see shared/verify_tensile_parser.py for the replay check against
every real processed file, and pipeline-roadmap.md item 2.1 for why this
exists (the parser needed to be runnable without deploying it).
"""

import io
import json
from datetime import datetime, timezone

import pandas as pd

from shared.id_validation import validation_status

FOOTER_LABELS = {"mean", "sd", "min", "max"}

# Identity constants for the specimen key model (Phase 2.2). Match main.py's
# own MACHINE_ID/PIPELINE_NAME exactly; kept here too (a small, deliberate
# duplication of two fixed strings, not the column-list kind of drift risk
# Phase 2.3 is about) so the parser can build specimen_key without needing
# config passed in.
MACHINE_ID = "tensiletester-1"
PIPELINE_NAME = "tensile"

# The single definition of films_tensile_results' column set (Phase 2.3).
# main.py's load_to_bigquery() selects and orders by this instead of holding
# its own copy. Deliberately excludes row_num: that column exists on the
# live table but nothing has populated it in years, a fossil from before
# this pipeline's current form, not a current field to perpetuate.
TABLE_COLUMNS = [
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
    "row_state",
    "database_revision",
    "archived_at",
    "archived_by",
    "revised_at",
    "revised_by",
]


def _is_footer_row(first_cell: str) -> bool:
    if first_cell is None:
        return False
    return str(first_cell).strip().lower() in FOOTER_LABELS


def extract_relevant_dataframe(csv_bytes: bytes, source_file: str):
    """
    Rules from you:
      - First row irrelevant (title line) -> drop it
      - Second row contains headers
      - Four footer rows start with Mean/SD/Min/Max in first column -> ignore them
      - Valuable rows are between header and footer

    Returns (df, rows_dropped, row_errors). rows_dropped is the count of rows
    removed for missing a sample number or having an unparseable timestamp,
    since both are required by the specimen key model. row_errors is a list
    of dicts (row_number, reason, raw_row) for those same rows, one per row,
    for the row-errors table.
    """
    text = csv_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("CSV too short (needs title + header + data)")

    # Row 1 is the VectorPro template name. Previously discarded outright;
    # captured now as provenance (Phase 2.2 key model) since copying a
    # template to make a major edit is how sample numbering resets, and a
    # distinct template name is the only thing that records which version
    # of the test produced this file.
    template_name = lines[0].strip()

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

    # Clean index so row_number below is a stable 1-based position within
    # the data block, and so out's index lines up with df's for the bad-row
    # lookup at the end.
    df = df.reset_index(drop=True)

    out = pd.DataFrame()

    out["sample"] = pd.to_numeric(df["Sample"], errors="coerce").astype("Int64")
    out["youngs_modulus_mpa"] = pd.to_numeric(df.get("Young's Modulus (MPa)", ""), errors="coerce")
    out["offset_yield_mpa"] = pd.to_numeric(df.get("Offset Yield (MPa)", ""), errors="coerce")
    out["max_load_n"] = pd.to_numeric(df.get("Max Load (N) (N)", ""), errors="coerce")
    out["max_stress_mpa"] = pd.to_numeric(df.get("Max Stress (MPa) (MPa)", ""), errors="coerce")
    out["break_pct"] = pd.to_numeric(df.get("Break (%)", ""), errors="coerce")
    out["toughness_mpa"] = pd.to_numeric(df.get("Toughness (MPa)", ""), errors="coerce")

    ts_raw = df.get("Timestamp - Start ", "").astype(str).str.strip()

    out["timestamp_start"] = pd.to_datetime(
        ts_raw,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    mask = out["timestamp_start"].isna() & ts_raw.ne("")
    out.loc[mask, "timestamp_start"] = pd.to_datetime(
        ts_raw[mask],
        format="%Y-%m-%d %H:%M",
        errors="coerce"
    )

    mask = out["timestamp_start"].isna() & ts_raw.ne("")
    out.loc[mask, "timestamp_start"] = pd.to_datetime(
        ts_raw[mask],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce"
    )

    # This final fallback used to pass errors="raise", which meant a single
    # row with a timestamp in none of the four formats killed the entire
    # file rather than just that row. Bad rows are now routed to the
    # row-errors table below instead.
    mask = out["timestamp_start"].isna() & ts_raw.ne("")
    out.loc[mask, "timestamp_start"] = pd.to_datetime(
        ts_raw[mask],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    # Leading/trailing whitespace carries no information and is invisible in
    # every UI, but silently breaks exact-match lookups (e.g. the 1264/1279
    # roll code confusion traced back to values like " AO 260701 LR 1379").
    # Extrusion already trims on ingestion; tensile did not until now.
    out["pellet_id"] = df.get("Pellet ID (Prompt For Value - Before Test)", "").astype(str).str.strip()
    out["extrusion_id"] = df.get("Extrusion ID (Prompt For Value - Before Test)", "").astype(str).str.strip()
    out["test_direction"] = df.get("Test Direction (Prompt For Value - Before Test)", "").astype(str).str.strip()
    out["sample_number"] = df.get("Sample Number  (Prompt For Value - Before Test)", "").astype(str).str.strip()
    out["sample_thickness_mm"] = pd.to_numeric(
        df.get("Sample Thickness (mm) (Prompt For Value - Before Test)", ""), errors="coerce"
    )
    out["relative_humidity_pct"] = pd.to_numeric(
        df.get("Relative Humidity (%) (Prompt For Value - Before Test)", ""), errors="coerce"
    )
    out["notes"] = df.get("Notes (Prompt For Value - After Test)", "").astype(str).str.strip()
    out["user_initials"] = df.get("User Initials (Prompt For Value - After Test)", "").astype(str).str.strip()

    # Ingestion metadata
    out["source_file"] = source_file
    out["processed_at"] = datetime.now(timezone.utc)

    # A row needs both a sample number and a parseable timestamp to satisfy
    # the specimen key model (timestamp_minute + sample). Anything else is
    # unusable: route it to the row-errors table with the raw values and
    # reason, instead of silently dropping it.
    bad_sample = out["sample"].isna()
    bad_timestamp = out["timestamp_start"].isna()
    bad_mask = bad_sample | bad_timestamp

    row_errors = []
    for idx, raw_row in df.loc[bad_mask].iterrows():
        reasons = []
        if bad_sample.loc[idx]:
            reasons.append("missing or non-numeric sample number")
        if bad_timestamp.loc[idx]:
            reasons.append(f"unparseable timestamp: {ts_raw.loc[idx]!r}")
        row_errors.append({
            "row_number": int(idx) + 1,
            "reason": "; ".join(reasons),
            "raw_row": json.dumps(raw_row.to_dict()),
        })

    out = out[~bad_mask]
    rows_dropped = len(row_errors)
    if len(out) == 0:
        raise ValueError("No valid specimen rows (sample column empty after cleaning)")

    # Specimen key model (Phase 2.2). timestamp_minute is a genuine minute
    # truncation, not a formatting nicety: Excel strips seconds during the
    # manual check step, so any key tolerant of that workflow must be built
    # at minute resolution. The %Y-%m-%dT%H:%M format is deliberate and must
    # stay in sync with the SQL backfill used for historical rows
    # (FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M', ...)) - a mismatched format would
    # give the same real specimen two different key strings depending on
    # which path produced its row.
    out["template_name"] = template_name
    out["timestamp_minute"] = out["timestamp_start"].dt.floor("min")
    minute_str = out["timestamp_minute"].dt.strftime("%Y-%m-%dT%H:%M")
    out["specimen_key"] = MACHINE_ID + "|" + PIPELINE_NAME + "|" + minute_str + "|" + out["sample"].astype(str)

    # ID format validation (Phase 3.1). Flag, don't reject - see
    # shared/id_validation.py.
    out["validation_status"] = [
        validation_status(p, e) for p, e in zip(out["pellet_id"], out["extrusion_id"])
    ]

    return out, rows_dropped, row_errors
