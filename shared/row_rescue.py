"""Reconstruction + revalidation helpers for the row-rescue Cloud Function
(pipelines/films-pipeline-row-rescue). Not used by any ingestion pipeline -
staged alongside them by scripts/deploy.sh like every other shared/ module
(it stages all of shared/*.py into whichever pipeline it deploys), but only
films-pipeline-row-rescue actually imports it.

Why this exists: films_pipeline_row_errors and the three live results
tables store each row's fields under the pipeline's *internal* snake_case
names (tensile/extrusion rename literal spreadsheet headers via a fixed
map; friction normalizes headers with a generic, idempotent rule instead).
Rebuilding a one-row CSV that shared.<pipeline>_parser will actually accept
means reconstructing whichever header text each parser is really looking
for - get it wrong and a resubmitted row either fails validation (safe, if
disappointing) or silently loads with fields shifted (not safe). This
module is the single place that knows each parser's expectations, so the
rescue function never re-derives them.
"""

import csv
import io

from shared.extrusion_parser import HEADER_MAP as _EXTRUSION_HEADER_MAP
from shared.extrusion_parser import extract_extrusion_dataframe
from shared.friction_parser import extract_friction_dataframe
from shared.tensile_parser import extract_relevant_dataframe

# Fields every pipeline adds itself (computed provenance/derived columns),
# never something a human should be editing or that any parser expects as
# an input header. Stripped before rebuilding a CSV and before rendering
# the rescue form.
DERIVED_FIELDS = {
    "source_file", "processed_at", "validation_status", "row_state",
    "database_revision", "archived_at", "archived_by", "revised_at",
    "revised_by", "specimen_key", "timestamp_minute", "template_name",
    "link_time_delta_seconds", "link_method", "linked_specimen_key",
    "row_num",
}

# Reverse of shared/tensile_parser.py's literal df.get(...) lookups in
# extract_relevant_dataframe - keep in sync if that function's column list
# ever changes.
TENSILE_HEADERS = {
    "sample": "Sample",
    "youngs_modulus_mpa": "Young's Modulus (MPa)",
    "offset_yield_mpa": "Offset Yield (MPa)",
    "max_load_n": "Max Load (N) (N)",
    "max_stress_mpa": "Max Stress (MPa) (MPa)",
    "break_pct": "Break (%)",
    "toughness_mpa": "Toughness (MPa)",
    "timestamp_start": "Timestamp - Start ",
    "pellet_id": "Pellet ID (Prompt For Value - Before Test)",
    "extrusion_id": "Extrusion ID (Prompt For Value - Before Test)",
    "test_direction": "Test Direction (Prompt For Value - Before Test)",
    "sample_number": "Sample Number  (Prompt For Value - Before Test)",
    "sample_thickness_mm": "Sample Thickness (mm) (Prompt For Value - Before Test)",
    "relative_humidity_pct": "Relative Humidity (%) (Prompt For Value - Before Test)",
    "notes": "Notes (Prompt For Value - After Test)",
    "user_initials": "User Initials (Prompt For Value - After Test)",
}

# Reverse of shared/extrusion_parser.py's HEADER_MAP (literal -> snake).
# Two literal headers map to the same snake name there ("% Variation" /
# "Variation"); reversing picks one arbitrarily, which is fine, either is
# accepted by the real parser.
EXTRUSION_HEADERS = {snake: literal for literal, snake in _EXTRUSION_HEADER_MAP.items()}


def strip_derived_fields(raw_row: dict) -> dict:
    return {k: v for k, v in raw_row.items() if k not in DERIVED_FIELDS}


def _write_csv(rows) -> bytes:
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    return buf.getvalue().encode("utf-8")


def build_tensile_csv(raw_row: dict) -> bytes:
    """extract_relevant_dataframe requires: row 1 = title (any text,
    discarded), row 2 = literal headers, then data rows, then a footer
    block whose first cell under the 'Sample' column reads Mean/SD/Min/Max
    (only that column's footer-row content matters - the rest is sliced
    away before it's ever read)."""
    headers = list(TENSILE_HEADERS.values())
    keys = list(TENSILE_HEADERS.keys())
    sample_idx = headers.index("Sample")

    data_row = [str(raw_row.get(k, "") or "") for k in keys]

    def footer_row(label):
        row = [""] * len(headers)
        row[sample_idx] = label
        return row

    rows = [
        ["Row rescue resubmission"],
        headers,
        data_row,
        footer_row("Mean"),
        footer_row("SD"),
        footer_row("Min"),
        footer_row("Max"),
    ]
    return _write_csv(rows)


def build_friction_csv(raw_row: dict) -> bytes:
    """extract_friction_dataframe requires: row 1 = title (discarded), row
    2 = headers, then data. normalize() is idempotent on already-normalized
    snake_case text, so raw_row's own keys work directly as the header
    row - no reverse map needed here, unlike tensile/extrusion."""
    headers = list(raw_row.keys())
    data_row = [str(raw_row[h]) if raw_row[h] is not None else "" for h in headers]
    rows = [["Row rescue resubmission"], headers, data_row]
    return _write_csv(rows)


_EXTRUSION_IDENTITY_PRIORITY = ["trial_code", "pellet_id", "extrusion_id"]


def build_extrusion_csv(raw_row: dict) -> bytes:
    """extract_extrusion_dataframe reads with header=1: row index 0 is
    skipped entirely (a real section-header row in production, content
    irrelevant), row index 1 is the real header. It then drops any row
    whose FIRST column is blank (real files always have a populated
    identity field there, e.g. Trial Code) - raw_row's own key order is
    arbitrary (whatever order BigQuery/JSON produced), so a column that
    happens to be blank for this row (e.g. amp_a) could land first and
    get the whole row dropped as "blank column A" before the real
    no-identity check ever runs. Force whichever identity column is
    actually populated to the front instead of trusting raw_row's order."""
    keys = list(raw_row.keys())
    first_key = next(
        (k for k in _EXTRUSION_IDENTITY_PRIORITY if raw_row.get(k)),
        None,
    )
    if first_key is not None:
        keys = [first_key] + [k for k in keys if k != first_key]

    headers = [EXTRUSION_HEADERS.get(k, k) for k in keys]
    data_row = [str(raw_row[k]) if raw_row[k] is not None else "" for k in keys]
    rows = [["placeholder section header row"], headers, data_row]
    return _write_csv(rows)


_BUILDERS = {
    "tensile": (build_tensile_csv, extract_relevant_dataframe),
    "friction": (build_friction_csv, extract_friction_dataframe),
    "extrusion": (build_extrusion_csv, extract_extrusion_dataframe),
}


def revalidate_and_rebuild(pipeline: str, raw_row: dict):
    """Runs the edited row through the exact parser the live processor
    uses. Returns (ok, error_message, csv_bytes, validation_status).
    csv_bytes is the one-row CSV to actually upload on success; None
    otherwise. validation_status is the parser's own flag-don't-reject
    verdict ('valid' / 'invalid_pellet_id' / ...) for the rebuilt row -
    structurally valid (not rejected) is not the same thing as
    validation_status == 'valid'; a rescue of a *flagged* row should
    require the latter too, which only the caller knows to check (a
    rejected-row rescue does not need to). Never raises - any parser
    exception becomes a returned error message, same as a validation
    failure, so the caller can redisplay the form."""
    builder = _BUILDERS.get(pipeline)
    if builder is None:
        return False, f"Unknown pipeline: {pipeline!r}", None, None
    build_csv, extract_dataframe = builder

    cleaned = strip_derived_fields(raw_row)
    csv_bytes = build_csv(cleaned)

    try:
        df, rows_dropped, row_errors = extract_dataframe(csv_bytes, source_file="row-rescue-resubmission")
    except Exception as exc:
        return False, str(exc), None, None

    if rows_dropped or row_errors:
        reasons = "; ".join(e["reason"] for e in row_errors) if row_errors else "row dropped"
        return False, f"Still invalid: {reasons}", None, None
    if len(df) != 1:
        return False, f"Expected exactly 1 row after parsing, got {len(df)}", None, None

    return True, None, csv_bytes, df.iloc[0]["validation_status"]
