"""ID format validation, shared across all three parsers (Phase 3.1).

Flag, don't reject: a malformed pellet_id/extrusion_id is recorded as such,
never used to drop a row. See CLAUDE.md's "ID formats, decoded field by
field" section for what each field means; a hard reject would have
discarded roughly 150 legitimate one-off packaging study rows to catch 25
typos.

The same two regexes back the BigQuery backfill that computed
validation_status for every pre-existing row
(`UPDATE ... SET validation_status = CASE WHEN REGEXP_CONTAINS(...`); keep
them in sync if either ever changes.
"""

import re

import pandas as pd

PELLET_ID_RE = re.compile(r"^[A-Z]{2} [A-Z]{2} [A-Z]{2} [A-Z]{2} [0-9]{6} [A-Z]{2} [A-Z]{2} [0-9]{4}$")
EXTRUSION_ID_RE = re.compile(r"^[A-Z]{2} [0-9]{6} [A-Z]{2} [0-9]{4}$")


def _as_text(value):
    # A blank cell in a column pandas has inferred as numeric (extrusion's
    # parser doesn't force dtype=str, unlike tensile/friction) arrives here
    # as a float NaN, not None or "". NaN is truthy in Python, so a plain
    # `value or ""` lets it through to re.match() as a float and crashes.
    # pd.isna() catches None, NaN and NaT uniformly.
    return "" if pd.isna(value) else str(value)


def validation_status(pellet_id, extrusion_id):
    """Returns 'valid', 'invalid_pellet_id', 'invalid_extrusion_id', or
    'invalid_both'. None, NaN, or empty values are treated as not matching,
    same as BigQuery's REGEXP_CONTAINS(NULL, ...) evaluating to not-true."""
    pellet_ok = bool(PELLET_ID_RE.match(_as_text(pellet_id)))
    extrusion_ok = bool(EXTRUSION_ID_RE.match(_as_text(extrusion_id)))
    if pellet_ok and extrusion_ok:
        return "valid"
    if not pellet_ok and not extrusion_ok:
        return "invalid_both"
    if not pellet_ok:
        return "invalid_pellet_id"
    return "invalid_extrusion_id"
