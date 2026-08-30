"""Parses a raw curve-point file (one row per timepoint) shared by both the
tensile and friction raw-samples folders - confirmed identical 5-column
format across every real file checked in both. Pure function: bytes in,
dataframe out, no cloud clients, same convention as tensile_parser.py /
friction_parser.py so it can be replayed against a real backlog without
deploying anything.
"""

import io
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "Time (s)",
    "Load (N)",
    "Displacement (mm)",
    "Stress (MPa)",
    "Strain (%)",
]

# e.g. "raw-TensileTest-Films(V1)-sample-100.csv" -> template "TensileTest-Films(V1)", sample 100.
# raw_sample_number is explicitly a different namespace from the summary
# tables' `sample` column (see CLAUDE.md's "Sample numbers are not stable
# identifiers") - never join on it directly, that's what curve_linking.py
# is for.
FILENAME_PATTERN = re.compile(r"^raw-(?P<template>.+)-sample-(?P<sample>\d+)\.csv$", re.IGNORECASE)


def parse_filename(filename):
    match = FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Filename doesn't match the expected raw-<template>-sample-<n>.csv shape: {filename}")
    return match.group("template"), int(match.group("sample"))


def extract_curve_dataframe(csv_bytes, source_file):
    """source_file: bare filename (e.g. 'raw-TensileTest-Films(V1)-sample-100.csv'),
    not a full path - matches tensile_parser.py's convention of storing the
    caller-supplied source identifier separately from parsing."""
    template_name, raw_sample_number = parse_filename(source_file)

    text = csv_bytes.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}. Found columns: {list(df.columns)}")

    out = pd.DataFrame()
    # Row position in the original file, assigned before any rows are
    # dropped, so a dropped bad row leaves a gap rather than shifting every
    # later row's number - time_s alone isn't guaranteed unique, so this is
    # the one stable position marker within the curve.
    out["row_number"] = df.index
    out["time_s"] = pd.to_numeric(df["Time (s)"], errors="coerce")
    out["load_n"] = pd.to_numeric(df["Load (N)"], errors="coerce")
    out["displacement_mm"] = pd.to_numeric(df["Displacement (mm)"], errors="coerce")
    out["stress_mpa"] = pd.to_numeric(df["Stress (MPa)"], errors="coerce")
    out["strain_pct"] = pd.to_numeric(df["Strain (%)"], errors="coerce")
    out["raw_sample_number"] = raw_sample_number
    out["template_name"] = template_name
    out["source_file"] = source_file
    out["processed_at"] = datetime.now(timezone.utc)

    numeric_cols = ["time_s", "load_n", "displacement_mm", "stress_mpa", "strain_pct"]
    all_nan_mask = out[numeric_cols].isna().all(axis=1)

    row_errors = [
        {
            "row_number": int(row_number),
            "reason": "all numeric fields unparseable",
            "raw_row": ",".join(df.loc[idx].astype(str)),
        }
        for idx, row_number in zip(df.index[all_nan_mask], out.loc[all_nan_mask, "row_number"])
    ]

    out = out[~all_nan_mask]

    if out.empty:
        raise ValueError("No valid numeric rows found")

    return out, row_errors


def downsample_curve_minmax(df, value_col="load_n", n_buckets=100):
    """Reduces a curve to at most 2 rows per bucket (its min and its max on
    value_col), bucketed by row position rather than time_s since time_s
    isn't guaranteed unique. Chosen over fixed-interval decimation because
    decimation keeps whatever lands on a fixed grid regardless of where the
    signal actually peaks - for an oscillating curve that can understate or
    even hide real amplitude depending on phase luck. Min/max-per-bucket
    always keeps the true local extremes, so a violently oscillating stretch
    can't be flattened by unlucky sampling, and a genuinely flat stretch
    isn't manufactured to look busy either. For a monotonic stretch (e.g.
    most of a tensile curve) the bucket's min and max are just its first and
    last point, so this degrades gracefully to something like decimation
    there. Output stays in original row order (by row_number) so a line plot
    draws correctly regardless of which of min/max occurred first in time.

    n_buckets=100 -> up to 200 output points per curve, chosen to keep
    tensile's smooth curves visually smooth while still resolving friction's
    slower, large-amplitude stick-slip oscillation (observed ~1.5-1.7s
    period on real files) as a visible zigzag rather than a single min/max
    span. Faster, smaller-amplitude wiggle (observed ~0.12-0.5s period on
    other real files) gets compressed into an envelope band rather than a
    fully resolved wave shape - accepted since curve data here is for visual
    comparison only, not analysis (the summary tables are the source of
    truth for that); confirmed with Peter 30 August 2026.
    """
    n = len(df)
    if n <= n_buckets * 2:
        return df

    bucket = np.arange(n) * n_buckets // n
    df = df.assign(_bucket=bucket)
    keep = set()
    for _, group in df.groupby("_bucket"):
        keep.add(group[value_col].idxmax())
        keep.add(group[value_col].idxmin())

    out = df.loc[sorted(keep)].drop(columns="_bucket")
    return out.sort_values("row_number").reset_index(drop=True)
