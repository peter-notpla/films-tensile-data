"""Excel-processed detection, shared across all three pipelines (Phase 3.3).

A file whose row 1 ends in comma padding, or where every present timestamp
has zero seconds, has been through Excel during the manual check step (it
drops seconds from timestamps and pads row 1 with trailing commas on save).
Flag it, don't reject it - see CLAUDE.md's "Excel destroys precision, and
this is accepted" section. Recorded once per file on films_pipeline_manifest
(a file-level property), not per row on the results tables.

Deliberately not part of the parsers: the padding signal must be available
even when parsing fails outright (a padded row 1 can itself be the cause of
a parse failure), and the timestamp signal only applies where a
seconds-resolution timestamp exists at all - extrusion has no such field,
only a DATE with no time-of-day component, so it relies on the padding
signal alone.
"""

import pandas as pd


def is_excel_processed(csv_bytes: bytes, timestamps=None) -> bool:
    """timestamps, if given: an iterable of parsed datetime-like values
    (None/NaT entries are ignored, not counted as zero-second)."""
    lines = csv_bytes.decode("utf-8", errors="replace").splitlines()
    row1_padded = bool(lines) and lines[0].rstrip("\r").endswith(",")

    all_zero_seconds = False
    if timestamps is not None:
        present = [t for t in timestamps if t is not None and not pd.isna(t)]
        all_zero_seconds = bool(present) and all(getattr(t, "second", None) == 0 for t in present)

    return row1_padded or all_zero_seconds


def clean_template_name(raw_row1: str) -> str:
    """Row 1 is the VectorPro template name, but Excel's trailing-comma row-1
    padding (see module docstring) leaks straight into it if only whitespace
    is stripped - confirmed on real files where the same template produced
    both "TensileTest-Films(V1)" and "TensileTest-Films(V1),,,,,,,,,,,,,,,"
    depending on whether Excel had touched the file. Stripping trailing
    commas collapses both back to one value; real template names observed so
    far are plain identifiers with no legitimate trailing comma to protect.
    """
    return raw_row1.strip().rstrip(",").strip()
