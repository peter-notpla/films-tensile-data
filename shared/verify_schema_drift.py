"""Diff each pipeline's TABLE_COLUMNS against its live BigQuery table schema.

Phase 2.3 ("schema from one definition") in practice for this codebase: not
a live migration tool, a repeatable drift check. TABLE_COLUMNS in
shared/extrusion_parser.py and shared/tensile_parser.py is the single
in-repo definition each pipeline's main.py now builds its output frame
from; this script is how "nothing keeps them in agreement" (the roadmap's
own description of how row_num, sd_percent_variation and
percent_variation_end became fossils) gets checked instead of assumed.

friction is deliberately not covered here: its parser keeps whatever
columns survive rather than a fixed list (see shared/friction_parser.py's
docstring), which resists a fixed-list diff. Phase 2.4 (typed columns) is
the real fix for friction's schema looseness, not this script.

Read-only: describes tables, writes nothing.

Usage: python3 shared/verify_schema_drift.py
"""

import sys
from pathlib import Path

from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.tensile_parser import TABLE_COLUMNS as TENSILE_COLUMNS
from shared.extrusion_parser import TABLE_COLUMNS as EXTRUSION_COLUMNS

PROJECT_ID = "notpla-machine-data"

# (pipeline label, table id, parser's TABLE_COLUMNS, known-fossil columns
# that are expected to exist live but are deliberately excluded from the
# parser's list rather than something to fix here)
CHECKS = [
    (
        "tensile",
        "films_tensile_london.films_tensile_results",
        TENSILE_COLUMNS,
        {"row_num"},
    ),
    (
        "extrusion",
        "machine_collin_e25e.raw_films_extrusion",
        EXTRUSION_COLUMNS,
        set(),
    ),
]


def main():
    client = bigquery.Client(project=PROJECT_ID)
    any_unexpected_drift = False

    for label, table_id, parser_columns, known_fossils in CHECKS:
        table = client.get_table(f"{PROJECT_ID}.{table_id}")
        live_columns = {f.name for f in table.schema}
        parser_set = set(parser_columns)

        missing_from_parser = live_columns - parser_set - known_fossils
        missing_from_live = parser_set - live_columns

        print(f"--- {label} ({table_id}) ---")
        print(f"  live columns: {len(live_columns)}, parser columns: {len(parser_set)}")

        if known_fossils & live_columns:
            print(f"  known fossils present live, excluded from parser by design: {sorted(known_fossils & live_columns)}")

        if missing_from_parser:
            any_unexpected_drift = True
            print(f"  DRIFT: live column(s) not in parser's TABLE_COLUMNS: {sorted(missing_from_parser)}")
        if missing_from_live:
            any_unexpected_drift = True
            print(f"  DRIFT: parser column(s) missing from the live table (would fail on load!): {sorted(missing_from_live)}")
        if not missing_from_parser and not missing_from_live:
            print("  no drift")
        print()

    if any_unexpected_drift:
        print("Drift found - see DRIFT lines above.")
        sys.exit(1)
    else:
        print("No unexpected drift in any checked pipeline.")


if __name__ == "__main__":
    main()
