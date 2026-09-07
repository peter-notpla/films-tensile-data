"""Best-effort linking of a raw curve file back to the specimen it belongs
to. Not pure (queries BigQuery live), same reasoning as
shared/revision_handling.py for keeping this out of curve_parser.py.

The raw curve file carries no pellet ID, extrusion ID, or real test
timestamp - only a bare sample number in its filename, which CLAUDE.md's
"Sample numbers are not stable identifiers" section already establishes
can't be trusted as a join key (confirmed again here: real raw files exist
with sample numbers that have no matching row in the summary table at all).
So linking goes by time proximity instead: the raw file's GCS creation
time as a proxy for when the test happened, matched against the nearest
`timestamp_start` in the summary table, within a generous window. Never
blocking: if nothing is found within the window, the caller stores NULL
rather than failing the file.

**template_name is now a required match, added 4 September 2026.** The
curve filename and the results row both carry the VectorPro template name
(`raw-<template>-sample-<n>.csv` vs. row 1 of the results CSV), and they
match cleanly on every real file checked. Requiring it alongside nearest-
time is a pure precision improvement for live traffic (rules out matching
a `TensileTest-Films[WIP](V1)` curve to a same-minute `TensileTest-
Films(V1)` specimen) with no coverage cost, since a live-triggered file's
GCS creation time already closely tracks its real test time.

**Does not help the historical backfill, and cannot be made to**: the
per-file GCS creation timestamp this function needs is only meaningful for
a file still sitting at its original upload path. `move_blob` (in each
raw-processor's main.py) moves a successfully-processed file via
`copy_blob` + `delete`, which gives the copy a new object generation and
therefore a new `time_created` reflecting the move, not the original
upload - confirmed on a real example 4 September 2026 (`sample-30.csv`:
`link_time_delta_seconds` recorded 69 seconds at ingest time; the same
file's current blob metadata, in the processed folder, implies a nearest
same-template candidate over 4,000 minutes away). No audit-log trail of
the original creation event exists either (checked: no matching
`storage.objects.create` entries for real filenames). The original signal
this function needs is gone for any file already moved, so widening the
window cannot recover historical coverage - only going-forward accuracy.

**find_specimen_link_by_sample - a deliberate lower-confidence fallback,
added 4 September 2026, Peter-approved.** Friction's historical backfill
coverage via the above was 2 linked specimens out of 928 curve files - the
GCS-time signal is gone for nearly all of it, same reasoning as above.
Matching by (template_name, sample number) alone instead recovers 387 of
928 files (13 pellets) on real data, checked before shipping. This is
exactly the join CLAUDE.md's "Sample numbers are not stable identifiers"
section warns against trusting alone - VectorPro resets the counter
whenever a template is copied, so the same template_name reused across
two separate physical sessions could match a curve file to the wrong
specimen. Mitigated, not eliminated: only returns a match when the
(template_name, sample) pair resolves to exactly one specimen_key among
current rows - an ambiguous pair (more than one specimen shares it)
returns None rather than guessing, same fail-open behavior as the
time-based function. Every caller must write the returned method
('time' vs 'sample_number') to a `link_method` column so a lower-
confidence link stays visually distinguishable downstream - never treat
its result as equivalent to a time-based match.

**find_specimen_link_by_mapped_sample - third tier, added 7 September
2026, Peter-approved.** Tried after find_specimen_link_by_sample returns
nothing: translates the raw filename's sample number through the
permanent `films_tensile_london.sample_number_map` table before matching,
recovering files whose bare sample number was itself renumbered away by
the backfill. See that function's docstring and
scripts/build_sample_number_map.py for how the map is built and why
ambiguous mappings (confirmed common, not rare, for tensile) are never
guessed at. Callers should write link_method='mapped_sample' when this
tier is what produced the link.
"""

from google.cloud import bigquery

DEFAULT_WINDOW_MINUTES = 30
DEFAULT_MAP_TABLE = "notpla-machine-data.films_tensile_london.sample_number_map"

# Confirmed 7 September 2026, while investigating why direct sample matching
# recovered 0 of 541 remaining friction files: the raw curve filename's
# parsed template ("FrictionTest-Films(V1)", from the export-time filename
# convention) doesn't match every current row it should. A live template
# rename (Films -> FilmsOld, the CLAUDE.md-documented pattern of copying a
# template and giving the copy a distinct name) left every FilmsOld summary
# row's template_name updated to the new name, while its historical raw
# curve files were never renamed to match. Safe to bridge explicitly:
# checked directly that the two templates' current sample-number ranges are
# fully disjoint (FilmsOld is exclusively >= 1,000,000, Films exclusively
# below - zero shared sample values), so trying both names can never
# introduce the same-sample-different-template collision the template
# requirement exists to prevent. Not a general template-matching relaxation -
# only this one verified, named pair.
TEMPLATE_ALIASES = {
    "frictiontest-films(v1)": ["FrictionTest-FilmsOld(V1)"],
    "frictiontest-filmsold(v1)": ["FrictionTest-Films(V1)"],
}


def find_specimen_link(bq_client, table_id, gcs_created_at, template_name, window_minutes=DEFAULT_WINDOW_MINUTES):
    """table_id: fully-qualified `project.dataset.table` for the pipeline's
    own *_all_revisions table. Only matches against row_state = 'current'
    rows, so a curve point links to the authoritative specimen, not an
    archived/superseded revision (see CLAUDE.md's "row_state"/
    "database_revision" section).

    template_name: the curve file's own template (from
    shared/curve_parser.parse_filename), matched case-insensitively against
    the results table's template_name. A curve file with no comparable
    template match simply won't link, same as one outside the time window.

    Returns (specimen_key, time_delta_seconds) - both None if nothing is
    within the window. time_delta_seconds is always non-negative and is
    returned even on a match, so confidence is judgeable later rather than
    collapsed into a single boolean (a 12-second delta and a 28-minute
    delta are not the same confidence).
    """
    window_seconds = window_minutes * 60
    query = f"""
        SELECT
            specimen_key,
            ABS(TIMESTAMP_DIFF(timestamp_start, @gcs_created_at, SECOND)) AS delta_seconds
        FROM `{table_id}`
        WHERE row_state = 'current'
          AND LOWER(TRIM(template_name)) = LOWER(TRIM(@template_name))
          AND ABS(TIMESTAMP_DIFF(timestamp_start, @gcs_created_at, SECOND)) <= @window_seconds
        ORDER BY delta_seconds ASC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("gcs_created_at", "TIMESTAMP", gcs_created_at),
        bigquery.ScalarQueryParameter("template_name", "STRING", template_name),
        bigquery.ScalarQueryParameter("window_seconds", "INT64", window_seconds),
    ])
    rows = list(bq_client.query(query, job_config=job_config).result())
    if not rows:
        return None, None
    return rows[0]["specimen_key"], rows[0]["delta_seconds"]


def find_specimen_link_by_sample(bq_client, table_id, template_name, raw_sample_number):
    """Fallback for when find_specimen_link finds nothing - see this
    module's docstring for why this exists and its confidence caveat.

    Only matches against row_state = 'current' rows, same as
    find_specimen_link. Returns specimen_key, or None if the
    (template_name, sample) pair matches zero or more than one specimen.

    `sample` is STRING on films_friction_raw_all_revisions but INT64 on
    films_tensile_results_all_revisions (confirmed 7 September 2026, this
    function's first real run against tensile) - CAST to STRING so the same
    query works against either table.

    Tries template_name and, if it has one, its known alias (see
    TEMPLATE_ALIASES) together in a single query, so a genuine collision
    across the two names would still be caught by the exactly-one-match
    check below rather than silently preferring one name over the other.
    """
    candidates = [template_name] + TEMPLATE_ALIASES.get(template_name.strip().lower(), [])
    query = f"""
        SELECT specimen_key
        FROM `{table_id}`
        WHERE row_state = 'current'
          AND LOWER(TRIM(template_name)) IN UNNEST(@template_names)
          AND TRIM(CAST(sample AS STRING)) = TRIM(@sample)
        GROUP BY specimen_key
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter(
            "template_names", "STRING", [c.strip().lower() for c in candidates]
        ),
        bigquery.ScalarQueryParameter("sample", "STRING", str(raw_sample_number)),
    ])
    rows = list(bq_client.query(query, job_config=job_config).result())
    if len(rows) != 1:
        return None
    return rows[0]["specimen_key"]


def find_specimen_link_by_mapped_sample(bq_client, table_id, map_table_id, test_type,
                                         template_name, raw_sample_number):
    """Second fallback, tried after find_specimen_link_by_sample returns
    nothing - handles a raw curve file whose bare filename sample number was
    itself renumbered away by the early-2026 manual Excel backfill (CLAUDE.md's
    "Sample numbers are not stable identifiers"), via the permanent
    `sample_number_map` table (scripts/build_sample_number_map.py, built by
    joining each instrument's pre-renumbering archive export back to the live
    table on measured values).

    Peter-approved policy (7 September 2026): a raw_sample_number with more
    than one candidate current_sample - confirmed on real tensile data to be
    common (570 of 711 archived sample numbers are reused across genuinely
    different physical tests from different template generations, not a rare
    edge case) - is never guessed at. sample_number_map itself already marks
    these `is_ambiguous = TRUE` at build time; this function only ever reads
    the unambiguous rows, so an ambiguous original_sample simply yields no
    candidate here and the file stays unlinked, same fail-open behavior as
    the other two tiers.

    Returns specimen_key, or None if no unambiguous mapping exists, or if the
    mapped (template_name, current_sample) pair itself matches zero or more
    than one specimen (same ambiguity check as find_specimen_link_by_sample).
    """
    query = f"""
        SELECT current_sample
        FROM `{map_table_id}`
        WHERE test_type = @test_type
          AND original_sample = @original_sample
          AND is_ambiguous = FALSE
    """
    job_config = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("test_type", "STRING", test_type),
        # int(...): raw_sample_number is a numpy.int64 on the live ingest
        # path (from a pandas column via curve_parser.py), which the
        # BigQuery client's request serialization cannot JSON-encode -
        # confirmed live 7 September 2026 (both negative-control test files
        # failed with "Object of type int64 is not JSON serializable" the
        # moment they fell through to this tier). find_specimen_link_by_sample
        # never hit this because it stringifies its own sample parameter.
        bigquery.ScalarQueryParameter("original_sample", "INT64", int(raw_sample_number)),
    ])
    rows = list(bq_client.query(query, job_config=job_config).result())
    if len(rows) != 1:
        return None
    return find_specimen_link_by_sample(bq_client, table_id, template_name, rows[0]["current_sample"])
