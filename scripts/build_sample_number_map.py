"""One-time (re-runnable) reconciliation: rebuilds
`films_tensile_london.sample_number_map`, translating the pre-backfill
sample number a raw curve filename carries into the current summary-table
sample number, for every specimen caught in the early-2026 manual Excel
renumbering (CLAUDE.md's "Sample numbers are not stable identifiers").

Method: join each instrument's archived pre-renumbering export back to the
live table on measured values the renumbering never touched, not on any
assumed arithmetic. See CLAUDE.md and pipeline-roadmap.md's "Sample number
map" entry for the full account of why arithmetic alone isn't trusted here
even for friction, where a constant offset was suspected.

Tensile: join key is (timestamp floored to the minute, max_stress_mpa
rounded to 3dp, youngs_modulus_mpa rounded to 3dp). The archive file has no
`sample_number` (hand-entered repeat) column duplicate risk here since
`sample` is what the raw curve filename carries.

Friction: 14 of the archive's 15 distinct source files carry no timestamp
column at all (checked directly, not assumed - only the 20260413-124733
export has one), so the join key here is instead (pellet_id, extrusion_id,
test_surface, sample_repeat_number, and five rounded force/CoF measurements).
Confirmed before use: this file set is a near-exact source for the current
1,000,000-series friction rows (531 archived rows for 536 live rows), not a
partial/unrelated sample.

Ambiguous matches (Peter's call, both directions - one original_sample
resolving to several current_sample values, e.g. the resequencing's known
duplicate-numbering artefact, or vice versa) are written with
`is_ambiguous = TRUE` and left for manual review, never guessed at or
collapsed. Nothing that reads this table should treat an ambiguous row as
a usable link.

Usage:
    python3 scripts/build_sample_number_map.py [--dry-run]

--dry-run prints the reconciliation report without writing to BigQuery.
"""

import argparse
import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery, storage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ID = "notpla-machine-data"
MAP_TABLE = f"{PROJECT_ID}.films_tensile_london.sample_number_map"

TENSILE_ARCHIVE_URI = (
    "gs://notpla-machine-data/machine-tensiletester-1/tensiletester-films/"
    "tensiletester-films-tensile/tensiletester-films-tensile-archive/"
    "reconciled-20260820/Results-TensileTest-Films(V1)-20260403-190500.csv"
)
FRICTION_ARCHIVE_PREFIX = (
    "machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/"
    "tensiletester-films-friction-archive/reconciled-20260820/"
)

FOOTER_LABELS = {"mean", "sd", "min", "max"}


def _drop_footer(df, first_col):
    footer_pos = None
    for i in range(len(df)):
        if str(df.iloc[i][first_col]).strip().lower() in FOOTER_LABELS:
            footer_pos = i
            break
    return df.iloc[:footer_pos].copy() if footer_pos is not None else df


def load_tensile_archive(bq_client=None):
    """Returns a dataframe of (original_sample, timestamp_minute,
    max_stress_r, ym_r) from the archive export. Not shared/tensile_parser.py:
    that parser expects a live-format 'Timestamp - Start ' column (trailing
    space); this archive export's header has no trailing space, confirmed by
    inspection, so a bespoke tolerant loader is used instead. Production
    parsing is untouched."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket_name = TENSILE_ARCHIVE_URI.split("/")[2]
    blob_path = "/".join(TENSILE_ARCHIVE_URI.split("/")[3:])
    content = storage_client.bucket(bucket_name).blob(blob_path).download_as_bytes()

    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    trimmed = "\n".join(lines[1:])  # drop row-1 template name
    df = pd.read_csv(io.StringIO(trimmed), dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df = _drop_footer(df, "Sample")

    out = pd.DataFrame()
    out["original_sample"] = pd.to_numeric(df["Sample"], errors="coerce").astype("Int64")
    out["max_stress_r"] = pd.to_numeric(df["Max Stress (MPa) (MPa)"], errors="coerce").round(3)
    out["ym_r"] = pd.to_numeric(df["Young's Modulus (MPa)"], errors="coerce").round(3)
    ts = pd.to_datetime(df["Timestamp - Start"].str.strip(), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    mask = ts.isna()
    ts.loc[mask] = pd.to_datetime(df["Timestamp - Start"].str.strip()[mask], errors="coerce")
    out["timestamp_minute"] = ts.dt.floor("min").dt.tz_localize("UTC")
    out = out.dropna(subset=["original_sample", "timestamp_minute", "max_stress_r", "ym_r"])
    return out


def load_friction_archive():
    """Returns a dataframe of (original_sample, pellet_id, extrusion_id,
    test_surface, repeat, 5 rounded measurements) from the 15 distinct
    archive exports (each present twice under an identical-content " (1)"
    duplicate name, per the key document; deduped here by keeping only
    non-"(1)" filenames)."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(PROJECT_ID.replace("notpla-machine-data", "notpla-machine-data"))
    blobs = [
        b for b in storage_client.list_blobs(bucket, prefix=FRICTION_ARCHIVE_PREFIX)
        if b.name.endswith(".csv") and " (1)" not in b.name and " (2)" not in b.name
        and " (3)" not in b.name and " (4)" not in b.name
    ]
    frames = []
    for blob in blobs:
        content = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(content), header=1, dtype=str, keep_default_na=False)
        df.columns = [c.strip() for c in df.columns]
        df = _drop_footer(df, "Sample")

        out = pd.DataFrame()
        out["original_sample"] = pd.to_numeric(df["Sample"], errors="coerce").astype("Int64")
        out["pellet_id"] = df["Pellet ID (Prompt For Value - Before Test)"].str.strip()
        out["extrusion_id"] = df["Extrusion code (Prompt For Value - Before Test)"].str.strip()
        out["test_surface"] = df["Test Surfaces (Prompt For Value - Before Test)"].str.strip()
        out["repeat"] = df["Sample repeat number (Prompt For Value - Before Test)"].str.strip()
        out["static_force_r"] = pd.to_numeric(df["Static Friction Force (Magnitude 1) (N)"], errors="coerce").round(3)
        out["backup_peak_r"] = pd.to_numeric(df["Backup Peak (N)"], errors="coerce").round(3)
        out["dynamic_force_r"] = pd.to_numeric(df["Dynamic Friction Force (N)"], errors="coerce").round(3)
        out["static_cof_r"] = pd.to_numeric(df["Static Coefficient of Friction"], errors="coerce").round(3)
        out["dynamic_cof_r"] = pd.to_numeric(df["Dynamic CoefficientOf Friction"], errors="coerce").round(3)
        out["source_archive_file"] = blob.name.split("/")[-1]
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    return combined.dropna(subset=["original_sample"])


def fetch_current_tensile(bq_client):
    query = f"""
        SELECT
            sample AS current_sample,
            specimen_key,
            timestamp_start,
            TIMESTAMP_TRUNC(timestamp_start, MINUTE) AS timestamp_minute,
            ROUND(max_stress_mpa, 3) AS max_stress_r,
            ROUND(youngs_modulus_mpa, 3) AS ym_r
        FROM `{PROJECT_ID}.films_tensile_london.films_tensile_results_all_revisions`
        WHERE row_state = 'current' AND sample >= 1000000
    """
    return bq_client.query(query).to_dataframe()


def fetch_current_friction(bq_client):
    query = f"""
        SELECT
            SAFE_CAST(sample AS INT64) AS current_sample,
            specimen_key,
            timestamp_start,
            TRIM(pellet_id_prompt_for_value_before_test) AS pellet_id,
            TRIM(extrusion_code_prompt_for_value_before_test) AS extrusion_id,
            TRIM(test_surfaces_prompt_for_value_before_test) AS test_surface,
            TRIM(sample_repeat_number_prompt_for_value_before_test) AS repeat,
            ROUND(static_friction_force_magnitude_1_n_num, 3) AS static_force_r,
            ROUND(backup_peak_n_num, 3) AS backup_peak_r,
            ROUND(dynamic_friction_force_n_num, 3) AS dynamic_force_r,
            ROUND(static_coefficient_of_friction_num, 3) AS static_cof_r,
            ROUND(dynamic_coefficientof_friction_num, 3) AS dynamic_cof_r
        FROM `{PROJECT_ID}.machine_data.films_friction_raw_all_revisions`
        WHERE row_state = 'current' AND SAFE_CAST(sample AS INT64) >= 1000000
    """
    return bq_client.query(query).to_dataframe()


def reconcile(original_df, current_df, join_cols, test_type, method):
    """Merges on join_cols, then classifies each original_sample as a clean
    1:1 match, ambiguous (either direction has >1 counterpart), or
    unmatched. Returns the sample_number_map rows for this test_type."""
    merged = original_df.merge(current_df, on=join_cols, how="inner")

    rows = []
    now = datetime.now(timezone.utc)

    matched_originals = set(merged["original_sample"].unique())
    for orig in sorted(original_df["original_sample"].unique()):
        candidates = merged[merged["original_sample"] == orig]
        if len(candidates) == 0:
            continue  # unmatched, reported separately below
        distinct_current = candidates["current_sample"].unique()
        is_ambiguous = len(distinct_current) > 1
        for _, cand in candidates.drop_duplicates(subset=["current_sample"]).iterrows():
            # also ambiguous if this current_sample is claimed by more than
            # one distinct original_sample (reverse-direction collision)
            reverse_hits = merged[merged["current_sample"] == cand["current_sample"]]["original_sample"].unique()
            row_ambiguous = is_ambiguous or len(reverse_hits) > 1
            rows.append({
                "test_type": test_type,
                "original_sample": int(orig),
                "current_sample": int(cand["current_sample"]),
                "timestamp_start": cand["timestamp_start"],
                "mapping_method": method,
                "is_ambiguous": bool(row_ambiguous),
                "created_at": now,
            })

    n_total = original_df["original_sample"].nunique()
    n_matched = len(matched_originals)
    n_unmatched = n_total - n_matched
    n_ambiguous = len({r["original_sample"] for r in rows if r["is_ambiguous"]})
    print(f"[{test_type}] archive rows (distinct original_sample): {n_total}")
    print(f"[{test_type}] matched at least one current_sample: {n_matched}")
    print(f"[{test_type}] unmatched (no candidate found): {n_unmatched}")
    print(f"[{test_type}] ambiguous (flagged, not linked): {n_ambiguous}")
    print(f"[{test_type}] clean 1:1 mappings: {n_matched - n_ambiguous}")

    return rows, n_unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bq_client = bigquery.Client(project=PROJECT_ID)

    print("Loading tensile archive...")
    tensile_orig = load_tensile_archive()
    print(f"  {len(tensile_orig)} usable rows")
    print("Fetching current tensile 1,000,000-series rows...")
    tensile_curr = fetch_current_tensile(bq_client)
    print(f"  {len(tensile_curr)} rows")
    tensile_rows, tensile_unmatched = reconcile(
        tensile_orig, tensile_curr,
        join_cols=["timestamp_minute", "max_stress_r", "ym_r"],
        test_type="tensile", method="measurement_join",
    )

    print()
    print("Loading friction archive...")
    friction_orig = load_friction_archive()
    print(f"  {len(friction_orig)} usable rows")
    print("Fetching current friction 1,000,000-series rows...")
    friction_curr = fetch_current_friction(bq_client)
    print(f"  {len(friction_curr)} rows")
    friction_rows, friction_unmatched = reconcile(
        friction_orig, friction_curr,
        join_cols=["pellet_id", "extrusion_id", "test_surface", "repeat",
                   "static_force_r", "backup_peak_r", "dynamic_force_r",
                   "static_cof_r", "dynamic_cof_r"],
        test_type="friction", method="measurement_join",
    )

    all_rows = tensile_rows + friction_rows
    print()
    print(f"Total sample_number_map rows to write: {len(all_rows)}")
    print(f"Total ambiguous rows flagged: {sum(1 for r in all_rows if r['is_ambiguous'])}")

    # Acceptance check from the key document: sample 436 -> 1000000/1000001
    # duplicate is specific to that document's own (unverified) dataset and
    # may not exist in this real data - report whatever duplicates this
    # reconciliation actually finds instead of checking for that literal case.
    dupes = {}
    for r in all_rows:
        if r["is_ambiguous"]:
            dupes.setdefault(r["original_sample"], []).append(r["current_sample"])
    if dupes:
        print("Ambiguous original_sample -> current_sample groups found:")
        for orig, currents in sorted(dupes.items())[:15]:
            print(f"  {orig} -> {sorted(set(currents))}")
        if len(dupes) > 15:
            print(f"  ... and {len(dupes) - 15} more")

    if args.dry_run:
        print("\n--dry-run: not writing to BigQuery.")
        return

    df = pd.DataFrame(all_rows)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("test_type", "STRING"),
            bigquery.SchemaField("original_sample", "INT64"),
            bigquery.SchemaField("current_sample", "INT64"),
            bigquery.SchemaField("timestamp_start", "TIMESTAMP"),
            bigquery.SchemaField("mapping_method", "STRING"),
            bigquery.SchemaField("is_ambiguous", "BOOL"),
            bigquery.SchemaField("created_at", "TIMESTAMP"),
        ],
    )
    job = bq_client.load_table_from_dataframe(df, MAP_TABLE, job_config=job_config)
    job.result()
    print(f"\nWrote {len(df)} rows to {MAP_TABLE}")


if __name__ == "__main__":
    main()
