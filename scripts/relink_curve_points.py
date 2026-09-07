"""One-off (re-runnable) backfill re-link pass: applies the tiered
shared/curve_linking.py lookups (time -> direct sample -> mapped sample via
sample_number_map) to every already-loaded curve_points row that has no
link yet, without re-parsing or re-downloading any raw curve file.

Only touches rows where linked_specimen_key IS NULL - this is additive,
never overwrites an existing link (time-based links made at live-ingest
time are higher confidence than anything this script can produce for
historical files, since the GCS-creation-time signal they used is
unrecoverable for files already moved - see curve_linking.py's docstring).

Time-based re-linking is not attempted here for the same reason: a
historical file's current blob metadata reflects its move into the
processed folder, not its original upload, so find_specimen_link would
only produce wrong or coincidental matches on already-moved files. Only
tiers 2 and 3 (direct sample match, then mapped-sample match) run here.

Usage:
    python3 scripts/relink_curve_points.py --dry-run
    python3 scripts/relink_curve_points.py             # writes for real
    python3 scripts/relink_curve_points.py --instrument tensile
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.curve_linking import (
    find_specimen_link_by_sample,
    find_specimen_link_by_mapped_sample,
    DEFAULT_MAP_TABLE,
)

PROJECT_ID = "notpla-machine-data"

INSTRUMENTS = {
    "tensile": {
        "curve_table": f"{PROJECT_ID}.films_tensile_london.films_tensile_curve_points",
        "results_table": f"{PROJECT_ID}.films_tensile_london.films_tensile_results_all_revisions",
        "test_type": "tensile",
    },
    "friction": {
        "curve_table": f"{PROJECT_ID}.machine_data.films_friction_curve_points",
        "results_table": f"{PROJECT_ID}.machine_data.films_friction_raw_all_revisions",
        "test_type": "friction",
    },
}


def fetch_unlinked(bq_client, curve_table):
    query = f"""
        SELECT DISTINCT source_file, template_name, raw_sample_number
        FROM `{curve_table}`
        WHERE linked_specimen_key IS NULL
    """
    return list(bq_client.query(query).result())


def relink(bq_client, name, config):
    print(f"\n=== {name} ===")
    candidates = fetch_unlinked(bq_client, config["curve_table"])
    print(f"Unlinked files: {len(candidates)}")

    results = []
    tier_counts = {"sample_number": 0, "mapped_sample": 0}
    for row in candidates:
        specimen_key = find_specimen_link_by_sample(
            bq_client, config["results_table"], row["template_name"], row["raw_sample_number"]
        )
        method = "sample_number" if specimen_key else None
        if specimen_key is None:
            specimen_key = find_specimen_link_by_mapped_sample(
                bq_client, config["results_table"], DEFAULT_MAP_TABLE, config["test_type"],
                row["template_name"], row["raw_sample_number"],
            )
            method = "mapped_sample" if specimen_key else None
        if specimen_key is not None:
            tier_counts[method] += 1
            results.append({
                "source_file": row["source_file"],
                "linked_specimen_key": specimen_key,
                "link_method": method,
            })

    print(f"Newly linkable via direct sample match: {tier_counts['sample_number']}")
    print(f"Newly linkable via mapped sample match: {tier_counts['mapped_sample']}")
    print(f"Total newly linkable: {len(results)}")
    print(f"Still unlinked after this pass: {len(candidates) - len(results)}")
    return results


def apply_updates(bq_client, name, config, results):
    if not results:
        print(f"[{name}] nothing to write.")
        return

    def _bq_ref(table_id):
        # bq cp wants project:dataset.table, not the dot-separated
        # project.dataset.table form used everywhere else in this codebase.
        project, dataset, table = table_id.split(".")
        return f"{project}:{dataset}.{table}"

    snapshot_name = f"{config['curve_table']}_presnap_{datetime.now(timezone.utc).strftime('%Y%m%d')}_relink"
    print(f"[{name}] snapshotting to {snapshot_name}...")
    import subprocess
    subprocess.run(["bq", "cp", "-n", _bq_ref(config["curve_table"]), _bq_ref(snapshot_name)], check=True)

    staging_table = f"{config['curve_table']}_relink_staging"
    df = pd.DataFrame(results)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("source_file", "STRING"),
            bigquery.SchemaField("linked_specimen_key", "STRING"),
            bigquery.SchemaField("link_method", "STRING"),
        ],
    )
    bq_client.load_table_from_dataframe(df, staging_table, job_config=job_config).result()

    merge_query = f"""
        MERGE `{config['curve_table']}` T
        USING `{staging_table}` S
        ON T.source_file = S.source_file
        WHEN MATCHED AND T.linked_specimen_key IS NULL THEN
          UPDATE SET T.linked_specimen_key = S.linked_specimen_key,
                     T.link_method = S.link_method
    """
    job = bq_client.query(merge_query)
    job.result()
    print(f"[{name}] MERGE affected {job.num_dml_affected_rows} rows.")

    bq_client.delete_table(staging_table, not_found_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instrument", choices=list(INSTRUMENTS), default=None)
    args = parser.parse_args()

    bq_client = bigquery.Client(project=PROJECT_ID)
    names = [args.instrument] if args.instrument else list(INSTRUMENTS)

    all_results = {}
    for name in names:
        all_results[name] = relink(bq_client, name, INSTRUMENTS[name])

    if args.dry_run:
        print("\n--dry-run: not writing to BigQuery.")
        return

    for name in names:
        apply_updates(bq_client, name, INSTRUMENTS[name], all_results[name])


if __name__ == "__main__":
    main()
