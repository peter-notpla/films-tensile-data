# Lab data pipeline: forward roadmap

Notpla F&P, machine data
As at 20 August 2026

Everything below came out of the August review and recovery. Ordered by
dependency, not by size. Items inside a phase can mostly run in parallel;
phases mostly cannot.

---

## Done

- Audited the deployed system against the repo and established they had diverged
- Recovered 52 genuinely missing tensile specimens
- Established that the apparent 1,020 loss was a keying artefact of the Excel backfill
- Corrected 26 malformed pellet and extrusion IDs, trimmed 88 whitespace values
- Quarantined the 98-row packaging shelf-life study to its own table
- Built `id_corrections_log` as an audit trail
- Emptied both failed-processing folders into dated archives
- Dropped 3 redundant snapshots, the stale deduped table, and 2 orphaned Eventarc triggers
- Settled the specimen key model against the full history
- Resolved the 1264/1279 roll code ambiguity for samples 1383-1392 (lab tech
  confirmed 1264); deleted the ten erroneous rows and logged the change
- Built and deployed the manifest table (1.1) across all three pipelines,
  each individually verified against a controlled bad CSV (24 August 2026)
- Made the extrusion alert email user-friendly: severity, per-file detail,
  plain-language summary ahead of the technical section (24 August 2026)
- Built and deployed the row-errors table (1.2) across all three pipelines,
  fixing two bugs along the way where a single bad row killed the entire
  file (tensile's timestamp fallback, friction's raise-on-blank-sample and
  its first-column-only blank filter); verified live end to end
  (24 August 2026)
- Built and deployed the hourly first-sighting alert (1.3): routes by
  `user_initials` to Katie, Emily, or a default recipient, deduplicated via
  an alerts-sent log; verified live end to end including real email
  delivery to all three recipients (24 August 2026)
- Reworked all pipeline alert emails for a non-technical audience: plain-
  language summary ahead of technical detail, readable pipeline names, real
  failure timestamps, and resolved/closed notifications suppressed
  (24 August 2026)
- Added auto-deletion of stale failed-processing copies on successful
  reprocessing, and a repeat-failure escalation alert for files that fail
  twice under the same name; verified live end to end (24 August 2026)
- Fixed alert email subject lines (condition `displayName`, not policy
  `displayName`, drives the subject) and suppressed the extrusion policy's
  leftover "recovered" emails, the one policy the earlier UX pass missed
  (24 August 2026)
- Built and deployed the Friday morning digest (1.4): one email per test
  type covering the last 7 days (files processed/failed, specimens
  ingested) plus the true most-recent-test date pulled from each pipeline's
  own results table, to catch a machine gone silent even when nothing ever
  reaches the failed folder; verified end to end via a real scheduler
  invocation, email delivery confirmed by Peter (27 August 2026)
- Extracted the tensile parser into `shared/tensile_parser.py`
  (`extract_relevant_dataframe()` and its `_is_footer_row()` helper), moved
  out verbatim from `main.py` except for `should_process()`, which stayed
  behind as routing logic, not parsing. Diffed identical against the
  original. Replayed against all 526 real processed tensile files: 0 parse
  errors, 0 exceptions. The deployed
  `pipelines/films-tensile-csv-processor/main.py` is untouched; wiring it to
  import from `shared/` is the remaining Phase 4 migration step, still
  needing an answer for how `gcloud functions deploy --source=.` picks up a
  sibling directory outside the pipeline's own folder (27 August 2026)
- Extracted the friction parser into `shared/friction_parser.py`
  (`extract_friction_dataframe()`, plus its `normalize()` and `parse_ts()`
  helpers), moved out verbatim from `main.py`'s inline event handler.
  Diffed identical against the original apart from file-read plumbing
  (bytes in memory vs. a downloaded tempfile) and one now-dead local
  variable (`rows_total`, unused in the pure function since the caller
  derives it as `len(df) + rows_dropped`, matching the tensile parser's
  pattern). Unlike tensile, friction keeps every surviving column rather
  than a fixed named set, since `films_friction_raw` is still all-STRING
  (Phase 2.4). Replayed against all 274 real processed friction files: 0
  parse errors, 0 exceptions. The manifest-based row-count cross-check
  doesn't apply here either: only 2 checksums are recorded for friction
  (table exists since 24 August) and none matched these 274 pre-dating
  files, same limitation as the tensile replay. The deployed
  `pipelines/films-friction-csv-processor/main.py` is untouched (27 August
  2026)
- Extracted the extrusion parser into `shared/extrusion_parser.py`
  (`extract_extrusion_dataframe()`, plus `normalize_header()` and
  `parse_date()`, and the `TABLE_COLUMNS` / `HEADER_MAP` / `IDENTITY_COLUMNS`
  / `MIN_MAPPED_COLUMNS` / `FLOAT_COLUMNS` constants it depends on), moved
  out verbatim from `main.py`'s event handler, including the 20-21 August
  guards (required-column check, identity-column check, whitespace
  trimming, drop-rows-with-no-identity). Diffed identical against the
  original apart from the same dead `rows_total` local dropped in the
  tensile and friction extractions. Replayed against all 6 real processed
  extrusion files (this pipeline has by far the smallest processed
  backlog): 0 parse errors. Only 1 manifest checksum recorded and it
  doesn't match any of the 6, same pre-24-August limitation as tensile and
  friction. The deployed `pipelines/films-extrusion-csv-processor/main.py`
  is untouched. This completes Phase 2.1 for all three pipelines (27 August
  2026)
- Answered the Phase 4 blocker: confirmed via gcloud docs and community
  precedent that `gcloud functions deploy --source=<dir>` packages only
  that directory, full stop; there is no flag or mechanism to reach a
  sibling directory like `shared/`, so it has to be physically staged
  inside the pipeline directory before every deploy. Built
  `scripts/deploy.sh`, which stages `shared/*.py` into
  `pipelines/<name>/shared/`, asserts the staged file count matches the
  source before deploying (never trust a copy loop silently), runs
  `gcloud functions deploy --source=<pipeline-dir>`, and always cleans the
  staged copy up on exit via a trap. `pipelines/*/shared/` added to
  `.gitignore` so the staged copy is never a second source of truth.
  Proven against a real deploy: ran it against extrusion with `main.py`
  still untouched (so behaviour is provably unchanged), new revision
  `films-extrusion-csv-processor-00012-yeh` went `ACTIVE`, staged copy
  confirmed cleaned up afterward, nothing stray left for git to pick up.
  The remaining Phase 4 step per pipeline is switching `main.py` from its
  inline parser copy to `from shared.<name>_parser import ...`, not yet
  done for any of the three (27 August 2026)
- Completed the Phase 4 migration for extrusion, the first of the three.
  `pipelines/films-extrusion-csv-processor/main.py` now imports
  `extract_extrusion_dataframe` from `shared/extrusion_parser.py` instead
  of holding its own copy; every other line (event handling, GCS moves,
  manifest/row-errors writes, the restored 20 August failure path) is
  untouched, confirmed by diff. `scripts/deploy.sh` updated to exclude
  `verify_*.py` from what gets staged, since those are dev-only. Deployed
  live: new revision `films-extrusion-csv-processor-00013-liz` went
  `ACTIVE`, and a one-time `PHASE_4_CUTOVER` startup log line (added
  because the ported logic is behaviourally identical and prints nothing
  else new to verify against) confirmed in Cloud Logging that the
  `shared.extrusion_parser` import actually resolved at cold start in the
  real deployed environment, not just locally. No errors in Cloud Logging
  for the service in the hour after deploy. Extrusion is now the first
  pipeline running its parser from `shared/` in production; friction and
  tensile remain on their inline copies (27 August 2026)
- Completed the Phase 4 migration for friction, the second of the three.
  `pipelines/films-friction-csv-processor/main.py` now imports
  `extract_friction_dataframe` from `shared/friction_parser.py`; the
  tempfile download, GCS moves, manifest/row-errors writes and error
  handling are untouched. Caught a real drift before deploying: friction's
  original code sets the data table's `source_file` column to the full
  `gs://` URI, unlike extrusion (basename) or tensile (object path), three
  different pre-existing conventions across the three pipelines. The
  extracted module was always correct (`source_file` is just whatever the
  caller passes); the first draft of the cutover call and of
  `verify_friction_parser.py` passed the bare filename instead, both fixed
  to pass the full URI before anything was deployed. Replayed against all
  274 real processed files again after the fix: still 0 parse errors.
  Deployed live: revision `films-friction-csv-processor-00009-cas` went
  `ACTIVE`, `PHASE_4_CUTOVER` confirmed in Cloud Logging, no errors in the
  hour after deploy. Two of three pipelines now run their parser from
  `shared/`; tensile remains on its inline copy (27 August 2026)
- Completed the Phase 4 migration for tensile, the last of the three.
  `pipelines/films-tensile-csv-processor/main.py` now imports
  `extract_relevant_dataframe` from `shared/tensile_parser.py` instead of
  defining it inline; `should_process()` stayed put as routing logic, same
  reasoning as the original Phase 2.1 extraction. The call site was already
  passing `source_file=name` (the full GCS object path, tensile's own
  pre-existing convention, distinct from extrusion's basename and
  friction's full `gs://` URI) and didn't need to change, so there was no
  drift risk here the way there was for friction. Added the
  `PHASE_4_CUTOVER` marker via `logger.info()` rather than `print()`, to
  match this pipeline's existing structured-logging style rather than
  extrusion/friction's plain prints. Replayed against all 526 real
  processed tensile files again before deploying: still 0 parse errors.
  Deployed live: revision `films-tensile-csv-processor-00019-pil` went
  `ACTIVE`, `PHASE_4_CUTOVER` confirmed in Cloud Logging, no errors in the
  hour after deploy.

  **Phase 4 is now complete for all three pipelines.** Every deployed
  `main.py` imports its parser from `shared/` rather than holding its own
  copy; `scripts/deploy.sh` stages `shared/` into the pipeline directory
  before every future deploy, since `gcloud functions deploy` never
  includes anything outside `--source`. Remaining roadmap work is Phase 2
  (key model, schema-from-one-definition, typed columns, least-privilege
  service accounts, metadata revision handling), Phase 3 (validation), and
  Phase 5/6 (friction curve long format, analysis layer) - none of which
  are blocked on anything from this session (27 August 2026)
- Started closing out Phase 2 and Phase 3 as one sequenced body of work
  (checkpoints A-I; see the plan this was built from for the full design).
  **Checkpoint A, Phase 3.2 (whitespace trimming), done for tensile and
  friction.** Correcting the roadmap's own standing-item note in the
  process: it said extrusion was missing whitespace trimming, but the code
  says the opposite - extrusion already got it 20 August, tensile and
  friction did not (friction worst of all: nothing but the timestamp was
  trimmed). `shared/tensile_parser.py` now strips `pellet_id`,
  `extrusion_id`, `test_direction`, `sample_number`, `notes`,
  `user_initials`. `shared/friction_parser.py` now strips every surviving
  column's values (not just column names, which `normalize()` already
  handled), since its whole design is "keep whatever survives" rather than
  a fixed list. Verified against all 526 real tensile and 274 real friction
  files: 0 parse errors, and an explicit spot-check (parsing 5 real files
  from each and asserting no value differs from its own `.str.strip()`)
  confirms the trim is actually running, not just present in the diff. Also
  folded in Phase 3.5 (template naming convention): added a documentation-
  only note to `CLAUDE.md` since there is no code to write for a lab
  workflow convention. Deployed live: `films-tensile-csv-processor-00020-pok`
  and `films-friction-csv-processor-00010-peg` both `ACTIVE`, no errors in
  Cloud Logging afterward. No new log marker was needed for this one since
  the behaviour verification already happened via the real-file replay and
  the explicit strip assertion before deploying (27 August 2026)
- **Checkpoint B, Phase 2.3 (schema drift-check tooling), done for tensile
  and extrusion.** Not a live migration tool - a repeatable check, since a
  migration tool implies a target to migrate to and there isn't one here,
  just three places that can silently disagree. Added `TABLE_COLUMNS` to
  `shared/tensile_parser.py` (tensile didn't have one; its column list only
  ever existed inline in `main.py`'s `load_to_bigquery()`), and switched
  `main.py` to import it instead of holding its own copy - one fewer place
  for tensile's schema to drift, mirroring what `shared/extrusion_parser.py`
  already had. Built `shared/verify_schema_drift.py`, which queries each
  live BigQuery table's actual schema and diffs it against the parser's
  `TABLE_COLUMNS`, run against both tensile and extrusion: **no unexpected
  drift in either**, tensile's known `row_num` fossil correctly excluded by
  design and reported as such rather than flagged. Friction deliberately
  not covered - its parser keeps whatever columns survive rather than a
  fixed list, which resists a fixed-list diff; Phase 2.4 (typed columns) is
  the real fix for friction's schema looseness, not this tool. Verified
  the tensile refactor against all 526 real files (0 parse errors,
  unchanged) and confirmed the shared import still resolves staged.
  Deployed live: `films-tensile-csv-processor-00021-voz` `ACTIVE`, no
  errors in Cloud Logging afterward (27 August 2026)
- **Checkpoint C, Phase 2.2 (key model), done for tensile and friction.**
  Added `template_name`, `timestamp_minute`, `specimen_key` to both
  `films_tensile_results` and `films_friction_raw` (`ALTER TABLE`,
  nullable, additive; both tables snapshotted first as
  `*_presnap_20260827`). Backfilled `timestamp_minute`/`specimen_key` for
  history via one `UPDATE ... WHERE TRUE` per table (3,510 tensile rows,
  3,790 friction rows); `template_name` left `NULL` for history since row
  1's text was never stored for already-processed files, populated going
  forward only. The one real risk in this checkpoint: the SQL backfill and
  the Python parsers must produce byte-identical `specimen_key` strings for
  the same real specimen, or the same roll's tests would carry two
  different keys depending on which path wrote the row. Both sides use an
  explicit `%Y-%m-%dT%H:%M` timestamp format rather than a default
  string cast specifically to guarantee this - verified by downloading a
  real processed file for both pipelines, re-parsing it with the updated
  parser, and confirming the Python-computed `specimen_key` matches the
  already-backfilled live value character for character (it did, both
  pipelines). `shared/tensile_parser.py` and `shared/friction_parser.py`
  now capture row 1 (the VectorPro template name, previously discarded
  outright) and compute the two new derived columns; tensile's
  `TABLE_COLUMNS` picked up the three new names automatically as a
  consequence of the Checkpoint B refactor, no `main.py` changes needed for
  either pipeline. Verified against all 526 real tensile and 274 real
  friction files: 0 parse errors. Confirmed both `main.py` modules import
  cleanly end-to-end (not just their shared submodule) when staged the way
  `scripts/deploy.sh` stages them. Deployed live:
  `films-tensile-csv-processor-00022-lad` and
  `films-friction-csv-processor-00011-xem`, both `ACTIVE`, no errors in
  Cloud Logging afterward (27 August 2026)
- **Checkpoint D, Phase 2.4 (typed columns for friction), done.** Added a
  `_num` FLOAT64 sibling for each of 9 numeric-measurement columns on
  `films_friction_raw`
  (`static_friction_force_magnitude_1_n`, `backup_peak_n`,
  `dynamic_friction_force_n`, `static_coefficient_of_friction`,
  `backup_static_cof`, `dynamic_coefficientof_friction`,
  `sample_number_prompt_for_value_before_test`,
  `sample_repeat_number_prompt_for_value_before_test`,
  `pctrh_prompt_for_value_before_test`) - additive only, original STRING
  columns untouched, so Looker Studio keeps working unchanged; pointing its
  charts at the new typed columns is a manual follow-up whenever convenient.
  Snapshotted first (`films_friction_raw_presnap2_20260827`). Backfilled via
  one `UPDATE ... SAFE_CAST ... WHERE TRUE` (3,790 rows): confirmed the
  resulting `NULL`s are all genuinely blank source values, zero
  non-blank-but-unparseable cases. `shared/friction_parser.py` now computes
  the same 9 `_num` columns going forward via `pd.to_numeric`, only when
  the source column is present in that file (some older files lack
  `backup_static_cof` per `CLAUDE.md`). Cross-checked a real row: Python's
  `pd.to_numeric` and BigQuery's `SAFE_CAST` agree exactly on the same
  source string. Verified against all 274 real friction files: 0 parse
  errors. `main.py` needed no changes - it loads whatever columns the
  parser produces - confirmed by a full `import main` against the staged
  `shared/`. Deployed live: `films-friction-csv-processor-00012-zit`
  `ACTIVE`, no errors in Cloud Logging afterward (27 August 2026)
- **Checkpoint E, Phase 3.1 (ID format validation), done for all three
  pipelines.** Added `validation_status` (`valid` /
  `invalid_pellet_id` / `invalid_extrusion_id` / `invalid_both`) to
  `films_tensile_results`, `films_friction_raw`, and
  `raw_films_extrusion` (`ALTER TABLE`, additive; all three snapshotted
  first). Built `shared/id_validation.py` as the single definition of both
  regexes, imported by all three parsers rather than tripling the logic.
  Backfilled all three tables via one `UPDATE ... CASE WHEN
  REGEXP_CONTAINS(...` per table. **Two real bugs caught and fixed by
  verification before this was trusted:**
  1. The Python validator's first draft used `pellet_id or ""`, and a
     pandas NaN float (which extrusion's columns can produce, since that
     parser doesn't force `dtype=str` the way tensile/friction do) is
     truthy in Python, so it reached `re.match()` as a float and crashed.
     Caught immediately by the standard replay check (4 of 6 real
     extrusion files started failing). Fixed with a proper `pd.isna()`
     guard.
  2. The SQL backfill's `CASE` used plain `NOT REGEXP_CONTAINS(...)`
     conditions, and SQL's three-valued logic means `NOT NULL` evaluates
     to `NULL`, not `TRUE` - so whenever one ID was invalid and the other
     was `NULL` (138 rows just on extrusion), the `invalid_both` branch
     silently failed to match and the row fell through to
     `invalid_pellet_id`/`invalid_extrusion_id` instead, under-reporting.
     Caught by cross-checking every live row's stored ID values directly
     against Python's `validation_status()` (not by re-parsing source
     files, which turned out to be the wrong comparison - see below).
     Fixed by wrapping both sides in `COALESCE(..., "")` before
     `REGEXP_CONTAINS`, and all three tables re-backfilled.
  A third apparent mismatch turned out not to be a bug: re-parsing an
  original GCS file and comparing against the live row can legitimately
  disagree when the stored value predates a later parser fix (e.g. a
  pre-20-August extrusion row still carries a trailing space the live
  whitespace-trim would now strip). The correct verification is Python's
  `validation_status()` against the value **actually stored live**, not
  a fresh re-parse of the source file - confirmed exactly this way across
  the full population of all three tables (3,510 + 3,790 + 338 = 7,638
  rows, not a sample): zero mismatches. Sanity-checked the resulting
  distribution too: tensile is 100% valid (consistent with its earlier
  26-ID cleanup and shelf-life quarantine), extrusion's high invalid rate
  (147/338) is almost entirely `NULL extrusion_id` on runs where it was
  never recorded - a genuine, meaningful gap, not a validation bug.
  Verified against all real files again after both fixes: 0 parse errors
  across tensile (526), friction (274), extrusion (6). Confirmed `import
  main` succeeds for tensile and friction against staged `shared/`;
  extrusion's local import still can't complete because
  `functions_framework` isn't installed in this dev environment (same
  known, unrelated gap noted during its Phase 4 cutover), so only its
  `shared.extrusion_parser` import was checked directly. No `main.py`
  changes needed for any of the three. Deployed live:
  `films-tensile-csv-processor-00023-tum`,
  `films-friction-csv-processor-00013-cah`,
  `films-extrusion-csv-processor-00014-cav`, all three `ACTIVE`, no errors
  in Cloud Logging afterward (27 August 2026)
- **Checkpoint F, Phase 3.3 (Excel detection), done for tensile and
  friction; deliberately not applied to extrusion.** Added
  `excel_processed` BOOL to the shared `films_pipeline_manifest` table
  (one schema change covers all three pipelines, since it's a file-level
  property, not per-specimen; snapshotted first). Built
  `shared/excel_detection.py` as the single definition of both signals (row
  1 ending in comma padding; every present timestamp having zero seconds).
  Computed in `main.py`, not the parsers: a padding-only pre-check runs
  right after download so it's available even when parsing fails outright
  (a padded row 1 can itself be the cause of the failure), then refined
  with the timestamp signal after a successful parse. `write_manifest()`
  gained an `excel_processed` parameter in all three pipelines. **Caught
  before deploying**: extrusion's row 1 is a real section-header row
  (`Film Thickness Profile Average,,,Pellets QC,,,...`) that legitimately
  ends in a comma by normal structure, not as an Excel artifact - all 6
  real processed extrusion files "flagged" on the padding signal alone,
  a 100% rate that was the tell it was a false positive rather than a
  finding. Extrusion also has no seconds-resolution timestamp field to
  fall back on, and its source machine was never part of the "opened in
  Excel during the manual check" workflow `CLAUDE.md` describes for the
  Mecmesin tensile tester's VectorPro exports in the first place, so
  extrusion's `excel_processed` is left `NULL` always, not computed.
  Validated the detection logic in isolation (synthetic all-zero /
  non-zero / mixed-seconds timestamps, padded / clean row 1: all five
  cases behaved exactly as expected) and against real files (tensile: 4
  of 20 sampled flagged, friction: 0 of 20 sampled flagged - both
  plausible, not universal like the false positive would have been).
  Confirmed `import main` succeeds for tensile and friction against
  staged `shared/`; extrusion checked via compile only, same
  `functions_framework`-not-installed gap as every prior checkpoint.
  Deployed live: `films-tensile-csv-processor-00024-pol`,
  `films-friction-csv-processor-00014-yim`,
  `films-extrusion-csv-processor-00015-nex`, all three `ACTIVE`, no errors
  in Cloud Logging afterward (27 August 2026)
- **Checkpoint G, Phase 3.4 (cross-reference IDs against the extrusion
  table), done.** Two BigQuery views, no pipeline code change, no redeploy:
  `films_pipeline_ops.films_tensile_id_cross_reference` and
  `films_pipeline_ops.films_friction_id_cross_reference`, each left-joining
  its results table against a `pellet_id`+`extrusion_id`-deduplicated
  lookup over `raw_films_extrusion` (`MIN(date)` per pair, since a given
  roll can have several extrusion rows), flagging `roll_exists` and
  `extruded_before_test`. Verified the join doesn't fan out (both views
  return exactly as many rows as their source table: 3,510 and 3,790).
  The headline finding: most well-formed IDs don't match anything in
  `raw_films_extrusion` at all - 2,031/3,510 (58%) for tensile, 3,403/3,790
  (90%) for friction. Confirmed this is a genuine coverage gap, not a join
  bug, by hand-checking one specimen's exact `pellet_id` and a substring of
  its `extrusion_id` against the live extrusion table directly: zero
  matches either way. Makes sense given the numbers - `raw_films_extrusion`
  holds only 338 rows total against thousands of downstream specimens, and
  is the same underlying gap the parked pass-filter extrusion lookup work
  already surfaced (18 of 36 rolls unresolved there). Views are trivially
  reversible (`DROP VIEW`) (27 August 2026)
- **Checkpoint H, Phase 2.6 (least-privilege service accounts), IAM setup
  done, per-pipeline cutover starting.** Granted `roles/storage.objectAdmin`
  to the existing, previously-unused `sa-tensile-ingest` (it already held
  `bigquery.dataEditor`, `bigquery.jobUser`, `eventarc.eventReceiver`, plus
  a leftover `pubsub.subscriber` left untouched, not part of this cleanup).
  Created `sa-friction-ingest` and `sa-extrusion-ingest` fresh, both with
  the same four-role template as `leistritz-ingest-sa`
  (`bigquery.dataEditor`, `bigquery.jobUser`, `eventarc.eventReceiver`,
  `storage.objectAdmin`) rather than the compute default SA's
  project-wide `roles/editor` plus five other broad roles all three film
  functions currently run as. `scripts/deploy.sh` gained an optional third
  argument, the service account to deploy with (defaults to leaving the
  function's current SA untouched, matching `gcloud functions deploy`'s
  own behaviour when `--service-account` is omitted) (27 August 2026)

---

## Phase 0: stop active harm

Do these before anything else. Each is small and each is currently costing
something.

### 0.1 Fix the extrusion function's silent failure
The deployed version has `move_blob(..., FAILED_PREFIX)` and `raise` removed
relative to the repo. Failed files are never surfaced and stay in
`to-be-processed`, where the uploader's `blob.exists()` check treats every
future copy as a duplicate. Restore both lines. This is the only pipeline
actively losing data with no trace.

### 0.2 Push the deployed code back to GitHub
Tensile gained a timestamp format on 14 May; extrusion was rewritten on 8 May.
Neither was pushed. Until this is done the repo is fiction and no review of it
means anything. Include the 0.1 fix in the same commit so the repo becomes
truthful and correct at the same moment.

### 0.3 Fix the backfill script's date parsing
`backfill/backfill.py` still uses `pd.to_datetime(..., errors="coerce")` with no
explicit format. That is the exact inference bug your own briefing document
calls the most important lesson learned.

### 0.4 Resolve the 1264 versus 1279 ambiguity
Ten specimens from 4 August carry two different roll codes. Circumstantial
evidence favours 1264. Confirm with whoever ran it, then delete the wrong ten
rows and log the change.

---

## Phase 1: know what is happening

Nothing else in the roadmap is safe to build until the system can tell you
what it has done. This phase is the foundation.

### 1.1 Build the manifest table
One row per file seen, with: filename, checksum, test type, machine, outcome,
row count, error message, timestamps. This is the single highest-value
addition. It is what would have exposed the 179 processed-but-empty files in a
day rather than four months. Callum's `tensile_v21_file_manifest` is a working
model.

### 1.2 Build the row-errors table
Currently one bad row rejects an entire file. Instead, load the good rows,
write the bad ones to a row-errors table with the reason. This is what turns
a 217-file backlog into a queryable list of specific problems.

### 1.3 Hourly first-sighting alert
A scheduled job checks the failed folders. On first sighting of a file it
emails the responsible person, then records it so the same file never alerts
again. Route by `user_initials` with a lookup table; fall back to the pellet
ID's owner, then to a default recipient. Never fail silently because the
routing failed: the 28 friction files had no initials column at all.

### 1.4 Friday morning digest
09:00 every Friday, one email per test type: files processed, files failed,
specimens ingested, most recent test date. This covers the silence case that
a failure-triggered alert cannot, which matters because extrusion currently
fails without producing a failed file.

---

## Phase 2: v2 architecture

Built for a new instrument first, so nothing existing is at risk. The
existing three migrate onto it later, one at a time.

### 2.1 Shared parsing library
Extract parsing into a pure function: bytes in, dataframe out, no cloud
clients. The tensile parser is already close to this and could be replayed
against 217 real files in seconds. The friction parser has its logic inline
in the event handler and had to be hand-transcribed to test at all. Code you
cannot run without deploying it is code you will only ever debug in production.

### 2.2 Implement the key model

| Purpose | Fields |
|---|---|
| Specimen identity | `machine_id`, `test_type`, `timestamp_minute`, `sample` |
| Provenance | `template_name` (row 1), `source_file`, full `timestamp_start` |
| Analysis | `pellet_id`, `extrusion_id`, `test_direction` |

Verified against the full tensile history: zero conflations across both
backfills and every template reset. Machine and test type come from pipeline
config so they cannot be mistyped. Minute resolution is deliberate, because
Excel strips seconds during the manual check and that workflow is staying.

### 2.3 Schema from one definition
Currently the extrusion schema exists in three places that nothing keeps in
agreement, which is how `row_num`, `sd_percent_variation` and
`percent_variation_end` became fossils. Define once, generate the BigQuery
schema and the parser mapping from it.

### 2.4 Typed columns everywhere
`films_friction_raw` stores every measurement as STRING because the friction
parser never calls `pd.to_numeric`. Blanks land as `""` rather than NULL and
all sorting is lexicographic.

### 2.5 Metadata revision handling
Same key, same measurements, different metadata is a correction, not a
duplicate and not a new specimen. Update in place, log the prior value. The
1264/1279 case is exactly this shape. Re-upload of a corrected file is a
healthy part of the workflow and v2 should treat it as such.

### 2.6 Least-privilege service accounts
The three film functions run as the compute default account, which holds
`roles/editor` on the project. `leistritz-ingest-sa` shows the right pattern
already.

---

## Phase 3: validation

### 3.1 ID format validation, flag not reject
- `pellet_id`: `^[A-Z]{2} [A-Z]{2} [A-Z]{2} [A-Z]{2} [0-9]{6} [A-Z]{2} [A-Z]{2} [0-9]{4}$`
- `extrusion_id`: `^[A-Z]{2} [0-9]{6} [A-Z]{2} [0-9]{4}$`

Load every row, mark failures with a `validation_status` column, filter the
dashboard to clean rows by default. A hard reject would have discarded roughly
150 legitimate shelf-life rows to catch 25 typos.

### 3.2 Silent whitespace trimming
No information is carried by leading or trailing spaces and they are invisible
in every UI. Trim on ingestion, always.

### 3.3 Excel detection
A file whose row 1 ends in comma padding, or where every timestamp has zero
seconds, has been through Excel. Flag it, do not reject it. Useful for knowing
which files have lost precision.

### 3.4 Cross-reference IDs against the extrusion table
Format validation catches typos; it cannot catch substitution. Both 1264 and
1279 are well-formed. Flag a roll that does not exist, or one extruded after
the test date.

### 3.5 Template naming convention
When copying a VectorPro test, give the new template a distinct name rather
than reusing the old one. Row 1 then becomes meaningful provenance. Cannot
repair history, but costs nothing going forward.

---

## Phase 4: migration

One pipeline at a time. Run old and new in parallel against the same files,
compare row counts, then cut over. A sweeping rewrite of all four means no
rollback and no comparison baseline.

Suggested order: extrusion first (smallest, already needs surgery), then
friction (worst schema), then tensile (highest value, most risk).

---

## Phase 5: the friction curve problem

The thing you flagged at the outset and deferred.

A friction curve can be flat or violently oscillating, and both are currently
reduced to one mean value that cannot distinguish them. Long format is the
answer, one row per timepoint per test, which is the pattern already in use
elsewhere in the business and already in your own project:
`process_parameters_long_raw` holds 1.14 million rows across 8 files.

Assets already waiting:
- 739 friction raw files in failed processing
- 197 friction raw files queued for a processor that was never deployed
- 1,244 tensile raw sample files in a folder nothing watches

That last set means you have been collecting tensile curve data since February
without realising it, which gives you a corpus to prototype against before
touching friction.

---

## Phase 6: analysis layer

### 6.1 The `films_results_long` view
One row per test type, sample, metric name, metric value, plus shared identity
columns. This is what makes "select a Pellet ID and see every test on that
roll" work. A wide join would produce a cartesian product: 10 tensile against
15 friction specimens is 150 rows and every average would be wrong.

Prefer a BigQuery view over Looker blending. The join logic lives in one place,
all controls work normally, and it can be version-controlled.

### 6.2 Deduplication in the view, not the dashboard
`QUALIFY ROW_NUMBER() OVER (PARTITION BY specimen_key ORDER BY processed_at DESC) = 1`.
Dropdowns hide duplicate rows but do not stop them being aggregated. Friction
currently has 18 samples with genuinely conflicting values and no rule deciding
which wins.

---

## Standing items

- ~~Bucket versioning is Suspended. Any delete is permanent. Worth enabling.~~
  Checked 27 August 2026: `notpla-machine-data` is a hierarchical-namespace
  bucket, so versioning cannot be enabled on it at all (unsupported by GCS).
  It already has a 7-day soft-delete policy active since creation, so
  deletes are recoverable, not permanent. No action needed.
- **Key rotation.** `mecmesin-uploader` holds a user-managed key from January
  2026, never rotated, on a lab PC. The appspot account holds one from April
  2025 plus `roles/editor`.
- **Naming convention.** Three film tests live in three datasets. Elsewhere:
  `film_tensile_data`, `tensiletester_1`, `Rigid_Tensile`, `Rigid_Tensile_euw2`,
  and `machine_leistrtiz_1`, an empty dataset created by a typo.
- **Units mislabelling.** `average_thickness_mm` in the extrusion table holds
  microns (117.7, not 0.1177).
- **Extrusion table whitespace.** Still present; today's cleanup only covered
  tensile.
- **Friction CoF precedence.** Use `static_coefficient_of_friction` where
  present, fall back to `backup_static_cof`, record which was used.
- **Talk to Callum.** His `tensile_v21_*` tables already have the manifest and
  row-errors pattern. Worth aligning before building a second version of it.
- **Amend the review document.** It states 26% of specimens missing; the true
  figure was closer to 1.5%. The findings behind it stand.
- **Tests and CI.** No test directory, no fixtures, no compile check. Every
  regression is currently found by a scientist noticing a gap in a chart.
