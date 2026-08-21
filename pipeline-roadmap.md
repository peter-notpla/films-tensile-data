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

- **Bucket versioning is Suspended.** Any delete is permanent. Worth enabling.
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
