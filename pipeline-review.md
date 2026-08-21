# Lab data pipeline: design review

Notpla F&P, machine data ingestion
Assessment as at 5 August 2026

Scope: the four deployed Cloud Functions in `notpla-machine-data`, the Windows
uploader scripts, the BigQuery tables they write to, and the Looker layer above
them. Findings are drawn from the deployed source, the live table schemas, GCS
folder state, IAM policy, BigQuery job history, and a replay of the deployed
parsers against 426 real files.

---

## 1. What is well designed

These are not consolation prizes. Several of them are decisions I would keep
unchanged in v2.

### 1.1 The folder state machine

`to-be-processed` → `processed` / `failed` is the single best decision in the
system. The location of a file *is* its processing state, visible in a console
with no tooling, no database, and no logs. It survives function redeployment.
It let this entire audit happen.

The industry name for this is the claim-check pattern, and the fact that you
arrived at it independently is worth noting.

### 1.2 A human gate before ingestion

Files land in a manual-check folder before they reach the watched folder.
Most lab pipelines automate straight from the instrument and end up ingesting
aborted runs, calibration tests and mis-labelled specimens. Yours cannot.

### 1.3 Least privilege on the uploader

`mecmesin-uploader` holds `storage.objectCreator` and `storage.objectViewer`
and nothing else. This is exactly right, and notably better than the runtime
side (see 2.11). A key sitting on a Windows box in a lab is the most exposed
credential in the system, and it is correctly the least powerful one.

### 1.4 Configuration by environment variable

`PROJECT_ID`, `BQ_DATASET`, `BQ_TABLE`, `WATCH_PREFIX`, `PROCESSED_PREFIX`,
`FAILED_PREFIX` are injected, not hardcoded. This is why you have four
pipelines rather than one unmaintainable monolith, and why adding a fifth is a
deployment rather than a rewrite. It is the main reason the expansion you are
planning is feasible at all.

### 1.5 Explicit timestamp formats

Every parser tries an ordered list of exact `strptime` formats and refuses to
guess. You hit the day/month inversion bug once, understood it, wrote it down,
and applied the correct fix everywhere. This is the single most common silent
data-corruption bug in CSV ingestion and it is not present in your system.

### 1.6 Duplication is tolerated rather than fought

Append-only loading, plus overlapping exports, plus
`RETRY_POLICY_DO_NOT_RETRY`, means a re-uploaded file is harmless. Verified:
328 tensile samples appear more than once and **zero** of them disagree on any
measured value. Your reasoning here was sound.

This tolerance also saved you. Of 1,638 specimens stranded in orphan files,
618 were recovered automatically because a later overlapping export carried
them in. Without the overlap the loss would have been 60% larger.

### 1.7 The direction of travel is right

The `leistritz-csv-ingest-raw` function, the most recent one, has a dedicated
service account, a 540-second timeout, 1 GiB of memory, long-format storage,
and an `ingestion_file_log` table. Every one of those is an improvement on the
film pipelines. You are already converging on the right answers; the problem is
that the earlier pipelines never got the benefit.

---

## 2. Where it falls short

Ranked by consequence, not by effort to fix.

### 2.1 No record of what was ingested

**There is no manifest.** Nothing anywhere answers "did file X land, and how
many rows did it contribute?". The processed folder claims to answer it and
does not.

Evidence: 179 files sit in `tensiletester-films-tensile-processed` with zero
rows in `films_tensile_results`. All 179 parse cleanly. Their date distribution
is February 107, March 14, April 58, and nothing afterwards, which matches the
April table rebuild that also left behind `films_tensile_results_backup` (3,949
rows) and `films_tensile_results_deduped` (2,090 rows).

The rows were destroyed by a manual operation on a live table. The folder still
said "processed". Nothing reconciled the two, so the loss was invisible for
four months.

This is the root cause of the largest data loss in the system, and every other
diagnostic difficulty in this review traces back to it.

### 2.2 No alerting

Confirmed: zero alert policies, zero log-based metrics on the project.

Evidence: 217 tensile files and 30 friction files accumulated in failed folders
across five months. 206 of the 217 parse cleanly against the parser deployed on
14 May, meaning they were fixed by that deployment and simply never replayed.

The Looker freshness timer is a genuinely clever stopgap, but it only detects
total stoppage. It cannot detect partial loss, which is the failure mode that
actually occurred.

### 2.3 Failure semantics differ between pipelines

Three pipelines, three behaviours:

| Pipeline | On failure |
|---|---|
| tensile | moves to failed, re-raises |
| friction | moves to failed, re-raises |
| extrusion | prints traceback, **returns normally** |

The deployed extrusion function has both `move_blob(..., FAILED_PREFIX)` and
`raise` removed relative to the repo version. A failed extrusion file is never
moved, never surfaced, and stays in `to-be-processed`, where the uploader's
`blob.exists()` check then treats every future copy as a duplicate. It fails
completely silently and its failed folder will always be empty.

The extrusion table holds 338 rows from 4 source files. That deserves checking
against how many extrusion runs you have actually logged.

### 2.4 The repository is not the source of truth

Deployed code has diverged from the repo in both directions:

- tensile gained the `%Y-%m-%d %H:%M` timestamp format on 14 May, never pushed
- extrusion was substantially rewritten on 8 May (`TABLE_COLUMNS`, header
  normalisation for embedded newlines, `parse_date` fallbacks), never pushed
- `backfill/backfill.py` still uses `pd.to_datetime(..., errors="coerce")`
  with no explicit format, which is the exact inference bug your own briefing
  document calls the most important lesson learned

Deploying from a console means the artefact under review is fiction. This is
why the first three turns of this review were spent establishing what the
system actually does.

### 2.5 Parsing is not separable from I/O

The friction function has its entire parsing logic — `normalize`, `parse_ts`,
footer stripping, column selection — defined *inside* the event handler, along
with the GCS download and the BigQuery load.

The practical consequence was demonstrated during this review: the tensile
parser could be imported and replayed against 217 real files in seconds,
because `extract_relevant_dataframe(bytes, source_file)` is a pure function.
The friction logic had to be hand-transcribed to test it at all.

Code that cannot be executed without deploying it is code that will only ever
be debugged in production.

### 2.6 The schema is defined in three places

For extrusion alone: the BigQuery table schema, the `TABLE_COLUMNS` list, and
`HEADER_MAP`. Nothing keeps them in agreement.

Resulting fossils, all confirmed present in live tables and written by no
deployed code:

- `films_tensile_results.row_num`
- `raw_films_extrusion.sd_percent_variation`
- `raw_films_extrusion.percent_variation_end`

`HEADER_MAP` also maps both `"% Variation"` and `"Variation"` to `variation`,
and both `"% Variation End"` and `"Variation End"` to `variation_end`. If an
export ever contains both spellings, the result is duplicate column names.

### 2.7 No type discipline in friction

`films_friction_raw` stores every measurement as `STRING`, because the friction
parser never calls `pd.to_numeric`. Static CoF, dynamic CoF, all forces.

Consequences: blanks are stored as `""` rather than `NULL`; sorting and
comparison are lexicographic; every Looker aggregation needs an explicit cast;
and `SAFE_CAST` failures will be silent.

The table is also named `_raw` while holding summary data, and sits in
`machine_data` while its curve counterpart was created in `films_tensile_london`.

### 2.8 The contract with the instrument is implicit and unvalidated

This is the finding with the largest implication for your expansion.

28 of the 30 failed friction files have **no `Timestamp - Start` column at
all**. Not a renamed column, not a reformatted value — the field is absent
from the export. Someone ran those tests with a Mecmesin template that did not
include the timestamp prompt.

Your parser's contract is defined by a test template on the instrument, which
any operator can reconfigure, and nothing validates that the template still
matches what the pipeline expects. Every instrument you add multiplies this
surface.

Related: 11 of the 217 tensile failures throw
`AttributeError: 'str' object has no attribute 'astype'`. This is lesson G in
your own briefing document, still live. `df.get(col, "")` returns a bare string
when a column is absent, and `.astype(str)` then fails. These are the
`FILMLONGGAUGE` and `CYCLICALLOADING` variants, which are legitimate tests from
a different protocol that the parser has no way to recognise or reject cleanly.

### 2.9 There is no key model

`sample` is a per-test-type counter. Tensile sample 250 and friction sample 250
are unrelated specimens. Nothing in any table encodes which instrument, which
test type, or which run a row belongs to, beyond the table it happens to live
in.

This directly blocks the thing you said you want most: selecting a Pellet ID
and seeing every test performed on that roll. It also guarantees collisions the
moment a second instrument starts its own counter at 1.

The candidate join keys are worse than they look. Friction calls it
`Extrusion code`, tensile calls it `Extrusion ID`, and both are free-typed
operator prompts, so whitespace, case and typos will silently drop rows from
any join.

### 2.10 Naming has no convention

Three film tests live in three datasets: `films_tensile_london`,
`machine_data`, `machine_collin_e25e`. Elsewhere in the project:
`film_tensile_data`, `tensiletester_1`, `Rigid_Tensile`, `Rigid_Tensile_euw2`,
and `machine_leistrtiz_1` — an empty dataset created by a typo, sitting next to
the correctly spelled one.

At four pipelines this is untidy. At twelve it is a barrier to entry for
anyone but you.

### 2.11 Over-privileged runtime

All three film functions run as `462425991200-compute@developer.gserviceaccount.com`,
which holds `roles/editor` on the project. A parsing bug in a lab CSV pipeline
has project-wide write access.

`leistritz-ingest-sa` is scoped correctly to `bigquery.dataEditor`,
`bigquery.jobUser` and `eventarc.eventReceiver`, which shows you already know
the right pattern.

Also: `mecmesin-uploader` holds a user-managed key created 29 January 2026,
never rotated, living on a lab PC. And the `notpla-machine-data@appspot` account
holds a user-managed key from April 2025 and `roles/editor`.

### 2.12 Destructive operations on live tables, with no safety net

Bucket versioning is **Suspended**. BigQuery tables have no partitioning, no
clustering, and no snapshot policy beyond whatever manual `_backup` table
someone thought to make.

The April rebuild destroyed 1,020 specimens' worth of rows. It was recoverable
only because the source CSVs happened to still exist in the processed folder.

### 2.13 Orphaned infrastructure

Two Eventarc triggers, `trigger-tensile-summary-upload` and `trigger-38nk5n1e`,
fire on every object finalised in the bucket and deliver to Cloud Run services
`tensile-summary-parser` and `tensile-results-parser`, neither of which exists.
Every upload has been generating failed deliveries for months.

`films_friction_curve_points` received 93 load jobs in April and the table has
since been dropped. The `films-friction-raw-processor` code exists in the repo
but is not deployed, while 197 files queue in its watch folder and 739 sit in
its failed folder.

1,167 tensile raw sample files have accumulated since February in a folder no
function watches.

### 2.14 Deduplication by dashboard control

Looker dropdowns hide duplicate rows; they do not prevent them being
aggregated. Any tile computing a mean, count or distribution across samples is
weighted by how many times each file happened to be re-uploaded.

For tensile this is currently harmless (zero conflicting values). For friction
it is not: 298 samples have copies and **18 of them hold genuinely conflicting
measurements** across two versions, with no rule anywhere deciding which wins.

### 2.15 No tests

No test directory, no CI, no fixtures, no compile check beyond running
`python -c "import main"` by hand. Every regression is discovered by a lab
scientist noticing a gap in a chart, or not noticing.

---

## 3. The headline number

| | Specimens |
|---|---:|
| In `films_tensile_results` today | 2,906 |
| Recoverable from orphaned files | 1,638 |
| — of those, already present via overlap | 618 |
| — of those, **genuinely missing** | **1,020** |

The tensile dashboard is missing approximately **26%** of every specimen ever
tested on that instrument.

---

## 4. The single sentence version

The data path is well conceived and the parsing is careful; what is missing is
everything that would tell you when the data path is not working. The system
has no memory of what it has done, and no voice with which to complain.
