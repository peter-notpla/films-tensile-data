# Notpla lab data pipeline: project briefing

Prepared 21 August 2026 to resume work in a new conversation.
Read this first. It contains the system state, the decisions made and why,
the working preferences, and the traps that cost time getting here.

---

## 1. What this project is

Peter is a process engineer at Notpla, a seaweed-derived biomaterials company
in London. He built a lab data pipeline that routes test results from lab
instruments into BigQuery and a Looker Studio dashboard. It was built with
Gemini and ChatGPT over several months and it works.

The goal is to expand it to more lab instruments. Before expanding, he asked
for a review of what exists: where it is well designed, where it falls short,
and how to build a better version.

**He explicitly did not want the working system changed** at the outset. That
position later softened to: do not touch it now, but migrate it onto a better
pattern once that pattern is proven on a new instrument. Design v2 so the
existing pipelines can migrate onto it later.

Two documents already exist and should be treated as current:
- `pipeline-review.md`: the assessment, 7 strengths and 15 shortcomings
- `pipeline-roadmap.md`: the forward plan, phases 0 to 6

One correction to `pipeline-review.md` has not yet been applied: it states the
tensile dashboard was missing 26% of specimens. The true figure was closer to
1.5%. See section 4.2. The findings behind the number all stand.

---

## 2. The system as it exists

### 2.1 Data flow

Mecmesin tensile tester (VectorPro software) exports CSVs
→ manual check folder on a Windows PC
→ Windows Task Scheduler uploads to GCS every ~10 minutes
→ Eventarc fires a Cloud Function per test type
→ BigQuery table per test type
→ Looker Studio, one page per test type

GCP project: `notpla-machine-data`, region `europe-west2`.
GitHub: `peter-notpla/films-tensile-data`.

### 2.2 Deployed Cloud Functions (all gen2)

| Function | Table | State |
|---|---|---|
| `films-tensile-csv-processor` | `films_tensile_london.films_tensile_results` | working |
| `films-friction-csv-processor` | `machine_data.films_friction_raw` | working, all-STRING schema |
| `films-extrusion-csv-processor` | `machine_collin_e25e.raw_films_extrusion` | fixed 21 Aug 2026 |
| `leistritz-csv-ingest-raw` | `machine_leistritz_1.process_parameters_long_raw` | Callum's, long format, 1.14M rows |

Also present, built by Callum: `tensile-router` and `tensile-v21-processor`,
writing to `Rigid_Tensile_euw2`. That dataset has `tensile_v21_file_manifest`
and `tensile_v21_row_errors`, which are the patterns v2 should adopt. Callum's
work covers injection-moulded dogbones; Peter's covers extruded film dogbones.
**A conversation with Callum is an open item.**

### 2.3 Key table row counts as at 21 August 2026

- `films_tensile_results`: 3,520 rows, 100% ID format compliance
- `films_friction_raw`: 3,790 rows, all measurement columns are STRING
- `raw_films_extrusion`: 338 rows from 4 source files
- `films_tensile_shelf_life_study`: 98 rows, quarantined packaging study
- `id_corrections_log`: audit trail, 98 quarantine + 88 trim + 26 correct

Snapshots retained as undo points:
- `films_tensile_results_snapshot_20260819` (before the 52-row recovery)
- `films_tensile_results_presnap_20260820_1420` (before the ID corrections)
- `films_tensile_results_backup` (April, 3,949 rows, pre-dates the rebuild)

### 2.4 GCS folder state

Both tensile and friction failed-processing folders are now **empty**, as is
the extrusion one. Anything appearing in them from now on is a live problem.

Archives created:
- `tensiletester-films-tensile-archive/reconciled-20260820` (218 files)
- `tensiletester-films-tensile-archive/discarded-20260820` (6 WIP files)
- `tensiletester-films-friction-archive/reconciled-20260820` (30 files)
- `machine-collin-e25e-archive/reconciled-20260821` (1 file)

Genuine backlogs, deliberately untouched:
- 738 friction raw files in failed processing, no processor deployed
- 197 friction raw files queued, waiting on a processor
- 1,244 tensile raw sample files in a folder nothing watches

**Bucket versioning is Suspended.** Any delete is permanent.

---

## 3. What was done in August 2026

1. Audited the deployed system and found it had diverged from the repo
2. Recovered 52 genuinely missing tensile specimens
3. Established the apparent 1,020-specimen loss was a keying artefact
4. Corrected 26 malformed pellet and extrusion IDs, trimmed 88 whitespace values
5. Quarantined a 98-row packaging shelf-life study to its own table
6. Built `id_corrections_log` as an audit trail
7. Emptied all failed-processing folders into dated archives
8. Dropped 3 redundant snapshots, a stale deduped table, 2 orphaned Eventarc triggers
9. Settled the specimen key model against the full history
10. Fixed the extrusion silent-failure bug and added required-column guards
11. Built and proved the first working email alert
12. Pushed the repo so it matches deployment for the first time

---

## 4. Decisions made, with their evidence

These were all hard-won. Do not relitigate them without new evidence.

### 4.1 The specimen key model

| Purpose | Fields |
|---|---|
| Identity | `machine_id`, `test_type`, `timestamp_minute`, `sample` |
| Provenance | `template_name` (row 1 of the CSV), `source_file`, full `timestamp_start` |
| Analysis | `pellet_id`, `extrusion_id`, `test_direction` |

Tested against the full tensile history. `timestamp_minute + sample` produced
2,990 distinct keys and **zero** conflations of genuinely different specimens.
Alternatives were rejected on evidence:
- `timestamp + stress + modulus`: 1 conflation, only 2,386 keys
- `+ pellet_id`: produced 3,000 keys, meaning it fragments identity, and it
  includes a mutable field that gets corrected by hand

Minute resolution is deliberate. See 4.4.

### 4.2 Sample numbers are not stable identifiers

VectorPro assigns sample numbers per test template. Minor edits to a template
overwrite it and the counter continues. Major edits require copying the
template, at which point the counter **restarts at 1** and the old data stays
inside the renamed original. Peter renames the old one to OLD.

Two manual Excel backfills then mangled this differently:
- **Tensile** was resequenced from 1,000,000 (sample 436 became 1000000)
- **Friction** was offset by exactly +1,000,000

Neither is recorded anywhere. This is why the initial reconciliation reported
1,020 missing specimens when the true figure was 52: it keyed on sample number,
which had been rewritten.

The template name in row 1 of every CSV is provenance, not identity, because a
copied template can carry the same name. `(V1)` refers to overwritten versions
only, not copies.

### 4.3 The shelf-life study

98 rows where an operator repurposed `pellet_id` and `extrusion_id` to hold
packaging format and time point (NAKED, DOYPACK, DAY 7, DAY 28). A one-off,
one person in a hurry, not expected to recur. Moved to
`films_tensile_shelf_life_study` with the full schema intact.

This is the reason ID validation must **flag rather than reject**. A hard gate
would have discarded roughly 150 legitimate rows to catch 25 typos.

### 4.4 Excel destroys seconds, and that is accepted

The manual check step involves opening files in Excel and saving. Excel parses
timestamps into typed cells, applies a display format that drops seconds, and
writes the displayed string on save. It also strips trailing zeros from
numbers (0.810 becomes 0.81) and pads row 1 with trailing commas.

25 files are affected, all from one person (KF), starting April 2026 when the
final check was introduced.

**Peter has decided to keep this workflow.** The team finds it works, and only
the test date matters for the dashboard. So the key uses minute resolution and
v2 must tolerate it. Detection signatures for flagging: row 1 ending in comma
padding, or every timestamp in a file having zero seconds.

Any key comparing measurements must compare them **numerically**, never as
strings, because of the trailing zero stripping.

### 4.5 Friction has two static CoF columns by design

`static_coefficient_of_friction` and `backup_static_cof`. VectorPro's peak
function sometimes fails to detect static CoF, so a second function was added
as a sanity backup. Introduced mid-development, which is why older files lack
it. All new files have both.

v2 precedence rule: use static where present, fall back to backup, record
which was used.

### 4.6 Friction recovery was correctly abandoned

30 files failed because 28 lacked a `Timestamp - Start` column entirely. They
turned out to be a bulk back-export performed on 9 April in sequential chunks
(samples 1-20, 21-40, and so on to 529) within 15 minutes. Their contents were
already in the table under the +1,000,000 offset. Nothing was loaded.

5 rows have no counterpart. 4 of those show static CoF around 5.0, roughly
twenty times every other reading, which looks like a units or calibration
fault. Archived, not loaded. **Worth investigating separately.**

### 4.7 Outstanding: samples 1383 to 1392

Ten specimens tested 4 August 17:12 to 17:25 exist twice with two different
roll codes: `...PF 1264 / AO 260701 LR 1379` and `...PF 1279 / BD 260708 LT 1397`.
Measurements are identical, so it is the same ten tests with two labels.

Evidence favours **1264**:
- 1264 exists in the extrusion table, extruded 1 July, average thickness
  117.7 microns, matching the 96 to 118 micron range measured
- 1279 was genuinely tested on 7 August (samples 1413 to 1422), so the labels
  were likely crossed while both files were open
- The file carrying 1279 is on the Excel-affected list; the other is not

**Not yet actioned.** Peter to confirm with whoever ran it, then delete the ten
1279-labelled rows for samples 1383 to 1392 and log it.

---

## 5. Working preferences

These matter. Several were stated explicitly and others became clear.

- **One clear ask per response.** Do not stack multiple tasks.
- **Succinct.** Brief answers, brief questions.
- **Full-file replacements** over "find this line" edits wherever practical.
- **Step by step**, confirming each step before moving on.
- **Explain roadblocks clearly** when they happen.
- **No em-dashes or en-dashes, ever.** Use a hyphen, comma, semicolon or colon.
- **No AI-tell sentence patterns.** Specifically avoid "Statement. Now the
  negated framing, then a snappy closing clause" and "Not A, not B. Just C."
- Prefers **GitHub-backed code and safe deployments**.
- Wants **dry runs before anything destructive**, and snapshots before writes.
- Appreciates being told when a proposal is wrong, and has overturned several
  suggestions on good grounds.
- Finds BigQuery unintuitive to query directly. **Prefers Looker.** This drove
  the decision to put pipeline health in Looker rather than expecting him to
  query a manifest table.
- Uses Cloud Shell. Has previously built a Google Apps Script tool with
  automated email reminders and liked it.

---

## 6. Operational traps that cost real time

Every one of these caused a silent failure during this work. They all share a
shape: a tool processed less than it was given and said nothing.

1. **`gsutil -m cp -I`** reads stdin in a child process and drops most input.
   Five paths in, two files out. Use wildcard copy and prune locally instead.
2. **`comm` requires matching sort order.** Python's sort and `sort -u` differ.
   Use `LC_ALL=C sort` on both sides, and check for the "not in sorted order"
   warning, which does not stop it returning a wrong answer.
3. **`while read` drops the final line if the file has no trailing newline.**
   Cost us one of six files silently.
4. **gsutil treats square brackets as wildcards.** `[WIP]` matches W, I or P.
   Use `?` instead. With `-q` the error is swallowed entirely.
5. **`rows` is a reserved word in BigQuery.** So are `range` and `groups`.
   Alias as `n_rows`. This bit three separate times.
6. **`bq query` defaults to 100 rows** unless `--max_rows` is set. A comparison
   against a 3,790-row table silently compared against 100.
7. **`bq` returns non-JSON output** for DDL and DML. Parse defensively.
8. **Cloud Shell upload does not overwrite.** It creates `main_(1).py`, often
   in the wrong directory. Always verify content after uploading, with a line
   count and a grep for a distinctive string, before deploying.
9. **Pasting heredocs into a terminal mangles them.** Upload or use the editor.
10. **GitHub push failures are silent in their consequences.** The 14 May
    tensile fix was committed locally and never pushed because auth failed and
    nobody retried. Three months of divergence from one unnoticed error.

The general lesson, and it belongs in v2: **assert expected counts at every
step**. Do not trust that a loop consumed everything or that a query returned
everything.

---

## 7. What is live now

### Extrusion alerting, working and proven end to end

- Log line `EXTRUSION_PIPELINE_FAILURE` emitted on any failure
- Log-based metric `extrusion_pipeline_failure`
- Notification channel `projects/notpla-machine-data/notificationChannels/14063382024575468776`
  (email to peter@notpla.com)
- Alert policy `projects/notpla-machine-data/alertPolicies/16570272964582018556`
- `autoClose: 86400s`, so a real failure will self-close after 24 hours whether
  or not anyone fixed it. Worth reviewing.
- No `severity` set, so emails read "No severity". Add `severity: ERROR` if wanted.

### Extrusion parser guards, added 21 August

- Rejects files with fewer than 10 recognised columns (`MIN_MAPPED_COLUMNS`)
- Rejects files with none of `trial_code`, `pellet_id`, `extrusion_id` present
- Drops rows with no identity value, fails if none remain
- Trims whitespace on free-text identifier columns
- Restored: move failed files to `FAILED_PREFIX` and re-raise

Before this, the parser accepted a file containing the literal text
`NotAHeader,AlsoNot` and loaded one row of nulls, logged as success.

---

## 8. The plan forward, with Peter's decisions

Full detail is in `pipeline-roadmap.md`. Peter's specific answers:

**Phase 0, complete** except the 1383-1392 roll code confirmation.

**Phase 1: know what is happening**
- 1.1 Manifest: **one shared table with a `pipeline` column**, not one per
  pipeline. Peter initially preferred per-pipeline for modularity but accepted
  the argument that per-pipeline views give the same thing while a shared table
  keeps the digest to one query and prevents schema drift.
- 1.2 Row-errors table: load good rows, capture bad ones with a reason.
- 1.3 **Hourly first-sighting alert.** Checks failed folders, emails on first
  sighting of a file, then records it so it never re-alerts. Route by
  `user_initials` via a lookup table. **Extrusion always goes to
  peter@notpla.com.** Fall back to the pellet ID's owner, then a default.
  Never fail silently because routing failed.
- 1.4 **Friday 09:00 digest.** Per test type: files processed, files failed,
  specimens ingested, most recent test date. Covers the silence case.
- 1.5 **Looker pipeline health page.** Peter's idea to adopt, and the right
  one given he reads Looker fluently and finds BigQuery unintuitive. Needs a
  `resolved_at` flag so the page shows live problems rather than an
  ever-growing historical count. Include per-pipeline row counts and last-seen
  dates so silence is visible, not just failures.

**Delivery mechanism for email: Google Apps Script**, because Peter has built
one before and liked it. Time-driven triggers, automated once set up. Two
caveats he should know: it runs under his account and sends as him, and
triggers can fail silently so the failure notification setting should be set
to notify immediately.

**Phase 2: v2 architecture.** Peter asked for **explanations when we arrive**
at 2.1 (shared parsing library) and 2.3 (schema from one definition), as he
does not currently follow what these mean.

**Phase 3: validation.** Peter asked whether these can be **tackled together
rather than individually** for cost efficiency. Answer given: yes, they belong
to the same validation layer and the same code path.

The agreed regexes:
- `pellet_id`: `^[A-Z]{2} [A-Z]{2} [A-Z]{2} [A-Z]{2} [0-9]{6} [A-Z]{2} [A-Z]{2} [0-9]{4}$`
- `extrusion_id`: `^[A-Z]{2} [0-9]{6} [A-Z]{2} [0-9]{4}$`

Flag, do not reject. Trim whitespace silently.

**Phase 4: migration.** One pipeline at a time, old and new in parallel,
compare row counts, then cut over. Suggested order: extrusion, friction,
tensile.

**Phase 5: friction curves.** The problem Peter raised at the outset. A
friction curve can be flat or violently oscillating and both currently reduce
to one mean. Long format is the answer. **He also wants this for tensile**,
and the 1,244 unprocessed tensile raw files make that the better prototype
since nothing depends on them.

**Phase 6: analysis layer.** A `films_results_long` view so selecting a
Pellet ID shows every test on that roll. Must be long, not a wide join, or
10 tensile against 15 friction specimens becomes 150 rows and every average is
wrong. Prefer a BigQuery view over Looker blending.

**A manual on the changes and how to interact with the system** is wanted, to
be written last so it describes what exists rather than what was planned.

---

## 9. Standing items not yet scheduled

- Extrusion table still has hanging whitespace; only tensile was cleaned
- `average_thickness_mm` in the extrusion table holds **microns**, not mm
- Over-privileged runtime: the three film functions run as the compute default
  account holding `roles/editor`. `leistritz-ingest-sa` shows the right pattern
- `mecmesin-uploader` holds a user-managed key from January 2026, never rotated,
  on a lab PC. The appspot account holds one from April 2025 plus `roles/editor`
- Naming has no convention: `films_tensile_london`, `machine_data`,
  `machine_collin_e25e`, `Rigid_Tensile`, and `machine_leistrtiz_1`, an empty
  dataset created by a typo
- No tests, no CI, no fixtures anywhere
- `backfill/backfill_friction_raw.py` is untracked in the repo
- The fine-grained GitHub token expires, typically in 30 days. When it does a
  push will fail exactly as it did today. Set a reminder or use a classic token
- VectorPro template naming: give copied templates distinct names so row 1
  becomes meaningful provenance
- The 4 friction rows with ~5.0 static CoF warrant investigation
- Recommend enabling bucket versioning

---

## 10. How to open the next conversation

Attach this document, `pipeline-review.md` and `pipeline-roadmap.md`.

Suggested opening: "Resuming the Notpla lab data pipeline work. Briefing
attached. Phase 0 is complete. Start Phase 1.1, the manifest table, one clear
ask at a time."
