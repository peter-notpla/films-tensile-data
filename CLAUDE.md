# CLAUDE.md

Project context for Claude Code. This file is read automatically at the start
of every session in this repository.

---

## What this project is

The Notpla lab data pipeline. Test results from lab instruments flow into
BigQuery and a Looker Studio dashboard.

```
Mecmesin tensile tester (VectorPro)
  -> CSV export
  -> manual check folder on a Windows PC (files are opened in Excel here)
  -> Windows Task Scheduler uploads to GCS every ~10 minutes
  -> Eventarc fires a Cloud Function per test type
  -> BigQuery table per test type
  -> Looker Studio, one page per test type
```

GCP project `notpla-machine-data`, region `europe-west2`.

---

## Working style

- **One clear ask per response.** Do not stack multiple tasks.
- **Succinct.** Brief answers, brief questions.
- **Full-file replacements** over "find this line" edits wherever practical.
- **Dry run before anything destructive.** Snapshot BigQuery tables before writes.
- **Never use em-dashes or en-dashes.** Use a hyphen, comma, semicolon or colon.
- **Avoid AI-tell sentence patterns**, specifically "Statement. Now the negated
  framing, then a snappy closing clause" and "Not A, not B. Just C."
- Peter finds BigQuery unintuitive to query directly and prefers Looker.
- Explain roadblocks clearly when they happen.

---

## Specimen key model

Settled 20 August 2026 after testing every alternative against the full history.

| Purpose | Fields |
|---|---|
| Identity | `machine_id`, `test_type`, `timestamp_minute`, `sample` |
| Provenance | `template_name` (row 1 of the CSV), `source_file`, full `timestamp_start` |
| Analysis | `pellet_id`, `extrusion_id`, `test_direction` |

`timestamp_minute + sample` produced 2,990 distinct keys with zero conflations.
Alternatives were rejected on evidence: `timestamp + stress + modulus` had 1
conflation, and adding `pellet_id` fragmented identity because it is a mutable
field that gets corrected by hand.

**Minute resolution is deliberate.** See the Excel section below.

---

## Things that are true and non-obvious

### Sample numbers are not stable identifiers

VectorPro assigns sample numbers per test template. Copying a template, which
is required for any major edit, restarts the counter at 1. Two manual Excel
backfills then rewrote them differently: tensile was **resequenced** from
1,000,000, friction was **offset** by exactly +1,000,000. Neither is recorded
anywhere.

This is why an early reconciliation reported 1,020 missing specimens when the
true figure was 52. Never key on sample number alone.

### Excel destroys precision, and this is accepted

The manual check step involves opening files in Excel and saving. Excel:
- drops seconds from timestamps (`09:54:00` becomes `09:54`)
- strips trailing zeros from numbers (`0.810` becomes `0.81`)
- pads row 1 with trailing commas

Peter has decided to keep this workflow. Any key must tolerate minute
resolution, and any comparison of measurements must be **numeric, never
string**.

Detection signatures for flagging: row 1 ending in comma padding, or every
timestamp in a file having zero seconds.

### ID formats, decoded field by field

- `pellet_id`: `^[A-Z]{2} [A-Z]{2} [A-Z]{2} [A-Z]{2} [0-9]{6} [A-Z]{2} [A-Z]{2} [0-9]{4}$`
- `extrusion_id`: `^[A-Z]{2} [0-9]{6} [A-Z]{2} [0-9]{4}$`

A full roll code is `pellet_id` + `extrusion_id` concatenated, e.g.
`EV AB AL AM 260310 LI PF 1133 BA 260324 KM 1279`. Per-field meaning:

| # | Field | Example | Meaning |
|---|---|---|---|
| 1 | 2 letters | `EV` | formulation ingredients |
| 2 | 2 letters | `AB` | proportions of those ingredients |
| 3 | 2 letters | `AL` | unique batch code |
| 4 | 2 letters | `AM` | machine used for compounding |
| 5 | 6 digits | `260310` | date of compounding, `YYMMDD` |
| 6 | 2 letters | `LI` | process settings for compounding |
| 7 | 2 letters | `PF` | product: `PF` = Pellet Films, `PR` = Pellet Rigids |
| 8 | 4 digits | `1133` | unique identifier for the bag of pellets |
| 9 | 2 letters | `BA` | machine used for cast film extrusion (PF only, Peter's focus) |
| 10 | 6 digits | `260324` | date of extrusion, `YYMMDD` |
| 11 | 2 letters | `KM` | extrusion processing code |
| 12 | 4 digits | `1279` | unique identifier for the roll |

Fields 1-8 are `pellet_id`, fields 9-12 are `extrusion_id`. Note field 8 (bag
ID) and field 12 (roll ID) are both 4 digits but identify different things,
which is the source of the 1264/1279 shorthand confusion in the Open item
below: "1264" and "1279" there refer to field 8, the pellet bag ID, not the
roll ID.

**Flag, do not reject.** A hard gate would have discarded roughly 150
legitimate rows from a one-off packaging study to catch 25 typos.

Trim whitespace silently. It carries no information and is invisible in every UI.

### Friction has two static CoF columns by design

VectorPro's peak function sometimes fails to detect static CoF, so
`backup_static_cof` was added as a fallback. Older files lack it. Precedence:
use static where present, fall back to backup, record which was used.

---

## Tool traps that have caused silent failures here

Every one of these processed less than it was given and said nothing.

1. `gsutil -m cp -I` drops most of stdin. Use wildcard copy and prune locally.
2. `comm` needs matching sort order. Use `LC_ALL=C sort` on both sides.
3. `while read` drops the last line if there is no trailing newline.
4. gsutil treats `[` `]` as wildcards. `[WIP]` matches W, I or P. Use `?`.
5. `rows`, `range` and `groups` are reserved words in BigQuery. Alias as `n_rows`.
6. `bq query` defaults to 100 rows. Always set `--max_rows`.
7. `bq` returns non-JSON output for DDL and DML. Parse defensively.
8. Pasting heredocs into a terminal mangles them.
9. A failed `git push` caused three months of undetected repo divergence.

**Assert expected counts at every step.** Do not trust that a loop consumed
everything or that a query returned everything.

---

## Repository layout

```
pipelines/films-tensile-csv-processor/     deployed, fixed 14 May 2026
pipelines/films-friction-csv-processor/    deployed, all-STRING schema
pipelines/films-extrusion-csv-processor/   deployed, fixed 21 Aug 2026
pipelines/films-friction-raw-processor/    NOT deployed
backfill/                                  one-off scripts, legacy
```

As of 21 August 2026 the repo matches what is deployed. Keep it that way:
commit and push in the same session as any deploy.

---

## Deploy and verify

```bash
cd ~/films-tensile-data/pipelines/<pipeline>
python3 -m py_compile main.py && echo "compiles clean"
gcloud functions deploy <function-name> --region=europe-west2 --gen2 --source=. --quiet
```

Always verify a deploy took effect by checking the logs for a distinctive
string from the new code, not by assuming.

---

## Live alerting

Extrusion only, as at 21 August 2026:

- Log line `EXTRUSION_PIPELINE_FAILURE`
- Metric `extrusion_pipeline_failure`
- Channel `projects/notpla-machine-data/notificationChannels/14063382024575468776`
- Policy `projects/notpla-machine-data/alertPolicies/16570272964582018556`
- `autoClose: 86400s`, so failures self-close after 24 hours regardless

Wider alerting is Phase 1 work. Extrusion errors always route to
peter@notpla.com.

---

## Current state

Phase 0 of `pipeline-roadmap.md` is fully complete, including 0.4 (the
1264/1279 roll code correction, resolved 23 August 2026).

Phase 1.1 (manifest table), as of 23 August 2026:
- `films_pipeline_ops.films_pipeline_manifest` created. New shared dataset
  `films_pipeline_ops` holds this and will hold the row-errors table (1.2),
  since neither belongs to any single test type's existing dataset.
- All three `main.py` files (tensile, friction, extrusion) updated to insert
  one manifest row per file processed, success or failure. Each write is
  wrapped so a manifest failure can never break the actual pipeline. Compiles
  clean. **Committed and pushed, but NOT YET DEPLOYED.** The three live
  Cloud Functions are still running the pre-manifest code.

**Blocked on:** gcloud/bq auth broke mid-session in this Cloud Shell
(`service account info is missing 'email' field` from the metadata server,
not a normal expired token — `gcloud auth login`'s own output confirms Cloud
Shell shouldn't need it). Likely fix: restart the Cloud Shell VM (⋮ menu →
Restart, or close/reopen Cloud Shell), which should be enough on its own.

**Next step on resume:** verify `bq query --use_legacy_sql=false "SELECT 1"`
works, then deploy the three functions one at a time:
```
cd ~/films-tensile-data/pipelines/<pipeline>
gcloud functions deploy <function-name> --region=europe-west2 --gen2 --source=. --quiet
```
Function names: `films-tensile-csv-processor`, `films-friction-csv-processor`,
`films-extrusion-csv-processor`. After each deploy, sanity-check with a
controlled bad CSV (e.g. the `NotAHeader,AlsoNot` case) uploaded to that
pipeline's watch prefix: confirm it lands in the failed-processing folder as
before, then query `films_pipeline_manifest` for a `status='failed'` row with
a matching `source_file` and populated `error_message`. This exercises the
new code without inserting anything into a real results table. Only once all
three are confirmed does 1.1 count as actually done; then move on to 1.2
(row-errors table), which 1.3/1.4/1.5 depend on.

All failed-processing folders are empty. Anything appearing in them is a live
problem.

Genuine backlogs, deliberately untouched:
- 738 friction raw files failed, no processor deployed
- 197 friction raw files queued
- 1,244 tensile raw sample files, no processor watches that prefix

**Bucket versioning is Suspended. Any delete is permanent.**

---

## Resolved: 1264/1279 roll code ambiguity

Samples 1383 to 1392 (tested 4 August, 17:12-17:25) existed twice under two
candidate roll codes. Lab tech confirmed roll **1264** (`AO 260701 LR 1379`)
on 23 August 2026. The ten rows carrying the erroneous `1279`
(`BD 260708 LT 1397`) combo were deleted and logged to `id_corrections_log`.
Snapshot taken first: `films_tensile_results_presnap_20260823`.
`films_tensile_results` is now 3,510 rows.
