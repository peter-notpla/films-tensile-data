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

**Phase 1.1 (manifest table) is complete, as of 24 August 2026.** All three
Cloud Functions are deployed with manifest logging and each was sanity-checked
individually with a controlled bad CSV (`NotAHeader,AlsoNot`) uploaded to its
watch prefix:

| Pipeline | Revision | Manifest `error_message` on bad CSV |
|---|---|---|
| tensile | `films-tensile-csv-processor-00015-jaw` | `CSV too short (needs title + header + data)` |
| friction | `films-friction-csv-processor-00005-dun` | `No data rows` |
| extrusion | `films-extrusion-csv-processor-00009-gis` | `Only 0 recognised columns (need at least 10)...` |

Each test file was confirmed in its failed-processing folder, confirmed as a
`status='failed'` row in `films_pipeline_manifest` with a populated
`error_message`, then deleted from the bucket (bucket versioning is
Suspended, so this was a real delete, not a soft one). The extrusion result
also re-confirms the Phase 0.1 fix (`move_blob` + `raise`) is live: the bad
file was routed to failed-processing, not silently dropped.

The auth issue from the prior session (`service account info is missing
'email' field`) did not recur; the Cloud Shell VM restart fixed it.

**Phase 1.2 (row-errors table) is complete, as of 24 August 2026.** New
table `films_pipeline_ops.films_pipeline_row_errors`: `pipeline`,
`source_file`, `checksum`, `row_number`, `reason`, `raw_row` (original row
as JSON), `processed_at`. Writes are best-effort, same wrapped pattern as
the manifest table.

This phase turned out to be two bug fixes, not just a new table. Before this
work, tensile and friction could both lose an entire file to a single bad
row:
- **Tensile:** the last timestamp-format fallback used
  `pd.to_datetime(..., errors="raise")` instead of `coerce`. One row with a
  timestamp in none of the four known formats killed the whole file.
- **Friction:** `if (blank sample).any(): raise` and
  `df["timestamp_start"].apply(parse_ts)` (which itself raised) both failed
  the entire file on the first bad row, with no partial-load path at all.

Both now drop only the bad row, load everything else, and write the bad row
(1-based position, reason, full original values as JSON) to the row-errors
table. Extrusion already dropped identity-less rows safely; it now also
captures them to row-errors instead of just counting them.

A second, subtler bug turned up in **friction** while testing this:
the "drop blank padding rows" filter checked only whether the *first* CSV
column (`Sample`, confirmed against real processed files) was blank, not
the whole row. A row with a blank Sample but real data in every other
column was silently swallowed by that filter before ever reaching the new
row-errors logic, exactly the "no trace" failure this phase was meant to
fix. Changed to check whether every column is blank (tensile already did
this correctly). Deployed as `films-friction-csv-processor-00007-sed`.

Verified live, not just locally: uploaded a file per pipeline with one good
row (marked `sample=999999999`/`trial_code=ROWERRTEST`) and one bad row.
Confirmed for all three: the file moved to `processed` (not `failed`), the
good row landed in the real results table, and the bad row landed in
`films_pipeline_row_errors` with the correct reason and raw values. All
test rows and files then deleted, **except** the three test rows in
`films_pipeline_row_errors` itself: BigQuery blocks DELETE on rows still in
the streaming buffer (up to ~90 minutes after a streaming insert). They're
easy to identify (`source_file LIKE '%mixedrow%'`) and harmless (that table
had no other data yet); delete on the next session once the buffer clears:
```sql
DELETE FROM `notpla-machine-data.films_pipeline_ops.films_pipeline_row_errors`
WHERE source_file LIKE '%mixedrow%'
```

Revisions deployed this phase: `films-tensile-csv-processor-00016-doj`,
`films-friction-csv-processor-00007-sed`, `films-extrusion-csv-processor-00010-duy`.

**Next queued:** 1.3, the hourly first-sighting alert.

### Resolved: extrusion alert email UX (24 August 2026)

The two alert emails Peter received were confirmed as a side effect of the
Phase 1.1 sanity-check upload, not a new organic failure. Root cause of the
missing detail: `main.py` already prints
`EXTRUSION_PIPELINE_FAILURE file=... error=...`, but the log-based metric
had no label extraction, so the alert had nothing to reference.

Changes made to the live policy and metric (not the repo; these are GCP
config, not code):
- `extrusion_pipeline_failure` log-based metric: filter narrowed to the
  `file=...error=...` line specifically (was matching any
  `EXTRUSION_PIPELINE_FAILURE` substring, which double-counted the rarer
  "could not move to failed prefix" line); added `file` and `error` STRING
  labels via `labelExtractors` (`REGEXP_EXTRACT`, using `[^ ]+` rather than
  `\S+`, since the extractor DSL's string-literal parser rejects `\S` as an
  unsupported escape sequence).
- Alert policy `16570272964582018556`: added `severity: WARNING` (chosen
  because a caught, quarantined file is not data loss, consistent with the
  flag-don't-reject philosophy elsewhere in this project); condition
  aggregation now has `groupByFields: [metric.label.file]` so each incident
  carries its own file's labels; documentation rewritten as a plain-language
  bulleted summary followed by a technical section with `${metric.label.file}`
  and `${metric.label.error}` inlined, plus a note that the "View Logs"
  button's default time window doesn't always cover an already-resolved
  incident.

Verified: regex tested against a real production log line (correct
extraction of both fields), sanity-check file confirmed processed and moved
to failed-processing as before. A live alert email was triggered to confirm
the new format end-to-end; test file then deleted from the bucket.

Config for both lives only in GCP (`gcloud logging metrics describe
extrusion_pipeline_failure`, `gcloud alpha monitoring policies describe
projects/notpla-machine-data/alertPolicies/16570272964582018556`), not
version-controlled. Worth a follow-up if there's ever a Terraform/config
pass over this project's alerting.

Not done: item 3 (the logs link showing nothing) was addressed by adding a
note in the alert body about widening the time range, not by fixing the
link's underlying default window, since that's a Cloud Monitoring platform
behaviour, not something this policy controls directly.

This same pattern (severity, label extraction, two-tier documentation) is
the template to reuse when tensile and friction get alert policies under
Phase 1.3, where the audience is less technical than Peter.

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
