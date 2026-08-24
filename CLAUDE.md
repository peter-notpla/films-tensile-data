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

The three leftover `mixedrow` test rows in `films_pipeline_row_errors` were
deleted once the streaming buffer cleared (24 August 2026, afternoon). That
table now holds only real data.

**Phase 1.3 (hourly first-sighting alert) is built and deployed, verification
paused mid-way (24 August 2026).** Original roadmap design (Cloud Function
sends email directly via SMTP) was rejected twice during scoping and replaced
with a design that reuses the existing Cloud Monitoring alerting pattern from
the extrusion alert (structured log line -> log-based metric -> alert policy
-> notification channel), specifically to avoid creating any new Google
account, alias, or app password. Full design reasoning is in
`~/.claude/plans/elegant-mixing-iverson.md`.

Routing: `KF` -> Katie, `ED` -> Emily, anything else (no initials column,
unrecognised initials, or extrusion, which has no initials field at all) ->
Peter, via `films_pipeline_ops.films_pipeline_user_directory`
(`user_initials` -> `route`).

Deployed:
- BQ tables `films_pipeline_ops.films_pipeline_user_directory` (seeded
  `KF`->`katie`, `ED`->`emily`) and `films_pipeline_ops.films_pipeline_alerts_sent`
  (dedup/audit log, pre-seeded with the 8 stale test-failure rows already in
  the manifest from 1.1/1.2 testing so they wouldn't fire on first run).
- Cloud Function `films-pipeline-failure-alerter`
  (`pipelines/films-pipeline-failure-alerter/`), gen2, HTTP-triggered, revision
  `films-pipeline-failure-alerter-00002-kiz`. Runs as new least-privilege SA
  `films-pipeline-alerter-sa` (dataset-level WRITER on `films_pipeline_ops` via
  the legacy ACL path, since dataset-level IAM policy bindings need an
  allowlist this project doesn't have; project-level `bigquery.jobUser`;
  bucket-level `storage.objectViewer`). Queries the manifest for `status='failed'`
  rows not yet in the dedup table, re-downloads the failed file to look for a
  `User Initials (Prompt For Value - After Test)` column, resolves a route, and
  prints one `PIPELINE_FAILURE_ALERT` log line per newly-failed file. Field
  order in that line is deliberately `route=... reason=... error=...` (error
  last), because `error_message` can contain embedded newlines that split into
  separate log entries and would otherwise silently truncate routing info.
- Log-based metric `pipeline_failure_alert` (labels: `pipeline`, `file`,
  `route`, `reason`, `error`), three new alert policies (`route="katie"` ->
  [Katie, Peter], `route="emily"` -> [Emily, Peter], `route="default"` ->
  [Peter]), two new notification channels (Katie, Emily) alongside the
  existing Peter one.
- Cloud Scheduler job `films-pipeline-failure-alert-hourly`, `0 * * * *`
  Europe/London, OIDC-authenticated as `films-pipeline-alerter-sa`.

**Verified so far:** uploaded one synthetic bad file per pipeline, each
carrying a real `user_initials` value where relevant
(`alert_test_katie_20260824_164432.csv` -> tensile, initials `KF`;
`alert_test_emily_20260824_164432.csv` -> friction, initials `ED`;
`alert_test_default_20260824_164432.csv` -> extrusion, no initials column).
All three failed as designed and landed in their pipeline's failed-processing
folder with a `status='failed'` manifest row. Manually invoked the alerter
twice: first call reported `{"checked":3,"alerted":3}` with correct
`route`/`route_reason` in both the function logs and the
`films_pipeline_alerts_sent` rows (`katie`/`initials:KF`,
`emily`/`initials:ED`, `default`/`no_initials_column`); second call reported
`{"checked":0,"alerted":0}`, confirming dedup. Queried the Monitoring API
directly and confirmed the log-based metric extracted all five labels
correctly for all three test incidents, with no truncation on the long
`error` fields.

**Email delivery confirmed (24 August 2026).** All 3 test alert emails
arrived correctly: Peter's inbox (all three policies attach to his channel),
plus Katie's and Emily's own channels each received their routed copy.
Remaining from the original plan, not yet done:
1. Clean up test artifacts: delete the 3 test files from each pipeline's
   failed-processing folder, delete the 3 test rows from
   `films_pipeline_manifest`, and delete the 3 test rows from
   `films_pipeline_alerts_sent` (mind the streaming-buffer delete restriction
   again, same as the 1.2 cleanup).
2. Update `pipeline-roadmap.md`'s Done list to mark 1.3 fully complete.

Nothing in this build touched or risked existing data: no changes to the
three processors, the manifest table, or the row-errors table.

### Resolved: pipeline alert email UX (24 August 2026)

Same motivation as the extrusion redesign below, applied to the 3 routed
policies (Katie/Emily/default): the raw Cloud Monitoring line and the
original documentation were too technical for Katie and Emily to self-serve
from. Reworked using the same two-tier pattern (plain-language summary,
then a technical section), plus three things new to this pass:

- **Readable pipeline names.** `main.py` now maps `tensile`/`friction`/
  `extrusion` to "Tensile Testing"/"Friction Testing"/"Extrusion" and prints
  it as a new `pipeline_readable` field in the `PIPELINE_FAILURE_ALERT` log
  line (Cloud Monitoring's doc template can't do conditional text, so this
  had to be computed upstream, not templated). Deployed as
  `films-pipeline-failure-alerter-00003-qis`.
- **A real failure timestamp.** Added `failed_at`, sourced from the
  manifest's `processed_at` (when the file actually failed), not Cloud
  Monitoring's built-in "Start time" field (when the hourly check happened
  to notice, up to ~60 minutes later, and which can't be relabelled or
  removed, it's fixed platform chrome above our custom content).
- **Subject line carries the plain-language framing.** Policy
  `displayName` drives the email subject, so that's now "Notpla Data
  Pipeline Warning: routed to Katie" etc., rather than the technical
  condition text. The fixed Cloud Monitoring summary line still appears
  above our content in the body, unavoidably; the subject and everything
  below it now lead in plain English instead.
- **Resolved/closed email suppressed.** `alertStrategy.notificationPrompts`
  set to `[OPENED]` on all three policies. The auto-close notification
  fired whenever no *new* failure landed in the next check window, not
  when the file was actually fixed, so it was actively misleading and has
  been turned off rather than reworded.

New `pipeline_failure_alert` metric labels: `pipeline_readable`,
`failed_at`, alongside the existing `pipeline`, `file`, `route`, `reason`,
`error`. Extractor regexes verified against a synthetic log line before
deploying (all 7 fields extracted cleanly, no truncation).

Row numbers were explicitly considered and dropped: this alert only fires
on whole-file failures (`manifest.status='failed'`), there is no single bad
row to point at. Row-level detail only exists for the separate, non-alerting
`films_pipeline_row_errors` path (a file that partially succeeded).

### Resolved: auto-cleanup and repeat-failure escalation (24 August 2026)

Both follow-on features from the email UX pass, now built, deployed, and
verified live end-to-end (not just locally).

**1. Auto-delete stale failed copy on successful reprocessing.** All three
processors (`films-tensile-csv-processor-00017-san`,
`films-friction-csv-processor-00008-ziq`,
`films-extrusion-csv-processor-00011-qok`) now call a
`delete_stale_failed_copy` helper right after a successful move to
`*-processed/`: if a file of the same name exists under `*-failed-processing/`,
it's deleted, best-effort, matching the existing manifest/row-errors
wrapped-try pattern. Needed because `move_blob` writes to
`{FAILED_PREFIX}{filename}` with no uniquifier, so a fixed file's success
never used to clean up the old failed copy sitting under a different
prefix.

**2. Repeat-failure escalation.** The alerter
(`films-pipeline-failure-alerter-00004-muw`) now looks up, for every new
failure, the most recent prior manifest row for the same pipeline+filename
via a new `find_previous_failure` query. If that prior row was also
`status='failed'` (no successful ingest for that filename since), it prints
a second, distinct `PIPELINE_FAILURE_ESCALATION` log line alongside the
normal `PIPELINE_FAILURE_ALERT` line, carrying both the current and
previous error messages and timestamps. Backed by a new log-based metric
`pipeline_failure_escalation` and a new alert policy (severity ERROR,
routed to Peter only, `notificationPrompts: [OPENED]`,
`projects/notpla-machine-data/alertPolicies/2248087195614091607`). Matched
on filename rather than checksum, since a fixed re-upload is a new
checksum; the documented fix procedure (see the Katie/Emily/default policy
docs) keeps the filename unchanged, which is what makes filename matching
reliable. Detected via manifest history, not by checking for two files on
disk: `copy_blob` to an existing destination name in GCS silently
overwrites it (versioning is Suspended), so the "two files with the same
name" state a same-name repeat failure was originally expected to produce
never actually occurs, filename-based manifest lookup is the reliable
signal instead.

**Two real bugs found and fixed during this build, both introduced by this
build:**
- The `error=(.*)"` label extractor for the escalation metric initially
  matched *inside* `previous_error=` (which contains the substring
  `error=`), corrupting the current error field. Caught by testing the
  regex against a realistic string before deploying, not live. Fixed by
  anchoring on `" error="` (leading space) instead.
- `delete_stale_failed_copy` in the tensile processor originally called
  `logger.info(..., extra={"filename": filename})`. Python's `logging`
  module reserves `filename` as a built-in `LogRecord` attribute, so this
  raised `Attempt to overwrite 'filename' in LogRecord` on every call,
  which (since it fired after the BigQuery load and file move had already
  succeeded) caused an otherwise-successful reprocess to be logged as a
  manifest failure. Caught live during verification testing, not by code
  review. Fixed by renaming the extra key to `target_filename`, redeployed
  as `films-tensile-csv-processor-00017-san` (the revision number above
  already reflects the fix, not the bug).

**Verified live**, using disposable synthetic files, same convention as
every other phase in this project: uploaded a bad file, confirmed it
failed; re-uploaded a fixed version under the same name, confirmed success
and confirmed the stale failed-processing copy was actually deleted (not
just logically expected to be); uploaded two different-content bad files in
a row under a second test filename, manually invoked the alerter, and
confirmed via the function logs and a direct Monitoring API query that the
escalation metric fired with all seven labels extracted correctly and
uncorrupted, including on real (not synthetic) error text from the bugs
above.

Test artifacts partially cleaned up: the two duplicate rows written to the
real `films_tensile_results` table (`sample=999999001`, a byproduct of the
logging bug above causing one successful reprocess to be retried) were
deleted, and both test files were deleted from GCS. **Still pending, same
streaming-buffer restriction as every prior phase:** 5 manifest rows and 4
alerts_sent rows for `source_file LIKE '%alert_test_cleanup_20260824_1900.csv%'`
or `'%alert_test_escalation_20260824_1900.csv%'` are stuck in the streaming
buffer (inserted ~19:00-19:06 UTC, 24 August). Delete once the buffer
clears (up to ~90 minutes after insert):
```sql
DELETE FROM `notpla-machine-data.films_pipeline_ops.films_pipeline_manifest`
WHERE source_file LIKE '%alert_test_cleanup_20260824_1900.csv%'
   OR source_file LIKE '%alert_test_escalation_20260824_1900.csv%'
```
```sql
DELETE FROM `notpla-machine-data.films_pipeline_ops.films_pipeline_alerts_sent`
WHERE source_file LIKE '%alert_test_cleanup_20260824_1900.csv%'
   OR source_file LIKE '%alert_test_escalation_20260824_1900.csv%'
```

One live escalation email should have gone to peter@notpla.com from the
verification test above; not yet confirmed by Peter reading his inbox.

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
