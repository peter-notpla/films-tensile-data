# Lab data pipeline: build history

Detailed log of what was built, verified, and fixed, in the order it happened.
For current phase status and what's next, see `pipeline-roadmap.md`. For
durable facts that should steer any session's work, see `CLAUDE.md`.

---

## Phase 1.1: manifest table (24 August 2026)

All three Cloud Functions deployed with manifest logging, each individually
sanity-checked with a controlled bad CSV (`NotAHeader,AlsoNot`) uploaded to
its watch prefix:

| Pipeline | Revision | Manifest `error_message` on bad CSV |
|---|---|---|
| tensile | `films-tensile-csv-processor-00015-jaw` | `CSV too short (needs title + header + data)` |
| friction | `films-friction-csv-processor-00005-dun` | `No data rows` |
| extrusion | `films-extrusion-csv-processor-00009-gis` | `Only 0 recognised columns (need at least 10)...` |

Each test file was confirmed in its failed-processing folder, confirmed as a
`status='failed'` row in `films_pipeline_manifest` with a populated
`error_message`, then deleted from the bucket (bucket versioning is
Suspended, so this was a real delete, not a soft one). The extrusion result
also re-confirmed the Phase 0.1 fix (`move_blob` + `raise`) is live: the bad
file was routed to failed-processing, not silently dropped.

The auth issue from the prior session (`service account info is missing
'email' field`) did not recur; the Cloud Shell VM restart fixed it.

## Phase 1.2: row-errors table (24 August 2026)

New table `films_pipeline_ops.films_pipeline_row_errors`: `pipeline`,
`source_file`, `checksum`, `row_number`, `reason`, `raw_row` (original row as
JSON), `processed_at`. Writes are best-effort, same wrapped pattern as the
manifest table.

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

A second, subtler bug turned up in **friction** while testing this: the
"drop blank padding rows" filter checked only whether the *first* CSV column
(`Sample`, confirmed against real processed files) was blank, not the whole
row. A row with a blank Sample but real data in every other column was
silently swallowed by that filter before ever reaching the new row-errors
logic, exactly the "no trace" failure this phase was meant to fix. Changed
to check whether every column is blank (tensile already did this correctly).
Deployed as `films-friction-csv-processor-00007-sed`.

Verified live, not just locally: uploaded a file per pipeline with one good
row (marked `sample=999999999`/`trial_code=ROWERRTEST`) and one bad row.
Confirmed for all three: the file moved to `processed` (not `failed`), the
good row landed in the real results table, and the bad row landed in
`films_pipeline_row_errors` with the correct reason and raw values. All test
rows and files then deleted, including the three leftover `mixedrow` rows in
`films_pipeline_row_errors` (blocked initially by BigQuery's streaming
buffer, deleted once it cleared later the same day).

Revisions deployed this phase: `films-tensile-csv-processor-00016-doj`,
`films-friction-csv-processor-00007-sed`, `films-extrusion-csv-processor-00010-duy`.

## Phase 1.3: hourly first-sighting alert (24 August 2026)

Original roadmap design (Cloud Function sends email directly via SMTP) was
rejected twice during scoping and replaced with a design that reuses the
existing Cloud Monitoring alerting pattern from the extrusion alert
(structured log line -> log-based metric -> alert policy -> notification
channel), specifically to avoid creating any new Google account, alias, or
app password. Full design reasoning is in
`~/.claude/plans/elegant-mixing-iverson.md`.

Routing: `KF` -> Katie, `ED` -> Emily, anything else (no initials column,
unrecognised initials, or extrusion, which has no initials field at all) ->
Peter, via `films_pipeline_ops.films_pipeline_user_directory`
(`user_initials` -> `route`).

Deployed:
- BQ tables `films_pipeline_ops.films_pipeline_user_directory` (seeded
  `KF`->`katie`, `ED`->`emily`) and
  `films_pipeline_ops.films_pipeline_alerts_sent` (dedup/audit log,
  pre-seeded with the 8 stale test-failure rows already in the manifest from
  1.1/1.2 testing so they wouldn't fire on first run).
- Cloud Function `films-pipeline-failure-alerter`
  (`pipelines/films-pipeline-failure-alerter/`), gen2, HTTP-triggered,
  revision `films-pipeline-failure-alerter-00002-kiz`. Runs as new
  least-privilege SA `films-pipeline-alerter-sa` (dataset-level WRITER on
  `films_pipeline_ops` via the legacy ACL path, since dataset-level IAM
  policy bindings need an allowlist this project doesn't have;
  project-level `bigquery.jobUser`; bucket-level `storage.objectViewer`).
  Queries the manifest for `status='failed'` rows not yet in the dedup
  table, re-downloads the failed file to look for a `User Initials (Prompt
  For Value - After Test)` column, resolves a route, and prints one
  `PIPELINE_FAILURE_ALERT` log line per newly-failed file. Field order in
  that line is deliberately `route=... reason=... error=...` (error last),
  because `error_message` can contain embedded newlines that split into
  separate log entries and would otherwise silently truncate routing info.
- Log-based metric `pipeline_failure_alert` (labels: `pipeline`, `file`,
  `route`, `reason`, `error`), three new alert policies (`route="katie"` ->
  [Katie, Peter], `route="emily"` -> [Emily, Peter], `route="default"` ->
  [Peter]), two new notification channels (Katie, Emily) alongside the
  existing Peter one.
- Cloud Scheduler job `films-pipeline-failure-alert-hourly`, `0 * * * *`
  Europe/London, OIDC-authenticated as `films-pipeline-alerter-sa`.

Verified: uploaded one synthetic bad file per pipeline, each carrying a real
`user_initials` value where relevant. All three failed as designed. First
alerter call reported `{"checked":3,"alerted":3}` with correct routing;
second call reported `{"checked":0,"alerted":0}`, confirming dedup.
Monitoring API confirmed all five labels extracted correctly with no
truncation. Email delivery confirmed the same day: Peter, Katie, and Emily
each received their routed copy. Test artifacts (files, manifest rows,
alerts_sent rows) fully cleaned up.

## Pipeline alert email UX (24 August 2026)

Same motivation as the extrusion redesign below, applied to the 3 routed
policies (Katie/Emily/default): the raw Cloud Monitoring line was too
technical for Katie and Emily to self-serve from. Reworked using the same
two-tier pattern (plain-language summary, then a technical section), plus:

- **Readable pipeline names.** `main.py` maps `tensile`/`friction`/
  `extrusion` to "Tensile Testing"/"Friction Testing"/"Extrusion" and prints
  a new `pipeline_readable` field (Cloud Monitoring's doc template can't do
  conditional text, so this had to be computed upstream). Deployed as
  `films-pipeline-failure-alerter-00003-qis`.
- **A real failure timestamp.** Added `failed_at`, sourced from the
  manifest's `processed_at`, not Cloud Monitoring's "Start time" (up to ~60
  minutes later, and fixed platform chrome that can't be relabelled).
- **Subject line carries the plain-language framing.** See "alert subject
  lines" below for the root-cause correction.
- **Resolved/closed email suppressed.**
  `alertStrategy.notificationPrompts` set to `[OPENED]` on all three
  policies; the auto-close notification fired on no *new* failure in the
  next check window, not on an actual fix, so it was misleading.

New `pipeline_failure_alert` metric labels: `pipeline_readable`, `failed_at`.
Extractor regexes verified against a synthetic log line before deploying.

Row numbers were considered and dropped: this alert only fires on
whole-file failures, there is no single bad row to point at. Row-level
detail exists only in the separate, non-alerting
`films_pipeline_row_errors` path.

## Auto-cleanup and repeat-failure escalation (24 August 2026)

**1. Auto-delete stale failed copy on successful reprocessing.** All three
processors (`films-tensile-csv-processor-00017-san`,
`films-friction-csv-processor-00008-ziq`,
`films-extrusion-csv-processor-00011-qok`) now call a
`delete_stale_failed_copy` helper right after a successful move to
`*-processed/`. Needed because `move_blob` writes to
`{FAILED_PREFIX}{filename}` with no uniquifier, so a fixed file's success
never used to clean up the old failed copy sitting under a different
prefix.

**2. Repeat-failure escalation.** The alerter
(`films-pipeline-failure-alerter-00004-muw`) looks up, for every new
failure, the most recent prior manifest row for the same pipeline+filename.
If that prior row was also `status='failed'`, it prints a second
`PIPELINE_FAILURE_ESCALATION` log line alongside the normal alert, carrying
both current and previous error messages/timestamps. Backed by metric
`pipeline_failure_escalation` and a new alert policy (severity ERROR,
routed to Peter only, `alertPolicies/2248087195614091607`). Matched on
filename rather than checksum, since a fixed re-upload is a new checksum;
the documented fix procedure keeps the filename unchanged. Detected via
manifest history, not two files on disk: `copy_blob` to an existing
destination silently overwrites (versioning Suspended), so filename-based
manifest lookup is the reliable signal.

Two real bugs found and fixed during this build, both introduced by this
build:
- The `error=(.*)"` label extractor for the escalation metric initially
  matched *inside* `previous_error=`, corrupting the current error field.
  Caught by testing the regex before deploying. Fixed by anchoring on
  `" error="` (leading space).
- `delete_stale_failed_copy` in the tensile processor called
  `logger.info(..., extra={"filename": filename})`. Python's `logging`
  module reserves `filename` as a built-in `LogRecord` attribute, so this
  raised on every call, which (since it fired after the BigQuery load and
  file move had already succeeded) caused an otherwise-successful reprocess
  to be logged as a manifest failure. Caught live during verification, not
  code review. Fixed by renaming the key to `target_filename`, redeployed
  as `films-tensile-csv-processor-00017-san`.

Verified live end to end with disposable synthetic files. Test artifacts
fully cleaned up, including two duplicate rows in `films_tensile_results`
(`sample=999999001`) caused by the logging bug above, and manifest/
alerts_sent rows stuck in the streaming buffer (cleared and deleted at
20:01 UTC).

One live escalation email should have gone to peter@notpla.com from the
verification test; not yet confirmed by Peter reading his inbox.

## Alert subject lines and extrusion recovery emails (24 August 2026)

Peter reported the Katie-routed subject line was rendering as the raw Cloud
Monitoring default. Root cause: contrary to what was assumed during the
email UX pass above, it's the alert **condition's** `displayName` that
drives the subject, not the policy's `displayName`. GCP appends an
`on {project} ... labels {...}` suffix itself; no policy field controls or
removes it.

Fixed by renaming the single condition on all 5 policies (full JSON
snapshotted first, replaced via `gcloud alpha monitoring policies update
--policy-from-file`; `--fields` only accepts `disabled`/
`notificationChannels` so a full-object replace was used instead):

| Policy | New condition `displayName` |
|---|---|
| Katie | `Pipeline Failure - For Katie` |
| Emily | `Pipeline Failure - For Emily` |
| Default | `Pipeline Failure - For Peter` |
| Escalation | `Pipeline Failure - Repeat, For Peter` |
| Extrusion | `Pipeline Failure - Extrusion, For Peter` |

Not independently re-verified against a real inbound email yet (no mail
access from this session); Peter to confirm on the next real alert.

Same investigation surfaced a second bug: Peter was still getting "Alert
recovered" emails despite the intent to suppress them everywhere. The
extrusion policy (`16570272964582018556`, the oldest of the five, predating
the UX pass) had never had `alertStrategy.notificationPrompts` set. Fixed
to match the other four (`notificationPrompts: ["OPENED"]`).

## Extrusion alert email UX (24 August 2026)

The two alert emails Peter received were confirmed as a side effect of the
Phase 1.1 sanity-check upload, not a new organic failure. Root cause of the
missing detail: `main.py` already prints
`EXTRUSION_PIPELINE_FAILURE file=... error=...`, but the log-based metric
had no label extraction, so the alert had nothing to reference.

Changes made to the live policy and metric (GCP config, not code, so not
version-controlled):
- `extrusion_pipeline_failure` metric: filter narrowed to the
  `file=...error=...` line specifically (was matching any
  `EXTRUSION_PIPELINE_FAILURE` substring, double-counting the rarer
  "could not move to failed prefix" line); added `file` and `error` STRING
  labels via `labelExtractors` (`REGEXP_EXTRACT`, using `[^ ]+` rather than
  `\S+`, since the extractor DSL rejects `\S` as an unsupported escape).
- Alert policy `16570272964582018556`: added `severity: WARNING` (a caught,
  quarantined file is not data loss); condition aggregation now has
  `groupByFields: [metric.label.file]` so each incident carries its own
  file's labels; documentation rewritten as plain-language summary plus
  technical section with `${metric.label.file}` and `${metric.label.error}`
  inlined, plus a note that the "View Logs" button's default time window
  doesn't always cover an already-resolved incident.

Verified: regex tested against a real production log line, sanity-check
file confirmed processed and moved to failed-processing, live alert email
triggered and confirmed end-to-end.

Not done: the logs link showing nothing on an already-resolved incident was
addressed with a note in the alert body, not a fix to the link's default
window (Cloud Monitoring platform behaviour, not something this policy
controls).

This pattern (severity, label extraction, two-tier documentation) is the
template reused for the tensile/friction alert policies in Phase 1.3.

## Resolved: 1264/1279 roll code ambiguity (23 August 2026)

Samples 1383 to 1392 (tested 4 August, 17:12-17:25) existed twice under two
candidate roll codes. Lab tech confirmed roll **1264** (`AO 260701 LR 1379`).
The ten rows carrying the erroneous `1279` (`BD 260708 LT 1397`) combo were
deleted and logged to `id_corrections_log`. Snapshot taken first:
`films_tensile_results_presnap_20260823`. `films_tensile_results` is now
3,510 rows.
