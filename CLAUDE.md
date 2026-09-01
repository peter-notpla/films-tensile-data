# CLAUDE.md

Project context for Claude Code. This file is read automatically at the start
of every session in this repository.

---

## NEXT STEP (as at 1 September 2026): two blockers left, both need Peter's judgment

Both raw curve pipelines (tensile, friction) are **live and fully
backfilled** - Phase 5 checkpoint 1 is done. The Looker-facing analysis
views Peter asked for exist and are verified
(`films_tensile_curve_analysis`, `films_friction_curve_analysis` - see the
"Curve analysis views" section below). Several standing items were also
closed out same-session. Full blow-by-blow in `pipeline-roadmap.md`'s 1
September entries.

**Cleared once Peter approved directly** (the auto-mode classifier's block
turned out to be inconsistent under retry, not a strict wall, for this
class of command):
- **Friction Gmail alerts fixed.** All three Gmail secrets granted to
  `sa-friction-ingest`. Verified live with a genuine negative-path test
  (malformed file → real `FRICTION_RAW_FAILURE_ALERT_SENT` with a Gmail
  message ID, not just "no error").
- **Extrusion table whitespace trimmed.** 338 rows updated; verified 0/338
  now have leading/trailing whitespace on `pellet_id`/`extrusion_id`.
  Snapshot kept (`raw_films_extrusion_snapshot_20260901_pre_whitespace_trim`).
- **`template_name` backfilled** on `films_tensile_results_all_revisions`
  and `films_friction_raw_all_revisions` (both snapshotted first). Friction
  fully resolved; tensile resolved 3,459/3,510 - 51 rows across 23 files
  can't be recovered because those source CSVs no longer exist anywhere in
  GCS (left `NULL`, not guessed). Also surfaced a real parser gap: Excel's
  trailing-comma row-1 padding leaks into `template_name` for ~2,022
  tensile rows, fragmenting the same template into two distinct string
  values - not fixed yet, needs a `shared/tensile_parser.py` change. Full
  account in `pipeline-roadmap.md`'s 1 September "`template_name` backfill"
  entry.

**Still open:**
1. **One commit is unpushed: `b9c03cb` adds `.github/workflows/ci.yml`**,
   and GitHub rejects it from a token without `workflow` scope - this is a
   GitHub permission restriction, not an auto-mode block, so retrying
   won't help. Either get a token with `workflow` scope, push it
   yourself, or add the file by hand via the GitHub web UI (content is
   already in the local commit / see `pipeline-roadmap.md`'s "Tests and
   CI" entry).
2. **Curve-to-specimen link coverage is thin** (108 tensile specimens / 17
   pellets; only 2 friction specimens / 2 pellets currently link cleanly -
   see the "Curve analysis views" section below for the root cause). Not
   a mechanical fix - needs a decision on how to improve it.

**Also flagged, not attempted (needs Peter's judgment, not a blocker to
clear quickly)**: key rotation (`mecmesin-uploader`'s Jan 2026 key, the
appspot default account's `roles/editor`), and the dataset-naming
consolidation (`film_tensile_data`/`tensiletester_1`/`Rigid_Tensile`/
`Rigid_Tensile_euw2`) - both need Peter to scope and coordinate, not
something to execute unilaterally. "Talk to Callum" about revision-handling
value semantics is a human conversation, not automatable.

Standing habit, worth restating since it has now bitten this project three
times (28 August, 30 August, 1 September): **log each step in
`pipeline-roadmap.md` as it happens, and commit before ending a session,
even mid-task.**

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

### Template naming convention (lab workflow, not code)

When copying a VectorPro test to make a major edit, give the new template a
distinct name rather than reusing the old one. Row 1 of the exported CSV is
the template name, so a distinct name turns it into meaningful provenance
(which version of the test produced this file). This cannot repair history,
since old files already share names across template revisions, but it costs
nothing going forward. No code enforces this; it is a note for whoever runs
the tests.

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
pipelines/films-tensile-raw-processor/     deployed, live trigger active
pipelines/films-friction-raw-processor/    deployed, live trigger active
backfill/                                  one-off scripts, legacy
```

As of 21 August 2026 the repo matches what is deployed. Keep it that way:
commit and push in the same session as any deploy.

---

## Deploy and verify

**Use `scripts/deploy.sh`, never a raw `gcloud functions deploy`.** All three
pipelines import from `shared/`, and `gcloud functions deploy --source=<dir>`
only packages the directory it's pointed at - nothing outside it, including
`shared/`, is ever included. `scripts/deploy.sh` stages `shared/*.py` into
the pipeline directory first, then cleans it up after. A raw `gcloud`
deploy will build successfully but fail its container health check at
startup (`ModuleNotFoundError: No module named 'shared'`) - this happened
once already, 27 August 2026; Cloud Run correctly kept the prior healthy
revision serving 100% of traffic rather than routing to the broken one, so
it wasn't an outage, just a wasted deploy.

```bash
cd ~/films-tensile-data
scripts/deploy.sh <pipeline-dir> [function-name] [service-account]
```

`function-name` defaults to `pipeline-dir` (matches all three today).
`service-account` defaults to leaving the function's current one untouched;
pass it explicitly to change it (e.g. after rotating a least-privilege SA).

Always verify a deploy took effect by checking the logs for a distinctive
string from the new code, not by assuming, and confirm
`gcloud run services describe <name> --region=europe-west2` shows the new
revision actually serving traffic (`status.traffic`), not just that the
deploy command exited 0.

---

## Live alerting

As of 28 August 2026, all pipeline emails (weekly digest, per-file failure
alerts, repeat-failure escalation, extrusion's own immediate self-alert)
are sent directly from each pipeline's own code via the Gmail API as
peter@notpla.com, not via Cloud Monitoring alert policies. This replaced
the original alert-policy-based design after the weekly digest was found
to arrive with every field `null`: Cloud Monitoring's alert condition
aggregation (`crossSeriesReducer`) only preserves label values for its
`groupByFields`, silently dropping every other label. See
`pipeline-history.md`, 28 August 2026, for the full root cause and rebuild.

- `shared/gmail_sender.py`: sends HTML email via the Gmail API. Credentials
  (OAuth refresh token, client ID, client secret) live in Secret Manager as
  `pipeline-email-gmail-refresh-token` / `-client-id` / `-client-secret`,
  granted to `films-pipeline-digest-sa`, `films-pipeline-alerter-sa`, and
  `sa-extrusion-ingest`. The refresh token was obtained once by hand via a
  one-time OAuth consent flow; if it's ever revoked, that flow needs
  repeating (get a new OAuth client from Console, run the consent flow,
  overwrite the three secrets).
- `shared/email_style.py`: shared HTML building blocks matching Peter's
  Notpla Holiday Handover email design system (`#E8623A` orange headers,
  600px white card, Arial throughout).
- The 6 old Cloud Monitoring alert policies (weekly digest, Katie, Emily,
  default, escalation, extrusion) are **disabled, not deleted** - reversible
  if the direct-send approach ever needs rolling back. Their log-based
  metrics (`pipeline_weekly_digest`, `pipeline_failure_alert`,
  `pipeline_failure_escalation`, `extrusion_pipeline_failure`) are now
  orphaned (the code no longer emits the exact log-line formats they
  matched) but left in place, harmless.
- Extrusion's own immediate self-alert (in
  `pipelines/films-extrusion-csv-processor/main.py`, separate from the
  hourly manifest-based alerter) still always routes to peter@notpla.com.
  Failure alerts from the hourly alerter route to Katie, Emily, or
  peter@notpla.com by `user_initials`, same as before.

---

## Current state

Phase 0 (including 0.3, `backfill/backfill.py`'s date parsing, fixed by
pointing the script at `shared/tensile_parser.py` instead of maintaining
its own inline copy), Phase 1 (manifest table 1.1, row-errors table 1.2,
hourly first-sighting alert 1.3 with its UX/escalation/subject-line
follow-ons, and the Friday morning digest 1.4), Phase 2 (v2 architecture:
shared parsing library, key model, schema drift-check tooling, typed
friction columns, metadata revision handling, least-privilege service
accounts), Phase 3 (validation: whitespace, ID format checks, Excel
detection, extrusion cross-reference, template naming convention), and
Phase 4 (migration: all three pipelines now import their parser from
`shared/`) are all built and deployed as of 27 August 2026. See
`pipeline-roadmap.md` for the full phase-by-phase log and what's next, and
`pipeline-history.md` for build history predating that.

Email delivery confirmed by Peter (27 August 2026): the repeat-failure
escalation email, the alert subject-line fix, and the Friday digest (1.4)
all reached peter@notpla.com. No open loose ends remain from Phase 1.

All failed-processing folders are empty. Anything appearing in them is a live
problem.

### Table naming: `films_tensile_results` / `films_friction_raw` are views

As of 27 August 2026, `films_tensile_london.films_tensile_results` and
`machine_data.films_friction_raw` are **views**, not the underlying tables -
`SELECT * FROM ... WHERE row_state = "current"`, i.e. deduplicated per the
Phase 2.5 revision model. The actual tables both pipelines write to (and
where full history, including archived duplicate rows, lives) are
`films_tensile_results_all_revisions` and `films_friction_raw_all_revisions`
- reflected in each Cloud Function's `BQ_TABLE` env var. This was done
specifically so Looker Studio, which already points at the original names,
sees deduplicated data with zero reconfiguration. If you're querying either
table directly (not through Looker), query the `_all_revisions` name if you
need archived rows or want to reason about revision history; query the
plain name if you just want "the current data," same as Looker sees.

### Curve analysis views, for Looker: pick a pellet/extrusion ID, see curves

Built 1 September 2026 so Peter can add these as Looker Studio data
sources and filter curve charts by pellet or extrusion ID:

- `films_tensile_london.films_tensile_curve_analysis` - columns:
  `specimen_key`, `pellet_id`, `extrusion_id`, `test_direction`,
  `relative_humidity_pct`, `repeat_no`, `test_date`, `timestamp_start`,
  plus the curve columns (`row_number`, `time_s`, `load_n`,
  `displacement_mm`, `stress_mpa`, `strain_pct`), `link_time_delta_seconds`,
  `source_file`. `repeat_no` is tensile's `sample` field - there's no
  separate hand-entered repeat number for tensile the way friction has
  one, so this is the closest equivalent, not a verified match.
- `machine_data.films_friction_curve_analysis` - same shape, plus
  `test_surface`; no `test_direction` column, because friction's raw data
  has no direction field at all (checked the schema and real rows before
  concluding this, not assumed).

**Coverage is currently thin, especially for friction.** Each view keeps
only one curve file per specimen (fixed a real fan-out bug where up to 61
unrelated files were all linking to the same specimen - see
`pipeline-roadmap.md`'s 1 September entry for the full account). After
that fix: tensile has 108 specimens across 17 pellets / 20 extrusions;
friction has only 2 specimens / 2 pellets. Root cause is
`shared/curve_linking.py` matching by GCS upload time, which is a good
proxy for test time on a live-ingested file but not for most of the
historical backfill, where upload time reflects whenever the file
happened to reach GCS, not when the test happened. Not fixed - flagged for
Peter, since widening the match window would reintroduce the fan-out bug
just fixed, and a real fix likely needs either accepting that backfill
coverage stays thin (going-forward files linked via the live trigger
should do much better, since GCS time is test time for those) or a
different join key entirely.

---

## In progress: pass-filter roll extrusion lookup

Separate from the alerting pipeline work above. Building a per-roll
Torque/Die Pressure/Melt Temperature lookup against
`machine_collin_e25e.raw_films_extrusion` for the 8 filtered tensile pass
tables in `gs://notpla-machine-data/claude/peter-files/tensile-exports/`.

**Blocked on Peter as of 26 August 2026**: 18 of 36 rolls across those
tables have no exact match in the extrusion table, including one likely
ID swap between two pellets and a June-2026 coverage gap in the extrusion
table. Full findings, confirmed output format, and the exact list of rolls
needing resolution are in `pass-filter-extrusion-lookup.md`. Once resolved,
build the 8 output tables per that file's spec.

Genuine backlogs, deliberately untouched:
- Tensile and friction raw curve backlogs are both resolved: Phase 5
  checkpoint 1 backfilled tensile (1,243/1,244) and friction (928/928),
  both verified clean against GCS and BigQuery. See the NEXT STEP section
  above and `pipeline-roadmap.md`.

**`notpla-machine-data` is a hierarchical-namespace bucket, so object
versioning cannot be enabled (GCS does not support it on HNS buckets).**
A soft-delete policy has been active since bucket creation instead: 7-day
retention, so deletes and overwrites are recoverable for 7 days via
`gcloud storage objects restore`, not permanent. Confirmed 27 August 2026.
