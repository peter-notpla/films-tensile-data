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
pipelines/films-friction-raw-processor/    NOT deployed
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
- 738 friction raw files failed, no processor deployed
- 197 friction raw files queued
- 1,244 tensile raw sample files, no processor watches that prefix

**`notpla-machine-data` is a hierarchical-namespace bucket, so object
versioning cannot be enabled (GCS does not support it on HNS buckets).**
A soft-delete policy has been active since bucket creation instead: 7-day
retention, so deletes and overwrites are recoverable for 7 days via
`gcloud storage objects restore`, not permanent. Confirmed 27 August 2026.
