# CLAUDE.md

Project context for Claude Code. This file is read automatically at the start
of every session in this repository.

---

## DONE (7 September 2026): Friction Curves legend/drill-down parity with Tensile Curves

Copied Tensile Curves' `Curve Detail Level` drop-down + `Curve Breakdown
Label` calculated-field pattern onto `films_friction_curve_analysis`
(built separately since Looker Studio parameters are scoped per data
source), replacing the raw `specimen_key` legend. Four modes: Pellet ID /
Extrusion ID / Test Surface (all mean curves, the last one new - friction
has no direction field to mirror tensile's, so test surface was the
natural addition) / Sample (individual curves). Chart's breakdown
dimension and Y-metric aggregation (Sum -> Average) updated to match.
Verified live via the actual control, all four modes. Full account,
including a stuck-editor-state scare during verification that a `Reset`
click cleared (nothing wrong with the new fields), in
`pipeline-roadmap.md`'s matching entry.

## DONE (7 September 2026): friction pellet/extrusion ID anomaly scan

Same kind of scan as the 5 September tensile one, applied to friction's
manually-entered `pellet_id`/`extrusion_id`. Full pairwise typo check
(Levenshtein <=2 across every distinct value, not a sample) found no real
fat-finger typos - every near-match is two genuinely different,
sequentially-numbered rolls/bags with comparable test counts. Real finds,
both fixed (snapshotted first): 5 rows with an invisible leading/trailing
space that had silently fragmented 3 real pellets and 2 real extrusions
into duplicate-looking entries (trimmed, e.g. one pellet went from
looking like 85+23 back to a single 108); and 2 genuine junk rows (`"x"`/
`"callum"`, notes "hi"/"Callum") sitting inside an otherwise-legitimate
~50-row summary file from 27 January 2026 - removed (BigQuery rows +
linked curve_points rows + the 2 GCS raw curve files), source summary CSV
left untouched since the rest of it is real data. Full account in
`pipeline-roadmap.md`'s matching entry.

## DONE (7 September 2026): curve-linking coverage, 30%/42% -> 92%/99.9%

Peter asked to solve curve-linking coverage properly: link as many existing
raw curve files as possible, permanently, and make new files link
immediately going forward. Full account in `pipeline-roadmap.md`'s matching
7 September Phase 5 entry; summary here.

- Added a third linking tier to `shared/curve_linking.py`,
  `find_specimen_link_by_mapped_sample`, backed by a new permanent
  `films_tensile_london.sample_number_map` table
  (`scripts/build_sample_number_map.py`) built by joining each instrument's
  archived pre-renumbering export to the live table on measured values, not
  arithmetic. Real result: friction resolved cleanly (497/522), tensile
  mostly didn't (570/711 ambiguous - confirmed the same original sample
  number attaches to genuinely different physical tests even within the
  archive file itself, not just a rare edge case). Ambiguous mappings are
  written `is_ambiguous = TRUE` and never linked, Peter's explicit call.
- Found and finished a second tier that already existed uncommitted and
  undocumented in the working tree, live in production for friction only
  since 4 September (`find_specimen_link_by_sample`) - the third instance of
  this project's "session work done but never logged" pattern. Fixed a real
  bug (`TRIM(sample)` failed outright on tensile's INT64 `sample` column)
  and extended it to tensile, which had never used it.
- Found and fixed a friction template-naming bug while investigating why
  direct sample matching recovered nothing: raw curve filenames for the
  renamed `FrictionTest-FilmsOld(V1)` template were never updated from
  "Films", the pre-rename name. Confirmed safe to bridge (the two
  templates' sample-number ranges are completely disjoint) before adding
  `TEMPLATE_ALIASES`.
- These two fixes, not the reconciliation table, did nearly all of the real
  work: tensile 372 -> 1,138 of 1,242 linked (92%), friction 387 -> 927 of
  928 (99.9%). Looker-facing coverage on the curve_analysis views: tensile
  108 -> 865 specimens (17 -> 34 pellets), friction 2 -> 820 specimens (2
  -> 29 pellets).
- Full verification discipline applied: replayed the parser against all
  1,242 real tensile files, extended `shared/verify_curve_parser.py` with
  real-data checks for all three tiers, snapshotted both curve_points
  tables before a single `MERGE` backfill each, then a live end-to-end test
  through both real GCS watch folders - which caught a genuine bug no
  offline check had (a numpy.int64 passed into a BigQuery INT64 parameter,
  only reachable when a file falls through to the third tier). Fixed,
  redeployed, re-verified clean before calling it done.
- **Checked in the browser afterward that Looker Studio was actually
  showing the new coverage, and found a real second bug**: Friction Curves
  rendered as scattered dots (raw `specimen_key` legend too), the same
  "dots not curves" categorical-axis bug fixed on Tensile Curves 5
  September, never applied to Friction Curves because it only had 2
  specimens then to expose it. Fixed the same way: added a
  `time_s_binned = ROUND(time_s/0.5,0)*0.5` calculated field on
  `films_friction_curve_analysis` and pointed the chart's X-axis at it.
  Verified live - genuine connected oscillating stick-slip curves now
  render, filtered and unfiltered. Tensile Curves' own data source/field
  bindings were checked too and were already correct. Full account in
  `pipeline-roadmap.md`'s matching entry.

---

## DONE (5 September 2026, third session): tensile data anomaly scan (case/garbage values)

Peter noticed apparent duplicate "9" options in the Repeat Number filter
and asked for a scan of `test_direction`, `sample_number`, `pellet_id`,
`extrusion_id`, `relative_humidity_pct` for trailing spaces or other
obviously-wrong values on `films_tensile_results_all_revisions`.

Snapshotted the table first
(`..._snapshot_20260905_pre_direction_repeat_cleanup`), then fixed two
real anomalies: `test_direction = 'md'` (lowercase, 2 rows) -> `MD`, and
`sample_number = '.'` (garbage, 2 rows) -> `''` (matching the existing
blank convention rather than guessing a number). One row with
`sample_number = '21-2'` is flagged but left alone - not clear whether it
means 2, 21, or something else. `pellet_id`/`extrusion_id` had zero
anomalies; `relative_humidity_pct` only real target values. Could not
reproduce the "duplicate 9" Peter saw in the underlying data at all -
likely a stale Looker filter-control cache from today's earlier
`repeat_no` type change (INT64 -> STRING), not a real data problem.

Full method and query results in `pipeline-roadmap.md`'s matching 5
September entry.

---

## DONE (5 September 2026, second session): fixed `Repeat No.` and the remaining dotted curves on Tensile Curves

Peter reported the same-day polish above hadn't fully landed: `Repeat No.`
was mapping to VectorPro's unstable `sample` counter (hundreds/thousands,
not 1-5), and several curves were still rendering as scattered dots.

- **`Repeat No.` fix**: `films_tensile_curve_analysis` was built (1
  September) on the stated assumption tensile has no hand-entered repeat
  field like friction's - wrong. `shared/tensile_parser.py` parses
  `sample_number` from the CSV's "Sample Number (Prompt For Value - Before
  Test)" column, a real per-test hand-entered value, distinct from the
  `sample` auto-counter, and it's exactly what the original "Tensile"
  page's own `Repeat Number` filter already uses (confirmed live before
  touching anything). Changed the view's `repeat_no` column from `r.sample`
  to `r.sample_number` (`CREATE OR REPLACE VIEW`, old SQL saved first),
  clicked **Refresh Fields** on the data source so the STRING type change
  took effect. Verified: filter values now read a clean 1-6.
- **Dots root cause, different from the same-day fix above**: the earlier
  fix rounded `displacement_mm`/`strain_pct` to 1 decimal to collapse the
  category axis. That resolution is too fine for the real data -
  `films_tensile_curve_points` is downsampled to roughly one point pair
  per 0.6s, which at typical crosshead speed skips 0.1mm bins
  unpredictably, leaving genuine gaps (1,000+ gaps over 1.5x bin width,
  confirmed by SQL). Widened the two calculated fields to
  `ROUND(displacement_mm/0.5,0)*0.5` and `ROUND(strain_pct/0.75,0)*0.75` -
  0 gaps at those bin sizes. Verified on both charts at all three Curve
  Detail Level settings; curves are now fully continuous.
- **Curve Detail Level dropdown**: tested directly, found no fault -
  correct exclusive selection, correctly drives both charts, resets to
  default correctly. Most likely the two bugs above were making the whole
  chart look broken regardless of the level selected. Worth knowing: the
  dropdown's click sometimes silently fails to register a selection
  (Looker Studio UI quirk) - confirm the header label text changed before
  closing it.

Full account, including the SQL gap analysis and exact bin-size
measurements, in `pipeline-roadmap.md`'s matching 5 September entry under
Phase 5.

---

## DONE (5 September 2026): Tensile Curves polish + Pipeline Health rebuilt as scorecards, via browser automation

Peter asked for four things on the Tensile Curves page and one on Pipeline
Health, then walked away and asked for autonomous completion (no further
questions). All done directly in Looker Studio via `claude-in-chrome`:

- **Axis titles**: both curve charts now show real axis titles - "Displacement
  (mm)" / "Load (N)" and "Strain (%)" / "Stress (MPa)" - by renaming the
  underlying fields on `films_tensile_curve_analysis` (was previously just
  the raw field name, and the axis-title toggle was off on one chart).
- **Dots-not-curves root cause found and fixed**: the charts were rendering
  as scattered points because each specimen's raw `displacement_mm`/
  `strain_pct` values are essentially unique floats, so the shared category
  axis had almost no overlap between series - Looker Studio draws isolated
  markers, not connected segments, when a series has data at only a handful
  of the axis's thousands of categories. Fixed by adding two calculated
  fields, `displacement_mm` and `strain_pct` **rounded to 1 decimal place**,
  used as the chart X-axes instead of the raw columns. This collapses the
  category count enough that curves render as continuous connected lines,
  and is also what makes the mean-curve aggregation below work.
- **Legend readability**: added a `Curve Breakdown Label` calculated field
  (CASE on the new drill-level parameter, see below) used as the Breakdown
  Dimension instead of raw `specimen_key` - legend now shows short codes
  like "EV AB AI AM 251117 HZ PF 1019" instead of
  "tensiletester-1|tensile|...".
- **Drill-down (Pellet ID mean -> Extrusion ID mean -> individual sample)**:
  Looker Studio's native multi-field "Drill down" feature (used on the
  Tensile page's bar chart, e.g. Pellet ID -> Extrusion ID -> ... for
  ranking scalar properties) does not work for a continuous XY curve chart -
  the drill-down field list is a categorical axis feature, not available on
  Breakdown Dimension, and X-axis has to stay the physical variable. Built
  the closest working equivalent instead: a report **Parameter**
  (`Curve Detail Level`, values `PELLET`/`EXTRUSION`/`SAMPLE`, default
  `PELLET`) driving a drop-down control next to the existing filters, plus
  a `Curve Breakdown Label` calculated field that returns `pellet_id`,
  `extrusion_id`, or a `RH x% | Direction | #SampleNo` string depending on
  the parameter, used as Breakdown Dimension with the Y metric aggregation
  set to **Average**. Result: with no extra selection the chart shows one
  mean curve per pellet; switching the new drop-down descends to
  per-extrusion mean curves, then to individual-sample curves (RH/Direction/
  Sample No. combination) - verified all three levels against real data.
  Flagging in case Peter would prefer the literal double-click drill
  interaction instead - technically not available for this chart type.
- **Pipeline Health rebuilt as scorecards, not tables**: replaced the
  `films_pipeline_summary` table with 5 individual bordered boxes (one per
  pipeline: extrusion, friction, friction_raw, tensile, tensile_raw) in the
  same black-border/white-background style as the "Total Tests" box on
  other pages, each a Scorecard on `open_issue_count` filtered to that
  pipeline (verified against BigQuery: 3/3/1/2/2 = 11 total, matches the
  known count). Added the Notpla header/title for consistency with other
  pages. Left `films_pipeline_open_issues` as a table below ("Open Issues
  (detail)") since it's a variable-length list of individual failures with
  free-text error messages, not fixed per-pipeline properties - flagging
  this judgment call in case Peter wants it converted too. Not added:
  `files_processed_ok` / `resolved_issue_count` as a second number per
  card - the Scorecard "Optional metric" field didn't render visibly in
  this Looker Studio version, and time didn't allow building a second
  scorecard row per pipeline; flagging as a follow-up if wanted.
- **Automation gotcha worth knowing if touching Looker Studio filters
  again**: the free-text "value" box in Looker Studio's filter editor
  looks like a plain text input but is actually an autocomplete combobox -
  typing a value and clicking a suggestion with the mouse does NOT commit
  it (Save enables, but the stored filter value is silently empty). The
  value only commits into a proper chip if you select it via keyboard
  (type, then `Down` arrow to highlight the match, then `Enter`). Cost
  significant back-and-forth this session before finding this; several
  scorecards briefly showed "No data" for exactly this reason.

## DONE (4 September 2026): Tensile/Friction Curves pages were broken, now fixed and Tensile split into 3 metrics

Peter reported the Tensile Curves and Friction Curves pages built earlier
the same day "do not work." Root cause on both: the line chart's Sort was
set to the metric descending instead of `time_s` ascending, and every
breakdown series past the first defaulted to "Bars" style instead of
"Line" (Looker Studio has no bulk-apply for per-series style, so each of
the 20 series slots had to be clicked individually) - together these
turned real curves into a meaningless sawtooth of disconnected spikes.
Fixed on both pages: sort to `time_s` ascending, all series set to Line,
point/series caps raised to their max (5000 points, 20 series), "Group
the rest as Others" turned off. Verified by selecting real pellets and
confirming genuine rising (tensile) / oscillating stick-slip (friction)
curve shapes.

Tensile Curves also split into three stacked line charts - "Load (N) vs
Time", "Stress (MPa) vs Time", "Strain (%) vs Time" - plus a small
`films_tensile_results`-sourced table for Young's Modulus (MPa), since
modulus is a scalar per specimen, not a value that varies over time, so
it can't be a fourth curve line. Friction Curves was left as the single
`load_n` chart - its schema carries unused stress/strain columns
inherited from the shared curve-parser output, but they're not
physically meaningful for a friction test. Full account, including the
exact broken settings found, in `pipeline-roadmap.md`'s Phase 5 entry
dated the same day.

---

## DONE (4 September 2026): all 3 Looker pages built via browser automation

The `claude-in-chrome` blocker from earlier the same day resolved itself -
the tools showed up as loadable in a later session with no further action
needed (never root-caused beyond that; if it recurs, confirm the extension
is installed at https://claude.ai/chrome, re-run `/chrome`, then search for
the browser tools again before assuming they're missing). All 3 Looker
Studio pages Peter wanted built directly are now live on the Films
Dashboard report:

- **Pipeline Health** tab - rebuilt from scratch. It had drifted onto the
  old `films_pipeline_manifest` table (plus a leftover stale instructional
  text box from an earlier session, referencing that same wrong table) -
  both deleted. Rebuilt against `films_pipeline_open_issues` /
  `films_pipeline_summary` per the roadmap's Phase 1.5 recipe: a
  `films_pipeline_summary` table sorted by `last_seen_at` ascending
  (surfaces silence fastest) and a `films_pipeline_open_issues` table
  filtered to `is_open = TRUE`, sorted by `first_failed_at` ascending.
  Verified against the roadmap's known count: 11 open issues, matching.
- **Tensile Curves** tab (new) - duplicated from the existing "Tensile"
  page to inherit its Notpla logo/header/scorecard styling exactly, then
  gutted (removed the mechanical-properties filters, results tables, and
  bar-chart "Graphs" section) and rebuilt per the roadmap's "Two new
  Looker pages" recipe: 6 multi-select filter controls (Pellet ID,
  Extrusion ID, Test Date, Relative Humidity, Test Direction, Repeat No.)
  bound to `films_tensile_curve_analysis`, plus a line chart -
  dimension `time_s`, metric `load_n`, **breakdown dimension
  `specimen_key`** (this is what makes overlay work: selecting several
  specimens in the filters draws one line per specimen automatically).
- **Friction Curves** tab (new) - same pattern against
  `films_friction_curve_analysis`, with `test_surface` swapped in for
  `test_direction` per the roadmap.

**Gotcha hit repeatedly, worth knowing if touching these pages again**:
Looker Studio's data-source-switch auto-remaps a filter control's bound
field by matching order/type, not name - it silently mapped "Repeat
Number" to `specimen_key` and "Relative Humidity" to `extrusion_id` more
than once. After changing any filter's data source, always check the
Control field, don't trust the auto-pick.

**Not done**: the roadmap's optional legend table under each curve chart
(listing `specimen_key`/`pellet_id`/`extrusion_id`/`test_date`/`repeat_no`
for whatever's currently filtered) - Peter can ask for it if wanted.

---

## PRIORITY FOR NEXT SESSION (1 September 2026): scope has drifted, check against the original plan first

Before picking up new work, compare against `project-briefing.md` §8 (the
agreed Phase 0-6 plan). Several sessions' worth of work has gone to things
adjacent to that plan rather than the plan itself. Read this before adding
anything else.

**Extra scope added, not in the original six phases:**
- The pass-filter roll extrusion lookup (`pass-filter-extrusion-lookup.md`)
  - a full separate workstream, not mentioned anywhere in the original
  briefing or roadmap. Stalled at 18/36 rolls unresolved, blocked on Peter.
- The curve-to-specimen analysis views (`films_tensile_curve_analysis`,
  `films_friction_curve_analysis`) and the linking-quality work behind
  them. Phase 5 as scoped stopped at "build the raw curve pipelines"
  (checkpoint 1); everything downstream of that, including this, was
  invented mid-session.
- The `template_name` NULL bug and its fix (parser fix, live redeploy,
  2,046-row re-normalization) - found only while chasing the curve-linking
  work above. Not anticipated anywhere.
- Alert delivery was rebuilt twice, neither time as originally decided.
  The plan named Google Apps Script. What got built was Cloud Monitoring
  alert policies first (found broken, the `crossSeriesReducer` bug), then
  replaced with direct Gmail API sends. Apps Script was never used.
- Tests and CI: flagged in the original briefing's "standing items not yet
  scheduled" (not one of the six phases), started this session on its own
  initiative rather than by request.

**Remains from the original plan, still genuinely open:**
- ~~**Phase 1.5, the Looker pipeline health page.**~~ Done 4 September
  2026: `films_pipeline_open_issues` / `films_pipeline_summary` views in
  `films_pipeline_ops`, `resolved_at` included, and the Looker page itself
  built and live the same day - see `pipeline-roadmap.md`'s Phase 1.5
  entry.
- ~~**Phase 2.4, typed columns.**~~ Corrected 4 September 2026: friction's
  `_num` siblings promoted to the live `films_friction_raw` view under
  their original names. Needs a one-time "Refresh Fields" click in Looker
  Studio on that data source - see the roadmap's Phase 2.4 entry.
- **Phase 5, everything past checkpoint 1.** ~~Two new Looker pages
  (tensile/friction curve browsers, filter + overlay).~~ Built and live 4
  September 2026 - see the roadmap's "Two new Looker pages" entry.
  ~~Curve-to-specimen link coverage.~~ Solved 7 September 2026 via two
  additional non-time-based linking tiers - the GCS-time signal itself is
  still destroyed for historical files exactly as found 4 September (that
  specific method really can't be improved), but sample-number-based
  matching doesn't need it. See the "DONE (7 September 2026)" entry at the
  top of this file.
- **Phase 6, in full.** `films_results_long` and its dedup rule (6.1, 6.2)
  were never built. This is the actual "pick a Pellet ID, see every test
  on that roll" deliverable - the curve views above are adjacent, not a
  substitute. **Needs a decision: build it, formally drop it, or keep
  letting adjacent work substitute for it.**
- Standing items from the original briefing §9, still open: key rotation
  (`mecmesin-uploader`'s Jan 2026 key, the appspot default account's
  `roles/editor`), the four-dataset naming consolidation, and the 5
  unmatched friction rows with ~5.0 static CoF flagged as a possible
  calibration fault, never investigated further.
- "Talk to Callum" about his `tensile_v21_*` pattern - open since the
  first briefing.
- The end-user manual, explicitly meant to be written last, once the
  system stopped changing. Never started, and "last" keeps moving.
- `README.md` is stale (still lists `films-friction-raw-processor` as "not
  deployed"; it's been live since Phase 5 checkpoint 1).

Fastest way to close the gap: land the blocked push below, then get a real
decision on Phase 6 scope before starting anything else adjacent to it.

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
- **`template_name` backfilled and normalized** on
  `films_tensile_results_all_revisions` and `films_friction_raw_all_revisions`
  (snapshotted before each write). Friction fully resolved; tensile
  resolved 3,459/3,510 - 51 rows across 23 files can't be recovered because
  those source CSVs no longer exist anywhere in GCS (left `NULL`, not
  guessed). Also fixed the parser gap this surfaced: Excel's trailing-comma
  row-1 padding was leaking into `template_name`
  (`shared/excel_detection.clean_template_name()` now handles it, both
  `shared/tensile_parser.py` and `shared/friction_parser.py` use it).
  Deployed to both `films-tensile-csv-processor` and
  `films-friction-csv-processor`, verified live with a real synthetic file
  through each GCS watch folder, then re-normalized the 2,022 + 24 already-
  affected historical rows. Full account in `pipeline-roadmap.md`'s 1
  September `template_name` entries.

**Still open:**
1. **One commit is unpushed: `b9c03cb` adds `.github/workflows/ci.yml`**,
   and GitHub rejects it from a token without `workflow` scope - this is a
   GitHub permission restriction, not an auto-mode block, so retrying
   won't help. Either get a token with `workflow` scope, push it
   yourself, or add the file by hand via the GitHub web UI (content is
   already in the local commit / see `pipeline-roadmap.md`'s "Tests and
   CI" entry).
2. ~~**Curve-to-specimen link coverage is thin.**~~ Solved 7 September
   2026: tensile 865 specimens / 34 pellets, friction 820 specimens / 29
   pellets - see the "DONE (7 September 2026)" entry at the top of this
   file and the "Curve analysis views" section below.

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

Checked live 4 September 2026: both failed-processing folders are empty.
The one file that was there (`raw-FILMS-CYCLICALLOADING(V1)-sample-1.csv`,
root-caused as a genuinely empty capture from a March 2026 `"TEST"`-labelled
calibration run, not a parser bug) has since been removed at Peter's
request along with the rest of that template's data - see
`pipeline-roadmap.md`'s "Failed raw curve check" entry under Phase 5 for
the full account, including a from-scratch pipeline health audit across
all 5 pipelines that confirmed everything else is green. Anything
appearing in either folder from here is a live problem.

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

Each view keeps only one curve file per specimen (fixed a real fan-out bug
where up to 61 unrelated files were all linking to the same specimen - see
`pipeline-roadmap.md`'s 1 September entry for the full account).

**Coverage as of 7 September 2026**: tensile 865 specimens across 34
pellets, friction 820 specimens across 29 pellets - up from 108/17 and 2/2
on 1 September. `shared/curve_linking.py` now has three linking tiers
(GCS-time proximity, direct sample-number match, and a mapped-sample match
via the permanent `films_tensile_london.sample_number_map` table), not just
the original GCS-upload-time match, which was never a good proxy for most
of the historical backfill (upload time there reflects whenever the file
happened to reach GCS, not when the test happened). Full account, including
the real friction template-naming bug this surfaced and the genuine
remaining gaps (tensile 104 files / friction 1 file, both structurally
unrecoverable, not a shortcoming of the method), in `pipeline-roadmap.md`'s
7 September entry.

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
