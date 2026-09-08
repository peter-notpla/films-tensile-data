# Pass-filter roll extrusion + SKU lookup

Status: **done, 8 September 2026** - both the extrusion lookup (originally
blocked here since 26 August) and a new SKU/formulation traceability lookup,
built together in the same session. Superseded the exact August spec (see
"How this differs from the original spec" below) - Peter gave a revised,
simpler column set that sidestepped the original blocker.

**Updated later the same day (8 September 2026)**: a colleague advised
broadening the Extended window further (see
`~/tensile-exports/260825_window_filter_methodology.txt` Section 3a -
EV AB's lower Strength bound and both grades' lower Modulus bound moved).
Re-filtering against the broadened bounds added 8 new EV AB rolls and 1
new GN AB roll (6 new distinct Pellet IDs: `EV AB AC AL 250812 HV PF 0863`,
`...AE AM 250815 HZ PF 0868`, `...AI AM 251111 HZ PF 0987`,
`...AP AM 260513 MQ PF 1211`, `...AS AM 260721 OD PF 1296`,
`GN AB AH AM 260616 NV PF 1256`), all classified Extended Window
(Broadened). The extrusion and SKU lookups were extended for these 6
new pellets using the same join logic below - all counts in this doc
updated accordingly (32 distinct Pellet IDs total, up from 26; 8
unrecoverable, up from 6 - 2 of the 6 new pellets have zero
`raw_films_extrusion` rows).

## What was built

16 CSVs (8 extrusion + 8 SKU, one pair per grade x RH x direction condition),
saved alongside the source tensile pass tables in both
`~/tensile_final/` and
`gs://notpla-machine-data/claude/peter-files/tensile-exports/`, prefixed
`260908_` (vs. the source tables' `260825_`), e.g.
`260908_EV_AB_RH50_MD_extrusion.csv` / `260908_EV_AB_RH50_MD_sku.csv`.

**Extrusion tables** - one row per pass Pellet ID (not per roll/Extrusion ID -
see below), sorted by trailing 4-digit pellet number ascending:

1. Pellet ID
2. Torque (%) - mean ± sample SD across all `machine_collin_e25e.raw_films_extrusion`
   rows matching that Pellet ID
3. Die Pressure (bar) - same
4. Melt Temperature (C) - same

Where a pellet has exactly one matching row, SD is shown as `N/A`. Where a
pellet has zero matching rows, the row is still included with blank T/P/M
cells (Peter's explicit choice, 8 September).

**SKU tables** - one row per ingredient (SKU) in that grade's formulation,
sorted by SKU code ascending:

1. SKU Code
2. Real Name (`ingredients.trade_name_inci`)
3. Concentration (%) (`wt_percent` from `v_formulations_flat.dry_weight_items`,
   constant across all batch-variant codes for a given grade - confirmed by
   direct query before building)
4. one column per pass Pellet ID in that condition, cell = that pellet's
   **ingredient batch/lot code** (`ingredient_batch_code` from
   `notpla-rnd-tracker.formulation_app_eu.batch_variant_items`) for that SKU -
   not a repeated SKU code. Peter's explicit spec (8 September): "include
   the SKU code in one column, and then in the rows for each sku code I need
   the batch code under a given pellet ID."

## How this differs from the original spec (26 August)

The original spec (Extrusion ID as its own column, join key = Pellet ID +
Extrusion ID together, Core/Broad classification column) is what caused the
18/36-unmatched block. Peter's 8 September ask dropped Extrusion ID and the
classification column entirely and asked for Pellet ID only. This matters:

- **Join key is now Pellet ID alone.** `raw_films_extrusion` has its own
  `pellet_id` column, so joining on Pellet ID alone (ignoring the tensile
  table's separately-recorded Extrusion ID field) recovered most of what was
  previously blocked, including confirming the suspected ID swap (26 August
  Finding 1: pellet `...1133` truly matches extrusion `KD 1248`, pellet
  `GN...1128` truly matches `KD 1247` - both now visible directly under
  Pellet ID with no ambiguity).
- **Genuinely unrecoverable: 8 of 32 distinct pass Pellet IDs** (originally
  6 of 26; the window-broadening update above added `EV AB AS AM 260721 OD
  PF 1296` and `GN AB AH AM 260616 NV PF 1256` to this list, both with zero
  `raw_films_extrusion` rows under any Extrusion ID) - not 18 of 36 rolls,
  the roll-level count doesn't apply once Extrusion ID is dropped from the
  output. Full list: `EV AB AI AM 251117 HZ PF 1023`, `...PF 1026`,
  `EV AB AL AM 260310 LI PF 1136`, `EV AB AR AM 260714 OC PF 1294`,
  `EV AB AS AM 260721 OD PF 1296`, `GN AB AE AM 260310 LI PF 1129`,
  `GN AB AH AM 260615 NU PF 1248`, `GN AB AH AM 260616 NV PF 1256`.
  Consistent with the 26 August finding of a real June/July 2026 coverage
  gap in the extrusion table plus a few pellets never logged there at all -
  not a matching bug, confirmed again this session by joining on Pellet ID
  alone rather than the (Pellet ID, Extrusion ID) pair.
- **32 distinct Pellet IDs, not 36 rolls** (originally 26 of 36 rolls,
  before the window-broadening update added 6 more distinct Pellet IDs).
  The August count of 36 was rolls (Pellet ID + Extrusion ID pairs);
  several Pellet IDs have more than one Extrusion ID recorded against them
  in the source tensile tables. Since the output is one row per Pellet ID,
  the source CSVs' Pellet ID column had to be de-duplicated before building
  the pellet lists (caught during verification - the first build attempt
  had duplicate pellet rows/columns from not deduping, fixed before this
  was written up).

## SKU/formulation data model (for future reference)

- `notpla-rnd-tracker.formulation_app_eu` needs `notpla-machine-data` (or
  another project you have `bigquery.jobs.create` in) as the billing
  project for `bq query` - direct `--project_id=notpla-rnd-tracker` queries
  get an Access Denied on job creation even though the dataset itself is
  listable.
- A Pellet ID's first three space-separated tokens (e.g. `EV AB AL`) are
  exactly `pellet_bags.pellet_bag_code`'s first three tokens and map to
  `v_formulations_flat`'s `set_code`, `weight_code`, `batch_variant_code`.
- Concentration (`dry_weight_items`/`wt_percent`) is keyed at
  `(set_code, weight_code)` only - i.e. per grade (`EV AB` has 7 SKUs, `GN AB`
  has 8), constant across every `batch_variant_code`. Confirmed directly
  before relying on it (checked distinct `dry_weight_items` per grade = 1).
- The actual ingredient batch/lot code used is keyed at
  `(set_code, weight_code, batch_variant_code, sku)` in `batch_variant_items` -
  this is per compounding batch, not per pellet bag, so sibling bags from the
  same batch (same first-three-token prefix) share identical batch codes.
  All 26 of our pass pellets had a full set of `batch_variant_items` records
  (no gaps here, unlike the extrusion table).
- `ingredient_batch_code` is sometimes the literal string `UNKNOWN` (real
  recorded value, not a null/missing marker) or free text like "No batch no.
  on sample" - left as-is in the output, not treated as blank.
