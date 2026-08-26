# Pass-filter roll extrusion lookup

Status: **blocked on Peter**, resolving the ID mismatches below. Started
26 August 2026.

## The task

For each of the 8 filtered tensile pass tables in
`gs://notpla-machine-data/claude/peter-files/tensile-exports/` (2 grades x
4 conditions: 50%/35% RH x MD/TD; see `260825_window_filter_methodology.txt`
in that folder for how the Core/Extended classification was derived),
produce a corresponding output table with one row per roll (Pellet ID +
Extrusion ID), six columns:

1. Pellet ID
2. Extrusion ID
3. Torque (`torque_percent` in `machine_collin_e25e.raw_films_extrusion`)
4. Die Pressure (`die_pressure_bar`)
5. Melt Temperature (`melt_temp_c`)
6. Core or Broad (from the source table's Window Classification: "Core
   Window (±1σ)" -> Core, "Extended Window (Broadened)" -> Broad)

Sort each output table by the trailing 4-digit number on Pellet ID,
ascending.

## Decisions confirmed with Peter (26 August 2026)

- All six columns above, kept separate (not merged into one "Roll" column).
- Where a roll has multiple extrusion database rows with genuinely
  different Torque/Die Pressure/Melt Temperature values, use the **mean**
  across all matching rows (consistent with the mean±SD convention already
  used in the source tables).
- Where a roll has **no exact match** in the extrusion table: paused,
  pending Peter resolving the mismatches below, rather than guessing or
  silently dropping rows.

## Why this is blocked

Cross-checking all 8 source tables against
`notpla-machine-data.machine_collin_e25e.raw_films_extrusion` found 36
distinct rolls across the 8 tables, of which **18 have no exact
(Pellet ID, Extrusion ID) match**.

**Finding 1 — likely swap.** `EV AB AL AM 260310 LI PF 1133` (tensile
table says extrusion `AO 260312 KD 1247`) and
`GN AB AE AM 260310 LI PF 1128` (tensile table says `AO 260312 KD 1248`)
look transposed: the extrusion database has `KD 1248` recorded against
pellet `...1133` and `KD 1247` against pellet `...1128`, the other way
round. Same date, same machine, adjacent roll numbers.

**Finding 2 — extrusion table coverage gap.** The extrusion database has
no rows at all for June 2026, and nothing after 2026-07-01. This likely
explains every June/July-dated miss below.

**Full list of the 18 unmatched rolls:**

| Pellet ID | Tensile table's Extrusion ID | Note | Appears in |
|---|---|---|---|
| `EV AB AI AM 251117 HZ PF 1019` | `BA 260324 KM 1255` | Pellet is on record as `BA 260324 KP 1311`, plus 4 unlabelled rows, same source file (`Backfill_260429.csv`) | EV RH35 MD, RH50 MD |
| `EV AB AI AM 251117 HZ PF 1019` | `BA 260324 KM 1256` | same as above | EV RH35 MD, RH50 MD |
| `EV AB AI AM 251117 HZ PF 1019` | `BA 260324 KM 1257` | same as above | EV RH35 MD, RH50 MD |
| `EV AB AL AM 260310 LI PF 1133` | `AO 260312 KD 1247` | See Finding 1, likely should be `KD 1248` | all 4 EV tables |
| `EV AB AL AM 260310 LI PF 1133` | `BA 260324 KM 1261` | Pellet not on record under this ID | all 4 EV tables |
| `EV AB AL AM 260310 LI PF 1133` | `BA 260324 KM 1279` | Pellet not on record under this ID | EV RH35 MD, RH50 MD |
| `GN AB AE AM 260310 LI PF 1128` | `AO 260312 KD 1248` | See Finding 1, likely should be `KD 1247` | all 4 GN tables |
| `GN AB AG AM 260505 ML PF 1201` | `AO 260602 LD 1358` | Pellet has a clean single match on 2026-05-06 as `AO 260506 KS 1326` instead (fits the sequence: sibling pellet `...1200` -> `KS 1325`); the June date looks wrong | all 4 GN tables |
| `EV AB AI AM 251117 HZ PF 1023` | `BD 260324 KM 1261` | Pellet absent entirely from extrusion table | all 4 EV tables |
| `EV AB AI AM 251117 HZ PF 1023` | `BD 260325 KN 1262` | Pellet absent entirely | EV RH35 MD, RH50 MD |
| `EV AB AI AM 251117 HZ PF 1026` | `BA 251216 JI 1139` | Pellet absent entirely | EV RH35 MD, RH50 MD |
| `EV AB AL AM 260310 LI PF 1136` | `BD 260324 KM 1260` | Pellet absent entirely | EV RH35 MD, RH50 MD |
| `EV AB AL AM 260310 LI PF 1136` | `BD 260325 KN 1267` | Pellet absent entirely | EV RH35 MD, RH50 MD |
| `EV AB AR AM 260714 OC PF 1294` | `AO 260715 MB 1410` | Post-cutoff date (Finding 2) | EV RH35 MD, RH50 MD |
| `EV AB AR AM 260714 OC PF 1294` | `AO 260715 MC 1411` | Post-cutoff date (Finding 2) | all 4 EV tables |
| `GN AB AE AM 260310 LI PF 1129` | `BD 260324 KO 1273` | Pellet absent entirely | GN RH35 MD, RH50 MD |
| `GN AB AE AM 260310 LI PF 1129` | `BD 260324 KO 1274` | Pellet absent entirely | all 4 GN tables |
| `GN AB AH AM 260615 NU PF 1248` | `AO 260618 LP 1373` | June date, coverage gap (Finding 2) | all 4 GN tables |

## Next step

Once Peter has resolved (or told me how to treat) the 18 rows above,
build the 8 output tables and save them alongside the source tables in
`gs://notpla-machine-data/claude/peter-files/`.
