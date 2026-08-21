# films-tensile-data

Data pipelines for Notpla's lab instruments. CSV exports from lab testing
software land in Google Cloud Storage, are parsed by Cloud Functions, and are
loaded into BigQuery for reporting in Looker Studio.

GCP project `notpla-machine-data`, region `europe-west2`.

## Pipelines

| Folder | Function | BigQuery table |
|---|---|---|
| `pipelines/films-tensile-csv-processor` | `films-tensile-csv-processor` | `films_tensile_london.films_tensile_results` |
| `pipelines/films-friction-csv-processor` | `films-friction-csv-processor` | `machine_data.films_friction_raw` |
| `pipelines/films-extrusion-csv-processor` | `films-extrusion-csv-processor` | `machine_collin_e25e.raw_films_extrusion` |
| `pipelines/films-friction-raw-processor` | not deployed | not created |

## Documentation

- `CLAUDE.md` project context, read automatically by Claude Code
- `project-briefing.md` current state, decisions and their evidence
- `pipeline-roadmap.md` the forward plan, phases 0 to 6
- `pipeline-review.md` design assessment (note: overstates data loss as 26%, true figure nearer 1.5%)

## Deploying

cd pipelines/<pipeline>
python3 -m py_compile main.py
gcloud functions deploy <function-name> --region=europe-west2 --gen2 --source=. --quiet


Commit and push in the same session as any deploy. This repo diverged from
production for three months because a failed `git push` went unnoticed.
