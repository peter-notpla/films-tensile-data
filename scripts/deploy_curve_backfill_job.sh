#!/usr/bin/env bash
# Builds and deploys scripts/backfill_curve_points.py as a Cloud Run Job,
# so a long backlog drain runs on GCP infra instead of an interactive
# Cloud Shell VM. Mirrors scripts/deploy.sh's staging pattern: shared/*.py
# and the backfill script are copied into scripts/backfill-job/ (the
# build context) immediately before build, then removed after, since
# `gcloud run jobs deploy --source=DIR` only packages DIR itself.
#
# backfill_curve_points.py reads its pipeline config (WATCH_PREFIX,
# BQ_DATASET, etc.) from the environment, tensile defaulted - see that
# file's own docstring. This script deploys one Cloud Run Job per
# pipeline, each with that pipeline's env vars baked in via
# --set-env-vars, so "which backlog does this job drain" is fixed at
# deploy time and can't be mixed up at execute time.
#
# Usage: scripts/deploy_curve_backfill_job.sh tensile
#        scripts/deploy_curve_backfill_job.sh friction
# Then:  gcloud run jobs execute films-tensile-curve-backfill --region=europe-west2
#        gcloud run jobs execute films-friction-curve-backfill --region=europe-west2

set -euo pipefail

if [ $# -ne 1 ] || { [ "$1" != "tensile" ] && [ "$1" != "friction" ]; }; then
    echo "Usage: $0 <tensile|friction>" >&2
    exit 1
fi
PIPELINE="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_DIR="$REPO_ROOT/scripts/backfill-job"
REGION="europe-west2"

if [ "$PIPELINE" = "tensile" ]; then
    JOB_NAME="films-tensile-curve-backfill"
    SERVICE_ACCOUNT="sa-tensile-ingest@notpla-machine-data.iam.gserviceaccount.com"
    ENV_VARS="PIPELINE_NAME=tensile_raw"
    ENV_VARS="$ENV_VARS,WATCH_PREFIX=machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/tensiletester-films-tensile-raw-samples/"
    ENV_VARS="$ENV_VARS,PROCESSED_PREFIX=machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/tensiletester-films-tensile-raw-samples-processed/"
    ENV_VARS="$ENV_VARS,FAILED_PREFIX=machine-tensiletester-1/tensiletester-films/tensiletester-films-tensile/tensiletester-films-tensile-raw-samples-failed-processing/"
    ENV_VARS="$ENV_VARS,BQ_DATASET=films_tensile_london,BQ_TABLE=films_tensile_curve_points"
    ENV_VARS="$ENV_VARS,RESULTS_TABLE=notpla-machine-data.films_tensile_london.films_tensile_results_all_revisions"
else
    JOB_NAME="films-friction-curve-backfill"
    SERVICE_ACCOUNT="sa-friction-ingest@notpla-machine-data.iam.gserviceaccount.com"
    ENV_VARS="PIPELINE_NAME=friction_raw"
    ENV_VARS="$ENV_VARS,WATCH_PREFIX=machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/tensiletester-films-friction-to-be-processed/tensiletester-films-friction-to-be-processed-raw/"
    ENV_VARS="$ENV_VARS,PROCESSED_PREFIX=machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/tensiletester-films-friction-processed/tensiletester-films-friction-processed-raw/"
    ENV_VARS="$ENV_VARS,FAILED_PREFIX=machine-tensiletester-1/tensiletester-films/tensiletester-films-friction/tensiletester-films-friction-failed-processing/tensiletester-films-friction-failed-processing-raw/"
    ENV_VARS="$ENV_VARS,BQ_DATASET=machine_data,BQ_TABLE=films_friction_curve_points"
    ENV_VARS="$ENV_VARS,RESULTS_TABLE=notpla-machine-data.machine_data.films_friction_raw_all_revisions"
fi

cleanup() {
    rm -rf "$JOB_DIR/shared" "$JOB_DIR/scripts"
}
trap cleanup EXIT

cp -r "$REPO_ROOT/shared" "$JOB_DIR/shared"
mkdir -p "$JOB_DIR/scripts"
cp "$REPO_ROOT/scripts/backfill_curve_points.py" "$JOB_DIR/scripts/backfill_curve_points.py"

gcloud run jobs deploy "$JOB_NAME" \
    --source="$JOB_DIR" \
    --region="$REGION" \
    --service-account="$SERVICE_ACCOUNT" \
    --set-env-vars="$ENV_VARS" \
    --task-timeout=86400 \
    --max-retries=0 \
    --memory=1Gi
