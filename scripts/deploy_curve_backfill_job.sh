#!/usr/bin/env bash
# Builds and deploys scripts/backfill_curve_points.py as a Cloud Run Job,
# so a long backlog drain runs on GCP infra instead of an interactive
# Cloud Shell VM. Mirrors scripts/deploy.sh's staging pattern: shared/*.py
# and the backfill script are copied into scripts/backfill-job/ (the
# build context) immediately before build, then removed after, since
# `gcloud run jobs deploy --source=DIR` only packages DIR itself.
#
# Usage: scripts/deploy_curve_backfill_job.sh
# Then:  gcloud run jobs execute films-tensile-curve-backfill --region=europe-west2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOB_DIR="$REPO_ROOT/scripts/backfill-job"
REGION="europe-west2"
JOB_NAME="films-tensile-curve-backfill"
SERVICE_ACCOUNT="sa-tensile-ingest@notpla-machine-data.iam.gserviceaccount.com"

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
    --task-timeout=86400 \
    --max-retries=0 \
    --memory=1Gi
