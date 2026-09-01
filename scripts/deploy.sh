#!/usr/bin/env bash
# Deploy one pipeline function, staging shared/ into its source directory
# first since `gcloud functions deploy --source=.` only packages the
# directory it's pointed at: nothing outside it, including repo-root
# shared/, is ever included. See pipeline-roadmap.md item 2.1 / Phase 4.
#
# Usage: scripts/deploy.sh <pipeline-dir> [function-name] [service-account] [runtime]
#   scripts/deploy.sh films-extrusion-csv-processor
#   scripts/deploy.sh films-extrusion-csv-processor films-extrusion-csv-processor sa-extrusion-ingest@notpla-machine-data.iam.gserviceaccount.com
#   scripts/deploy.sh films-friction-raw-processor films-friction-raw-processor sa-friction-ingest@notpla-machine-data.iam.gserviceaccount.com python312
#
# function-name defaults to pipeline-dir, which matches all pipelines
# today. service-account, if omitted, leaves the function's current
# service account untouched (gcloud functions deploy only changes it when
# --service-account is passed) - see pipeline-roadmap.md item 2.6 for the
# least-privilege SAs this is meant to cut over to, one pipeline at a time.
# runtime is only needed for a function's first-ever deploy (gcloud
# requires it then; an existing function keeps its current runtime
# automatically without it) - leave it unset for every normal redeploy.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <pipeline-dir> [function-name] [service-account]" >&2
    exit 1
fi

PIPELINE_DIR="$1"
FUNCTION_NAME="${2:-$PIPELINE_DIR}"
SERVICE_ACCOUNT="${3:-}"
REGION="europe-west2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$REPO_ROOT/pipelines/$PIPELINE_DIR"
STAGED_SHARED="$TARGET_DIR/shared"

if [ ! -d "$TARGET_DIR" ]; then
    echo "No such pipeline directory: $TARGET_DIR" >&2
    exit 1
fi

if [ -e "$STAGED_SHARED" ]; then
    echo "Refusing to run: $STAGED_SHARED already exists (stale copy from an" >&2
    echo "interrupted deploy?). Remove it and re-run." >&2
    exit 1
fi

cleanup() {
    rm -rf "$STAGED_SHARED"
}
trap cleanup EXIT

echo "Compiling $PIPELINE_DIR/main.py..."
python3 -m py_compile "$TARGET_DIR/main.py"
echo "  compiles clean"

echo "Staging shared/ into $PIPELINE_DIR/shared/ ..."
mkdir -p "$STAGED_SHARED"
# verify_*.py scripts are dev-only (they import google.cloud clients to
# replay against real buckets/BigQuery); the deployed function never needs
# them, so they're excluded from both counts below, not just skipped.
find "$REPO_ROOT/shared" -maxdepth 1 -name "*.py" ! -name "verify_*.py" \
    -exec cp {} "$STAGED_SHARED/" \;
STAGED_COUNT=$(find "$STAGED_SHARED" -name "*.py" | wc -l)
SOURCE_COUNT=$(find "$REPO_ROOT/shared" -maxdepth 1 -name "*.py" ! -name "verify_*.py" | wc -l)
if [ "$STAGED_COUNT" -ne "$SOURCE_COUNT" ]; then
    echo "Staged $STAGED_COUNT files but shared/ has $SOURCE_COUNT deployable files; refusing to deploy." >&2
    exit 1
fi
echo "  staged $STAGED_COUNT file(s)"

# --runtime is only required by gcloud on a function's first-ever deploy
# (an existing function keeps its current runtime automatically without
# this flag - and the three original pipelines are on python311, not
# python312, so passing this unconditionally would silently bump their
# runtime on next redeploy). RUNTIME_ARGS stays empty for every existing
# pipeline's normal redeploy; only a brand-new function's first deploy
# needs $4 set explicitly.
RUNTIME="${4:-}"
RUNTIME_ARGS=()
if [ -n "$RUNTIME" ]; then
    RUNTIME_ARGS=(--runtime="$RUNTIME")
fi

if [ -n "$SERVICE_ACCOUNT" ]; then
    echo "Deploying $FUNCTION_NAME (region=$REGION, source=$TARGET_DIR, service-account=$SERVICE_ACCOUNT) ..."
    gcloud functions deploy "$FUNCTION_NAME" \
        --region="$REGION" \
        --gen2 \
        "${RUNTIME_ARGS[@]}" \
        --source="$TARGET_DIR" \
        --service-account="$SERVICE_ACCOUNT" \
        --quiet
else
    echo "Deploying $FUNCTION_NAME (region=$REGION, source=$TARGET_DIR) ..."
    gcloud functions deploy "$FUNCTION_NAME" \
        --region="$REGION" \
        --gen2 \
        "${RUNTIME_ARGS[@]}" \
        --source="$TARGET_DIR" \
        --quiet
fi

echo "Deploy command completed. Verify against Cloud Logging for a" \
     "distinctive string from the new code before trusting it."
