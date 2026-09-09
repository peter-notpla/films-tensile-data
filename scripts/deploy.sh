#!/usr/bin/env bash
# Deploy one pipeline function, staging shared/ into its source directory
# first since `gcloud functions deploy --source=.` only packages the
# directory it's pointed at: nothing outside it, including repo-root
# shared/, is ever included. See pipeline-roadmap.md item 2.1 / Phase 4.
#
# Usage: scripts/deploy.sh <pipeline-dir> [function-name] [service-account] [runtime] [memory] [trigger] [entry-point] [env-vars]
#   scripts/deploy.sh films-extrusion-csv-processor
#   scripts/deploy.sh films-extrusion-csv-processor films-extrusion-csv-processor sa-extrusion-ingest@notpla-machine-data.iam.gserviceaccount.com
#   scripts/deploy.sh films-friction-raw-processor films-friction-raw-processor sa-friction-ingest@notpla-machine-data.iam.gserviceaccount.com python312
#   scripts/deploy.sh films-pipeline-failure-alerter films-pipeline-failure-alerter "" "" 512Mi
#   scripts/deploy.sh films-pipeline-row-rescue films-pipeline-row-rescue films-pipeline-row-rescue-sa@notpla-machine-data.iam.gserviceaccount.com python312 "" http rescue
#   scripts/deploy.sh films-pipeline-failure-alerter films-pipeline-failure-alerter "" "" "" "" "" ROW_RESCUE_URL=https://films-pipeline-row-rescue-plngeip6ya-nw.a.run.app
#
# function-name defaults to pipeline-dir, which matches all pipelines
# today. service-account, if omitted, leaves the function's current
# service account untouched (gcloud functions deploy only changes it when
# --service-account is passed) - see pipeline-roadmap.md item 2.6 for the
# least-privilege SAs this is meant to cut over to, one pipeline at a time.
# runtime is only needed for a function's first-ever deploy (gcloud
# requires it then; an existing function keeps its current runtime
# automatically without it) - leave it unset for every normal redeploy.
# memory, if omitted, leaves the function's current memory untouched, same
# reasoning as runtime - only pass it when actually changing the limit.
# trigger is only needed for a function's first-ever deploy too, same
# reasoning as runtime - an existing function keeps its trigger type
# automatically without it, and gcloud errors outright on a brand-new
# function with no trigger specified at all. Only "http" is supported
# (the only trigger type this repo has ever needed for a first-time HTTP
# function); event-triggered first deploys still need their own
# --trigger-bucket/--trigger-event-filters invocation, not this script.
# entry-point is, again, only needed for a function's first-ever deploy -
# an existing function keeps whatever entry point it was created with
# (e.g. films-pipeline-failure-alerter's is check_and_alert, not its
# function name) automatically on every redeploy. Without --entry-point,
# gcloud looks for a function named after the deployed function itself,
# which fails outright if the code's actual decorated function is named
# something else, as films-pipeline-row-rescue's ("rescue") is.
# env-vars, if given, is passed straight to --update-env-vars (comma-
# separated KEY=VALUE pairs, gcloud's own syntax) and merges into whatever
# env vars the function already has - existing ones not mentioned are left
# alone. Exists so an env-var-only change never has a reason to reach for
# a raw `gcloud functions deploy --source=...` instead of this script -
# that always omits shared/ staging and crashes the container outright
# (ModuleNotFoundError), even for a change that has nothing to do with
# shared/ at all.

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

MEMORY="${5:-}"
MEMORY_ARGS=()
if [ -n "$MEMORY" ]; then
    MEMORY_ARGS=(--memory="$MEMORY")
fi

TRIGGER="${6:-}"
TRIGGER_ARGS=()
if [ -n "$TRIGGER" ]; then
    case "$TRIGGER" in
        http) TRIGGER_ARGS=(--trigger-http) ;;
        *)
            echo "Unsupported trigger '$TRIGGER' - only 'http' is supported by this script." >&2
            exit 1
            ;;
    esac
fi

ENTRY_POINT="${7:-}"
ENTRY_POINT_ARGS=()
if [ -n "$ENTRY_POINT" ]; then
    ENTRY_POINT_ARGS=(--entry-point="$ENTRY_POINT")
fi

ENV_VARS="${8:-}"
ENV_VARS_ARGS=()
if [ -n "$ENV_VARS" ]; then
    ENV_VARS_ARGS=(--update-env-vars="$ENV_VARS")
fi

if [ -n "$SERVICE_ACCOUNT" ]; then
    echo "Deploying $FUNCTION_NAME (region=$REGION, source=$TARGET_DIR, service-account=$SERVICE_ACCOUNT) ..."
    gcloud functions deploy "$FUNCTION_NAME" \
        --region="$REGION" \
        --gen2 \
        "${RUNTIME_ARGS[@]}" \
        "${MEMORY_ARGS[@]}" \
        "${TRIGGER_ARGS[@]}" \
        "${ENTRY_POINT_ARGS[@]}" \
        "${ENV_VARS_ARGS[@]}" \
        --source="$TARGET_DIR" \
        --service-account="$SERVICE_ACCOUNT" \
        --quiet
else
    echo "Deploying $FUNCTION_NAME (region=$REGION, source=$TARGET_DIR) ..."
    gcloud functions deploy "$FUNCTION_NAME" \
        --region="$REGION" \
        --gen2 \
        "${RUNTIME_ARGS[@]}" \
        "${MEMORY_ARGS[@]}" \
        "${TRIGGER_ARGS[@]}" \
        "${ENTRY_POINT_ARGS[@]}" \
        "${ENV_VARS_ARGS[@]}" \
        --source="$TARGET_DIR" \
        --quiet
fi

echo "Deploy command completed. Verify against Cloud Logging for a" \
     "distinctive string from the new code before trusting it."
