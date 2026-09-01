# Cloud Run Job for scripts/backfill_curve_points.py, so a long backlog
# drain runs on GCP infra instead of an interactive Cloud Shell VM - immune
# to a disconnected terminal or a recycled Cloud Shell session, which is
# what silently killed the 28 August 2026 backfill run (see
# pipeline-roadmap.md's Phase 5 entry).
#
# shared/ and scripts/backfill_curve_points.py are staged into this
# directory by scripts/deploy_curve_backfill_job.sh immediately before
# build, then removed after - mirrors scripts/deploy.sh's staging pattern
# for the Cloud Functions, since `gcloud run jobs deploy --source=DIR` only
# packages DIR itself as build context.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared/ ./shared/
COPY scripts/backfill_curve_points.py ./scripts/backfill_curve_points.py
ENTRYPOINT ["python3", "scripts/backfill_curve_points.py"]
