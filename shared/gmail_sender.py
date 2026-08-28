"""Sends email as peter@notpla.com via the Gmail API, using an OAuth
refresh token obtained once by hand (see pipeline-history.md, 28 August
2026, for how it was created and why: Cloud Monitoring alert emails were
found to render most fields as null, since the alert condition's
crossSeriesReducer aggregation only preserves label values for the
groupByFields, silently dropping every other label the digest/alerts
relied on for content). Sending directly from the pipeline's own code
avoids that class of bug entirely and gives full control over formatting.
"""

import base64
from email.mime.text import MIMEText

import google.auth.transport.requests
from google.cloud import secretmanager
from google.oauth2.credentials import Credentials

GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

SECRET_NAMES = {
    "refresh_token": "pipeline-email-gmail-refresh-token",
    "client_id": "pipeline-email-gmail-client-id",
    "client_secret": "pipeline-email-gmail-client-secret",
}


def _get_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    return client.access_secret_version(name=name).payload.data.decode("utf-8")


def _get_credentials(project_id):
    return Credentials(
        token=None,
        refresh_token=_get_secret(project_id, SECRET_NAMES["refresh_token"]),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=_get_secret(project_id, SECRET_NAMES["client_id"]),
        client_secret=_get_secret(project_id, SECRET_NAMES["client_secret"]),
        scopes=["https://www.googleapis.com/auth/gmail.send"],
    )


def send_html_email(project_id, to, subject, html_body):
    """Raises on any failure rather than swallowing it: an email that
    silently fails to send is worse than one that errors loudly in the
    function's own logs, where the existing manifest/log-based tooling
    can still surface the failure."""
    creds = _get_credentials(project_id)
    creds.refresh(google.auth.transport.requests.Request())

    msg = MIMEText(html_body, "html")
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    session = google.auth.transport.requests.AuthorizedSession(creds)
    resp = session.post(GMAIL_SEND_URL, json={"raw": raw})
    resp.raise_for_status()
    return resp.json()
