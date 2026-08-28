"""Shared HTML building blocks for pipeline emails, matching the visual
language from Peter's Notpla Holiday Handover email design system (orange
#E8623A headers, 600px white card, Arial throughout) so pipeline emails
look consistent with the rest of what he forwards to non-technical
teammates. Kept deliberately small: plain functions returning HTML
fragments, no templating engine, since the set of blocks needed here is
fixed and small.
"""

from html import escape

ACCENT = "#E8623A"
TEXT = "#222222"
MUTED = "#888888"
DIVIDER_LIGHT = "#eeeeee"
DIVIDER_FOOTER = "#dddddd"

FOOTER_TEXT = "This is an automated message. Please do not reply to this email."


def section_header(text):
    return (
        f'<div style="font-family: Arial, sans-serif; font-size: 13px; '
        f'font-weight: bold; color: {ACCENT}; text-transform: uppercase; '
        f'letter-spacing: 0.5px; margin: 20px 0 8px 0;">{escape(text)}</div>'
    )


def divider():
    return f'<hr style="border: none; border-top: 2px solid {DIVIDER_LIGHT}; margin: 16px 0;">'


def paragraph(text):
    return f'<p style="font-family: Arial, sans-serif; font-size: 14px; color: {TEXT}; margin: 8px 0;">{text}</p>'


def muted_note(text):
    return f'<p style="font-family: Arial, sans-serif; font-size: 12px; color: {MUTED}; margin: 8px 0;">{text}</p>'


def key_value_table(rows):
    """rows: list of (label, value) tuples, value may contain HTML."""
    body = "".join(
        '<tr>'
        f'<td style="font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; '
        f'color: {TEXT}; padding: 5px 12px 5px 0; white-space: nowrap; vertical-align: top;">{escape(str(label))}</td>'
        f'<td style="font-family: Arial, sans-serif; font-size: 14px; color: {TEXT}; padding: 5px 0;">{value}</td>'
        '</tr>'
        for label, value in rows
    )
    return f'<table style="width: 100%; border-collapse: collapse;">{body}</table>'


def data_table(headers, rows, font_size=13):
    head = "".join(
        f'<th style="text-align: left; font-family: Arial, sans-serif; font-size: {font_size}px; '
        f'font-weight: bold; color: {TEXT}; border-bottom: 2px solid {DIVIDER_LIGHT}; padding: 5px 12px 5px 0;">{escape(h)}</th>'
        for h in headers
    )
    body = ""
    for row in rows:
        cells = "".join(
            f'<td style="font-family: Arial, sans-serif; font-size: {font_size}px; color: {TEXT}; padding: 5px 12px 5px 0;">{cell}</td>'
            for cell in row
        )
        body += f"<tr>{cells}</tr>"
    return f'<table style="width: 100%; border-collapse: collapse;"><tr>{head}</tr>{body}</table>'


def cta_link(text, url):
    return (
        f'<p style="font-family: Arial, sans-serif; font-size: 14px; margin: 16px 0;">'
        f'<a href="{escape(url)}" style="color: {ACCENT}; font-weight: bold; text-decoration: none;">{escape(text)}</a></p>'
    )


def wrap_email(salutation, body_html):
    return f"""
    <div style="max-width: 600px; margin: 0 auto; padding: 20px; background: #ffffff;
                font-family: Arial, sans-serif; font-size: 14px; color: {TEXT};">
      <p style="font-family: Arial, sans-serif; font-size: 14px; color: {TEXT}; margin: 0 0 12px 0;">{escape(salutation)}</p>
      {body_html}
      <hr style="border: none; border-top: 1px solid {DIVIDER_FOOTER}; margin: 30px 0;">
      <p style="font-family: Arial, sans-serif; font-size: 12px; color: {MUTED}; margin: 0;">{FOOTER_TEXT}</p>
    </div>
    """
