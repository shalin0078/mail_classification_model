from __future__ import annotations

import re
from email import policy
from email.parser import Parser
from html import unescape

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_email_body(raw_email: str) -> str:
    """Return readable text from a raw email or plain message body."""
    if not raw_email:
        return ""
    if "\n\n" not in raw_email and "\r\n\r\n" not in raw_email:
        return raw_email

    message = Parser(policy=policy.default).parsestr(raw_email)
    if message.is_multipart():
        parts: list[str] = []
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_content()
                if isinstance(payload, str):
                    parts.append(payload)
        if parts:
            return "\n".join(parts)

    payload = message.get_content() if message.get_content_type().startswith("text/") else raw_email
    return payload if isinstance(payload, str) else raw_email


def normalize_email_text(text: str) -> str:
    """Normalize text while preserving security-relevant placeholders."""
    text = extract_email_body(text)
    text = unescape(text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = EMAIL_PATTERN.sub(" EMAILTOKEN ", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip().lower()


def combine_email_fields(text: str, subject: str = "", sender: str = "") -> str:
    pieces = [
        f"subject: {subject or ''}",
        f"sender: {sender or ''}",
        f"body: {text or ''}",
    ]
    return "\n".join(pieces)
