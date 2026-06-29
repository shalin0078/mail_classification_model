from __future__ import annotations

from pathlib import Path

import pandas as pd

from mail_classifier.preprocessing import combine_email_fields

REQUIRED_COLUMNS = {"text", "label"}


def load_dataset(path: str | Path) -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required column(s): {missing_text}")

    data = data.dropna(subset=["text", "label"]).copy()
    if data.empty:
        raise ValueError("Dataset has no usable rows after dropping empty text/label values.")

    subjects = data["subject"] if "subject" in data.columns else pd.Series([""] * len(data))
    senders = data["sender"] if "sender" in data.columns else pd.Series([""] * len(data))
    combined = [
        combine_email_fields(text=str(text), subject=str(subject), sender=str(sender))
        for text, subject, sender in zip(data["text"], subjects, senders)
    ]
    return pd.Series(combined), data["label"].astype(str)
