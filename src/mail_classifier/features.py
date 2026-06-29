from __future__ import annotations

import re
from urllib.parse import urlparse

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MONEY_PATTERN = re.compile(r"[$₹€£]|\b(?:usd|inr|eur|gbp|dollars?|rupees?)\b", re.IGNORECASE)
URGENCY_TERMS = {
    "urgent",
    "immediately",
    "limited",
    "expire",
    "expires",
    "final notice",
    "act now",
    "verify now",
}
CREDENTIAL_TERMS = {
    "password",
    "otp",
    "pin",
    "login",
    "verify",
    "credential",
    "account suspended",
}
ATTACHMENT_TERMS = {"attached", "attachment", "invoice", "receipt", "statement", "pdf", "docx"}
PROMOTION_TERMS = {"discount", "offer", "sale", "deal", "coupon", "free", "cashback"}


class EmailRiskFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract dense numeric features that complement TF-IDF."""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.asarray([self._features(str(text)) for text in x], dtype=float)

    def _features(self, text: str) -> list[float]:
        lowered = text.lower()
        urls = URL_PATTERN.findall(text)
        domains = [urlparse(url if url.startswith("http") else f"http://{url}").netloc for url in urls]
        token_count = max(len(re.findall(r"\w+", text)), 1)
        uppercase_chars = sum(1 for char in text if char.isupper())
        alpha_chars = max(sum(1 for char in text if char.isalpha()), 1)

        return [
            len(text),
            token_count,
            len(urls),
            sum(1 for domain in domains if domain.count(".") >= 2),
            text.count("!"),
            text.count("?"),
            uppercase_chars / alpha_chars,
            len(MONEY_PATTERN.findall(text)),
            self._term_count(lowered, URGENCY_TERMS),
            self._term_count(lowered, CREDENTIAL_TERMS),
            self._term_count(lowered, ATTACHMENT_TERMS),
            self._term_count(lowered, PROMOTION_TERMS),
        ]

    @staticmethod
    def _term_count(text: str, terms: set[str]) -> int:
        return sum(1 for term in terms if term in text)

