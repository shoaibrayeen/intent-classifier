"""Query and example text normalization.

The same functions are used when building the BM25 index and when handling a
query; a mismatch between the two is the classic cause of silent BM25 misses.
"""

from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = "simple_v1"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize_text(text: str) -> str:
    """NFKC-fold, lowercase, collapse whitespace."""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.casefold().split())


def strip_accents(text: str) -> str:
    """Fold accents so "résumé" and "resume" produce the same BM25 token."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Digits are kept (ids, years, amounts)."""
    return _TOKEN_RE.findall(strip_accents(normalize_text(text)))
