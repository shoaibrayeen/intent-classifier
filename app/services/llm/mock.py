"""A deterministic, offline stand-in for a language model.

It reads the same prompt a real provider would receive and fills the schema
from the request text with plain rules: enum values by name, dates and years by
pattern, numbers by digit, booleans by keyword, and strings by capitalised
words. It is not clever, and that is the point -- it makes the *whole* pipeline
(instructions, history, validation, carry-over, tool routing) exercisable
without a key, and its output is stable enough to test against.

Enable with ``LLM_PROVIDER=mock``.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LONG_DATE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(20\d{2}|19\d{2})\b")
_NUMBER = re.compile(r"(?<![\w-])-?\d+(?:[.,]\d+)?(?![\w-])")
_CAPITALISED = re.compile(r"\b([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+)*)\b")
_ID_LIKE = re.compile(r"\b([A-Z]{1,4}-\d{2,})\b")

# Words that are capitalised for reasons other than being a name.
_SKIP = {
    "now",
    "then",
    "also",
    "next",
    "ok",
    "okay",
    "but",
    "so",
    "well",
    "great",
    "thanks",
    "hi",
    "hello",
    "hey",
    "yes",
    "no",
    "again",
    "instead",
    "same",
    "only",
    "i",
    "show",
    "find",
    "list",
    "give",
    "search",
    "what",
    "which",
    "when",
    "who",
    "how",
    "the",
    "a",
    "an",
    "all",
    "my",
    "our",
    "me",
    "please",
    "compare",
    "summarize",
    "summarise",
    "tell",
    "pull",
    "get",
    "display",
    "can",
    "could",
    "do",
    "does",
    "is",
    "are",
    "and",
    "or",
    "contracts",
    "contract",
    "agreements",
    "agreement",
    "employees",
    "employee",
    "q1",
    "q2",
    "q3",
    "q4",
    "tl",
    "dr",
    "msa",
    "sow",
    *MONTHS.split("|"),
}
_TRUE_WORDS = {"yes", "true", "active", "enabled", "renew", "renewal", "renewed"}
_FALSE_WORDS = {"no", "false", "inactive", "disabled", "expired", "cancelled", "canceled"}


class MockLLMClient:
    """Fills an entity schema from the request text by rule."""

    name = "mock"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def configured(self) -> bool:
        return True

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            return {}
        text = str(payload.get("request", ""))
        schema = payload.get("entity_schema") or {}
        today = payload.get("today")
        return extract_by_rule(text, schema, today)


def extract_by_rule(text: str, schema: dict[str, Any], today: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    lowered = text.casefold()
    used_spans: list[str] = []

    for name, spec in schema.items():
        spec = spec if isinstance(spec, dict) else {"type": str(spec)}
        declared = str(spec.get("type", "string")).lower()
        value: Any = None

        if declared == "enum":
            for option in spec.get("values", []):
                if re.search(rf"\b{re.escape(str(option).casefold())}\b", lowered):
                    value = option
                    break
        elif declared == "date":
            value = _find_date(text, lowered, today)
        elif declared == "integer":
            if "year" in name.lower():
                resolved = _find_date(text, lowered, today)
                value = int(resolved[:4]) if resolved and _YEAR.fullmatch(resolved[:4]) else None
            else:
                match = _NUMBER.search(text)
                if match:
                    value = int(match.group(0).replace(",", "").split(".")[0])
        elif declared == "number":
            match = _NUMBER.search(text)
            if match:
                value = float(match.group(0).replace(",", ""))
        elif declared == "boolean":
            words = set(re.findall(r"[a-z]+", lowered))
            if words & _TRUE_WORDS:
                value = True
            elif words & _FALSE_WORDS:
                value = False
        elif declared == "array":
            names = _proper_nouns(text, used_spans)
            value = names or None
        else:  # string
            if "id" in name.lower():
                # An identifier field never falls back to a name: "March" is
                # not a contract id, and a wrong id is worse than none.
                match = _ID_LIKE.search(text)
                value = match.group(1) if match else None
            else:
                names = _proper_nouns(text, used_spans)
                if names:
                    value = names[0]
                    used_spans.append(names[0])

        if value is not None:
            result[name] = value
    return result


def _find_date(text: str, lowered: str, today: str | None) -> str | None:
    if match := _ISO_DATE.search(text):
        return match.group(0)
    if match := _LONG_DATE.search(text):
        month = datetime.strptime(match.group(1)[:3], "%b").month
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    if today and ("this year" in lowered or "current year" in lowered):
        return today[:4]
    if today and "next year" in lowered:
        return str(int(today[:4]) + 1)
    if match := _YEAR.search(text):
        return match.group(0)
    return None


def _proper_nouns(text: str, exclude: list[str]) -> list[str]:
    found: list[str] = []
    for match in _CAPITALISED.finditer(text):
        candidate = match.group(1).strip(".,;:!?")
        tokens = candidate.split()
        # Drop leading capitalised function words ("Show Microsoft" -> "Microsoft").
        while tokens and tokens[0].casefold() in _SKIP:
            tokens = tokens[1:]
        if not tokens:
            continue
        candidate = " ".join(tokens)
        if candidate.casefold() in _SKIP or candidate in exclude or candidate in found:
            continue
        if _ID_LIKE.fullmatch(candidate) or _YEAR.fullmatch(candidate):
            continue
        found.append(candidate)
    return found
