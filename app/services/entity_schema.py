"""Entity schema: the contract between an intent and its downstream tool.

An intent declares the entities it cares about:

    {
      "counterparty": {"type": "string", "required": true},
      "status":       {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
      "signed_after": {"type": "date"}
    }

Anything a language model returns is untrusted input. It is coerced to the
declared type and dropped if it does not fit, so a downstream tool never
receives an argument the intent did not declare.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class EntityType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    ARRAY = "array"


_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}
_YEAR = re.compile(r"^\d{4}$")


class EntityRejected(Exception):
    """A supplied value does not satisfy the declared type."""


def _coerce_date(value: Any) -> str:
    """Return an ISO date string. A bare year is kept as-is."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if _YEAR.match(text):
        return text
    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%Y/%m/%d",
        "%B %d, %Y",
        "%d %B %Y",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise EntityRejected(f"'{text}' is not a recognizable date")


def coerce_value(value: Any, spec: dict[str, Any]) -> Any:
    """Coerce one value to its declared type, or raise EntityRejected."""
    if value is None:
        raise EntityRejected("value is null")

    declared = str(spec.get("type", EntityType.STRING)).lower()

    if declared == EntityType.ARRAY:
        items = value if isinstance(value, list) else [value]
        item_spec = spec.get("items", {"type": "string"})
        coerced = []
        for item in items:
            try:
                coerced.append(coerce_value(item, item_spec))
            except EntityRejected:
                continue
        if not coerced:
            raise EntityRejected("no usable items in array")
        return coerced

    if isinstance(value, list | dict):
        raise EntityRejected(f"expected {declared}, got {type(value).__name__}")

    if declared == EntityType.INTEGER:
        try:
            return int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise EntityRejected(f"'{value}' is not an integer") from exc

    if declared == EntityType.NUMBER:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise EntityRejected(f"'{value}' is not a number") from exc

    if declared == EntityType.BOOLEAN:
        if isinstance(value, bool):
            return value
        text = str(value).strip().casefold()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise EntityRejected(f"'{value}' is not a boolean")

    if declared == EntityType.DATE:
        return _coerce_date(value)

    if declared == EntityType.ENUM:
        allowed = [str(option) for option in spec.get("values", [])]
        text = str(value).strip()
        if not allowed:
            return text
        for option in allowed:
            if option.casefold() == text.casefold():
                return option  # normalize to the declared spelling
        raise EntityRejected(f"'{text}' is not one of {allowed}")

    text = str(value).strip()
    if not text:
        raise EntityRejected("empty string")
    return text


def validate(raw: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Filter and coerce raw values against a schema.

    Returns the accepted entities and a list of human-readable rejections.
    Keys absent from the schema are dropped: an intent's schema is the
    allowlist, not a suggestion.
    """
    accepted: dict[str, Any] = {}
    rejected: list[str] = []

    if not isinstance(raw, dict):
        return {}, ["extractor did not return an object"]

    for name, value in raw.items():
        spec = schema.get(name)
        if spec is None:
            rejected.append(f"{name}: not declared by this intent")
            continue
        if not isinstance(spec, dict):
            spec = {"type": str(spec)}
        try:
            accepted[name] = coerce_value(value, spec)
        except EntityRejected as exc:
            rejected.append(f"{name}: {exc}")

    return accepted, rejected


def required_names(schema: dict[str, Any]) -> list[str]:
    return [
        name
        for name, spec in schema.items()
        if isinstance(spec, dict) and bool(spec.get("required", False))
    ]


def missing_required(entities: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    return [name for name in required_names(schema) if name not in entities]
