"""Shared model helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum


class Status(StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"


def new_id() -> str:
    return uuid.uuid4().hex


def now_ts() -> float:
    """Current UTC timestamp as a float (Chroma metadata allows only scalars)."""
    return datetime.now(UTC).timestamp()


def to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def name_key(name: str) -> str:
    """Case-insensitive uniqueness key for domain / intent names."""
    return " ".join(name.strip().casefold().split())
