"""Jinja2 environment shared by every HTML route."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def json_pretty(value: Any) -> str:
    if not value:
        return "{}"
    return json.dumps(value, indent=2, ensure_ascii=False)


def ts(value: float | None) -> str:
    if not value:
        return "-"
    return datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%d %H:%M")


templates.env.filters["pct"] = pct
templates.env.filters["num"] = num
templates.env.filters["json_pretty"] = json_pretty
templates.env.filters["ts"] = ts
