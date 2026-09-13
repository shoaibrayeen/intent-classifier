"""Jinja2 environment shared by every HTML route."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def asset(path: str) -> str:
    """A versioned URL for a static file.

    Browsers cache /static/app.css aggressively, so a stylesheet change after a
    deploy would otherwise not reach anyone who had already loaded the page.
    Stamping the file's modification time makes a changed file a new URL.
    """
    url = f"/static/{path}"
    try:
        stamp = int((STATIC_DIR / path).stat().st_mtime)
    except OSError:
        return url
    return f"{url}?v={stamp}"


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


#: One definition of the confidence bands, used everywhere they are shown:
#: green above 0.8, amber 0.5 to 0.8, red below. Keeping it here means the dot
#: in a table and the number beside it can never disagree.
HIGH_BAND = 0.8
MEDIUM_BAND = 0.5


def band(value: float | None) -> str:
    if value is None:
        return "low"
    if value >= HIGH_BAND:
        return "high"
    return "medium" if value >= MEDIUM_BAND else "low"


def json_compact(value: Any) -> str:
    """One-line JSON, for a chat bubble where a block would dominate."""
    if not value:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def ts(value: float | None) -> str:
    if not value:
        return "-"
    return datetime.fromtimestamp(value, UTC).strftime("%Y-%m-%d %H:%M")


templates.env.filters["pct"] = pct
templates.env.filters["num"] = num
templates.env.filters["json_pretty"] = json_pretty
templates.env.filters["json_compact"] = json_compact
templates.env.filters["ts"] = ts
templates.env.filters["band"] = band
templates.env.globals["asset"] = asset
