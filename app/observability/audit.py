"""Append-only audit log.

Every change to the catalogue and every classification is recorded as one JSON
object per line. JSONL is chosen over a table because the workload is
append-only and read rarely, and because a file survives the database being
rebuilt -- which matters when the database is also the configuration store.

Queries are recorded, never stored verbatim by default: set
AUDIT_LOG_QUERY_TEXT to opt in, since user queries can carry personal data.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.observability.context import get_principal, get_request_id

logger = logging.getLogger(__name__)


class AuditLog:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._path = Path(settings.audit_log_path)
        self._ready = False

    @property
    def enabled(self) -> bool:
        return self._settings.audit_log_enabled

    @property
    def path(self) -> Path:
        return self._path

    def _ensure_ready(self) -> None:
        if not self._ready:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._ready = True

    def record(
        self,
        action: str,
        resource: str = "",
        outcome: str = "ok",
        **fields: Any,
    ) -> None:
        if not self.enabled:
            return
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "request_id": get_request_id(),
            "principal": get_principal(),
            "action": action,
            "resource": resource,
            "outcome": outcome,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        try:
            with self._lock:
                self._ensure_ready()
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError:
            # An unwritable audit log must not take the service down.
            logger.exception("could not write audit entry for %s", action)

    def tail(self, limit: int = 100) -> list[dict[str, Any]]:
        """Most recent entries, newest first. For the admin UI."""
        if not self._path.exists():
            return []
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                lines = handle.readlines()[-limit:]
        except OSError:
            logger.exception("could not read the audit log")
            return []
        entries: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def record_query(self, text: str) -> str | None:
        """Return the query only if recording query text is enabled."""
        return text if self._settings.audit_log_query_text else None
