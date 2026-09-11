"""Entity extraction.

Runs after an intent has been resolved, never before, and never for UNKNOWN:
extracting arguments for a request the system did not understand would produce
confident nonsense. Extraction failure is never fatal -- a classification with
no entities is still a useful answer.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.models.classification import EntityExtractionInfo
from app.services.entity_schema import validate
from app.services.llm.client import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You extract structured values from a user's request.\n"
    "You are given the request and a schema of the entities the caller accepts.\n"
    "Rules:\n"
    "1. Return a single JSON object and nothing else.\n"
    "2. Use only keys that appear in the schema. Never invent keys.\n"
    "3. Omit any entity the request does not actually mention. Do not guess.\n"
    "4. Copy values from the request; do not paraphrase or expand abbreviations.\n"
    "5. For an enum, return exactly one of the listed values.\n"
    "6. For a date, return ISO 8601 (YYYY-MM-DD), or just the year if only a "
    "year is given.\n"
    "If the request mentions none of the entities, return {}."
)


def build_user_prompt(text: str, schema: dict[str, Any], today: str | None = None) -> str:
    payload: dict[str, Any] = {"request": text, "entity_schema": schema}
    if today:
        # Relative expressions ("this year", "next month") need an anchor.
        payload["today"] = today
    return json.dumps(payload, ensure_ascii=False)


class EntityExtractor:
    def __init__(self, settings: Settings, client: LLMClient) -> None:
        self._settings = settings
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._settings.entity_extraction_enabled and self._client.configured

    @property
    def available(self) -> bool:
        """Configured provider, regardless of whether the flag is on."""
        return self._client.configured

    async def extract(
        self, text: str, schema: dict[str, Any], today: str | None = None
    ) -> tuple[dict[str, Any], EntityExtractionInfo]:
        # Whether extraction *should* run is the caller's decision, so that a
        # per-request override is not silently overruled by the global default.
        if not schema:
            return {}, EntityExtractionInfo(status="skipped", detail="intent declares no entities")
        if not self._client.configured:
            return {}, EntityExtractionInfo(
                status="unavailable", detail="no LLM provider configured"
            )

        try:
            raw = await self._client.complete_json(
                SYSTEM_PROMPT, build_user_prompt(text, schema, today)
            )
        except LLMUnavailable as exc:
            logger.warning("entity extraction unavailable: %s", exc)
            return {}, EntityExtractionInfo(status="failed", detail=str(exc))
        except Exception as exc:
            logger.exception("entity extraction raised unexpectedly")
            return {}, EntityExtractionInfo(status="failed", detail=f"{type(exc).__name__}: {exc}")

        entities, rejected = validate(raw, schema)
        if rejected:
            logger.info("entity extraction dropped %d value(s): %s", len(rejected), rejected)
        return entities, EntityExtractionInfo(
            status="ok", model=self._settings.openai_model, rejected=rejected
        )
