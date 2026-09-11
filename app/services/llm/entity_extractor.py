"""Entity extraction.

Runs after an intent has been resolved, never before, and never for UNKNOWN:
extracting arguments for a request the system did not understand would produce
confident nonsense. Extraction failure is never fatal -- a classification with
no entities is still a useful answer.

The prompt is assembled from three layers so each domain speaks its own
language without the code changing:

    base system prompt        (this file: the rules every domain shares)
    + domain.system_instructions
    + intent.extraction_hints
    -------------------------------------------------------------- system
    domain.user_instructions
    + conversation history    (earlier turns in the session, if any)
    + the request, schema and today's date                        -- user
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.models.classification import EntityExtractionInfo, SessionTurn
from app.models.domain import DomainConfig
from app.models.intent import IntentConfig
from app.services.entity_schema import validate
from app.services.llm.client import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = (
    "You extract structured values from a user's request.\n"
    "You are given the request, a schema of the entities the caller accepts, and "
    "possibly earlier turns of the same conversation.\n"
    "Rules:\n"
    "1. Return a single JSON object and nothing else.\n"
    "2. Use only keys that appear in the schema. Never invent keys.\n"
    "3. Omit any entity the request does not actually mention. Do not guess.\n"
    "4. Copy values from the request; do not paraphrase or expand abbreviations.\n"
    "5. For an enum, return exactly one of the listed values.\n"
    "6. For a date, return ISO 8601 (YYYY-MM-DD), or just the year if only a "
    "year is given.\n"
    '7. If the request refers back to something from an earlier turn ("it", '
    '"they", "the same one"), resolve it from the conversation history.\n'
    "If the request mentions none of the entities, return {}."
)


def build_system_prompt(domain: DomainConfig | None, intent: IntentConfig | None) -> str:
    parts = [BASE_SYSTEM_PROMPT]
    if domain and domain.system_instructions.strip():
        parts.append(f"Domain instructions ({domain.name}):\n{domain.system_instructions.strip()}")
    if intent and intent.extraction_hints.strip():
        parts.append(f"Intent notes ({intent.name}):\n{intent.extraction_hints.strip()}")
    return "\n\n".join(parts)


def build_user_prompt(
    text: str,
    schema: dict[str, Any],
    today: str | None = None,
    domain: DomainConfig | None = None,
    history: list[SessionTurn] | None = None,
) -> str:
    payload: dict[str, Any] = {}
    if domain and domain.user_instructions.strip():
        payload["instructions"] = domain.user_instructions.strip()
    if history:
        payload["conversation_history"] = [
            {
                "turn": turn.turn,
                "request": turn.text,
                "intent": turn.intent,
                "entities": turn.entities,
            }
            for turn in history
        ]
    payload["request"] = text
    payload["entity_schema"] = schema
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

    @property
    def provider_name(self) -> str:
        return getattr(self._client, "name", "unknown")

    async def extract(
        self,
        text: str,
        schema: dict[str, Any],
        today: str | None = None,
        domain: DomainConfig | None = None,
        intent: IntentConfig | None = None,
        history: list[SessionTurn] | None = None,
    ) -> tuple[dict[str, Any], EntityExtractionInfo]:
        # Whether extraction *should* run is the caller's decision, so that a
        # per-request override is not silently overruled by the global default.
        if not schema:
            return {}, EntityExtractionInfo(status="skipped", detail="intent declares no entities")
        if not self._client.configured:
            return {}, EntityExtractionInfo(
                status="unavailable", detail="no LLM provider configured"
            )

        system = build_system_prompt(domain, intent)
        user = build_user_prompt(text, schema, today, domain, history)
        try:
            raw = await self._client.complete_json(system, user)
        except LLMUnavailable as exc:
            logger.warning("entity extraction unavailable: %s", exc)
            return {}, EntityExtractionInfo(status="failed", detail=str(exc))
        except Exception as exc:
            logger.exception("entity extraction raised unexpectedly")
            return {}, EntityExtractionInfo(status="failed", detail=f"{type(exc).__name__}: {exc}")

        entities, rejected = validate(raw, schema)
        if rejected:
            logger.info("entity extraction dropped %d value(s): %s", len(rejected), rejected)
        model = (
            self._settings.openai_model if self.provider_name == "openai" else self.provider_name
        )
        return entities, EntityExtractionInfo(status="ok", model=model, rejected=rejected)
