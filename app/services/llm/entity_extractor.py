"""Entity extraction (Phase 3 placeholder).

Classification never depends on the LLM. When ENTITY_EXTRACTION_ENABLED is off
-- the default -- this returns an empty mapping and no network call is made.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings
from app.services.llm.client import OpenAIChatClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You extract structured entities from a user request. "
    "Return a JSON object whose keys are exactly the requested entity names. "
    "Omit any entity that is not present in the request. Do not invent values."
)


class EntityExtractor:
    def __init__(self, settings: Settings, client: OpenAIChatClient) -> None:
        self._settings = settings
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._settings.entity_extraction_enabled and self._client.configured

    async def extract(self, text: str, entity_schema: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or not entity_schema:
            return {}
        user = json.dumps({"request": text, "entities": entity_schema}, ensure_ascii=False)
        try:
            return await self._client.complete_json(SYSTEM_PROMPT, user)
        except Exception:
            logger.exception("entity extraction failed; returning no entities")
            return {}
