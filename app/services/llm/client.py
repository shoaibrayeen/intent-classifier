"""LLM abstraction.

Only the OpenAI Chat Completions API is implemented, and nothing in this phase
calls it: entity extraction is a later phase. The protocol exists so that phase
plugs in without touching the retrieval engine.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    async def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


class OpenAIChatClient:
    """Thin wrapper over the Chat Completions API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.openai_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or "not-configured",
            base_url=settings.openai_base_url,
        )

    @property
    def configured(self) -> bool:
        return bool(self._settings.openai_api_key)

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON content")
            return {}
        return parsed if isinstance(parsed, dict) else {}
