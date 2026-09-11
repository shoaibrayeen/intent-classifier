"""LLM abstraction.

The classification path never calls an LLM. This client exists for entity
extraction, where generation genuinely is the right tool. Keeping it behind a
protocol means a different provider is a constructor change, not a rewrite.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """The provider could not be reached, or returned nothing usable."""


class LLMClient(Protocol):
    """What the extractor needs from a provider."""

    @property
    def configured(self) -> bool: ...

    async def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating code fences."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMUnavailable("response contained no JSON object") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMUnavailable("response was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMUnavailable("response was not a JSON object")
    return parsed


class OpenAIChatClient:
    """OpenAI Chat Completions. Also works against any compatible endpoint
    through ``OPENAI_BASE_URL``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = settings.openai_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or "not-configured",
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    @property
    def configured(self) -> bool:
        return bool(self._settings.openai_api_key)

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.configured:
            raise LLMUnavailable("OPENAI_API_KEY is not set")
        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

        if not completion.choices:
            raise LLMUnavailable("provider returned no choices")
        return parse_json_object(completion.choices[0].message.content or "")


class StubLLMClient:
    """Never configured, never called. Used when no provider is set up."""

    @property
    def configured(self) -> bool:
        return False

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        raise LLMUnavailable("no LLM provider configured")


class ScriptedLLMClient:
    """A deterministic client for tests and demos.

    ``responses`` maps a substring of the user message to the object to return.
    Anything else returns ``default``. Raises whatever is in ``raises``.
    """

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | None = None,
        default: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.responses = responses or {}
        self.default = default if default is not None else {}
        self.raises = raises
        self.calls: list[tuple[str, str]] = []

    @property
    def configured(self) -> bool:
        return True

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        for needle, response in self.responses.items():
            if needle.casefold() in user.casefold():
                return response
        return self.default
