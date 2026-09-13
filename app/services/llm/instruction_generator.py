"""Generate a domain's extraction instructions from its name and a short brief.

An administrator knows what their domain is about but not what a good
extraction prompt looks like. This turns "contract — track vendor agreements"
into a first draft of system and user instructions, which they then edit on the
domain page. The generated text is a starting point stored in the database,
never a hidden runtime behaviour.
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel

from app.models.domain import DomainConfig
from app.services.llm.client import LLMClient, LLMUnavailable

logger = logging.getLogger(__name__)

#: Matches the model fields' max_length.
MAX_INSTRUCTION_CHARS = 4000

GENERATOR_SYSTEM_PROMPT = (
    "You write configuration for an intent-classification service.\n"
    "Given a business domain's name and a short description of what its users "
    "want to do, write two instruction blocks for the service's entity "
    "extractor:\n"
    "1. system_instructions: BEGIN with the role the assistant should adopt "
    "for this domain, stated directly, e.g. 'You are acting as a contracts "
    "and legal analyst.' for contractual work, or 'You are acting as an "
    "e-commerce shopping assistant.' for carts, orders and products. Pick the "
    "role the domain actually implies. Then give the domain's vocabulary and "
    "conventions, what its key terms mean, and what must never be inferred or "
    "guessed.\n"
    "2. user_instructions: who the users are and how they phrase requests, so "
    "the extractor reads their wording correctly.\n"
    "Rules: return ONLY a JSON object with exactly the keys "
    "'system_instructions' and 'user_instructions'. Plain prose, no markdown, "
    "no lists. At most 120 words per block. Ground every statement in the "
    "provided name and description; do not invent facts about the business."
)


class GenerationResult(BaseModel):
    status: str  # ok | failed | unavailable
    detail: str | None = None
    provider: str = ""
    system_instructions: str = ""
    user_instructions: str = ""


class InstructionGenerator:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client.configured

    @property
    def provider_name(self) -> str:
        return getattr(self._client, "name", "unknown")

    async def generate(self, name: str, brief: str) -> GenerationResult:
        if not self._client.configured:
            return GenerationResult(
                status="unavailable",
                detail="no LLM provider configured (set OPENAI_API_KEY, or LLM_PROVIDER=mock)",
                provider=self.provider_name,
            )

        user = json.dumps(
            {
                "task": "generate_domain_instructions",
                "domain_name": name,
                "description": brief,
            },
            ensure_ascii=False,
        )
        try:
            raw = await self._client.complete_json(GENERATOR_SYSTEM_PROMPT, user)
        except LLMUnavailable as exc:
            logger.warning("instruction generation unavailable: %s", exc)
            return GenerationResult(status="failed", detail=str(exc), provider=self.provider_name)
        except Exception as exc:
            logger.exception("instruction generation raised unexpectedly")
            return GenerationResult(
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
                provider=self.provider_name,
            )

        system = _clean(raw.get("system_instructions"))
        user_block = _clean(raw.get("user_instructions"))
        if not system and not user_block:
            return GenerationResult(
                status="failed",
                detail="provider returned no usable instructions",
                provider=self.provider_name,
            )
        return GenerationResult(
            status="ok",
            provider=self.provider_name,
            system_instructions=system,
            user_instructions=user_block,
        )


def _clean(value: object) -> str:
    """Model output is untrusted: coerce, trim, and cap to the field limit."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:MAX_INSTRUCTION_CHARS]


def brief_for(domain: DomainConfig, override: str | None = None) -> str:
    return (override or "").strip() or domain.description.strip()
