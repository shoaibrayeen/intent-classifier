"""Draft a domain's intents, and a single intent's examples, with an LLM.

Auto mode is a drafting aid, not an authority. Everything it produces is
ordinary configuration: saved to the same collections as hand-written intents
and editable through the same endpoints and screens afterwards.

Model output is untrusted input. Names are sanitised to the pattern the intent
model enforces, entity schemas are filtered to declared types, examples are
deduplicated against what the intent already has, and anything unusable is
skipped and reported rather than written.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.models.domain import DomainConfig
from app.models.intent import INTENT_NAME_PATTERN, IntentConfig, ToolRef
from app.services.entity_schema import EntityType
from app.services.llm.client import LLMClient, LLMUnavailable
from app.services.normalizer import normalize_text

logger = logging.getLogger(__name__)

MAX_INTENTS = 12
MAX_EXAMPLES_PER_INTENT = 25
MAX_NAME_CHARS = 120
MAX_DESCRIPTION_CHARS = 2000
MAX_HINT_CHARS = 2000
MAX_EXAMPLE_CHARS = 2000

_NAME_OK = re.compile(INTENT_NAME_PATTERN)
_NAME_CLEAN = re.compile(r"[^A-Za-z0-9_.\-]+")
_TOOL_CLEAN = re.compile(r"[^a-z0-9_]+")
_KNOWN_TYPES = {t.value for t in EntityType}

INTENTS_SYSTEM_PROMPT = (
    "You design the intent catalogue for a domain-aware intent classifier.\n"
    "Given a domain and what its users do, propose the distinct operations "
    "users ask for in that domain.\n"
    "Return ONLY a JSON object of the form:\n"
    '{"intents": [{"name": "UPPER_SNAKE_CASE", "description": "one sentence", '
    '"tool": "snake_case_function_name", "entity_schema": {"field": '
    '{"type": "string|integer|number|boolean|date|enum|array", "required": '
    'true|false, "values": ["only", "for", "enum"]}}, "extraction_hints": '
    '"short note or empty", "mcp_tool": "mcp__server__tool or empty", '
    '"examples": ["natural phrasing", "..."]}]}\n'
    "Rules:\n"
    "1. Each intent is one distinct operation. Do not create near-duplicates.\n"
    "2. Names are UPPER_SNAKE_CASE, prefixed with the domain, e.g. "
    "CONTRACT_SEARCH.\n"
    "3. entity_schema lists only what that operation genuinely needs as "
    "arguments. An intent may have none.\n"
    "4. examples are things a real user would type, varied in wording and "
    "length, all meaning that one intent. Do not number them.\n"
    "5. Ground everything in the domain provided. Do not invent unrelated "
    "operations.\n"
    "6. If available_mcp_tools is given, set mcp_tool to the one that would "
    "serve the intent, copied exactly from that list. Leave it empty when none "
    "fits. Never invent a tool that is not listed."
)

EXAMPLES_SYSTEM_PROMPT = (
    "You write training examples for one intent of an intent classifier.\n"
    'Return ONLY a JSON object of the form {"examples": ["...", "..."]}.\n'
    "Rules:\n"
    "1. Every example must mean this one intent, not a neighbouring one.\n"
    "2. Vary the wording, length and formality. Include both terse and full "
    "sentence phrasings.\n"
    "3. Write what a real user would type. No numbering, no quotes around "
    "individual items, no explanations.\n"
    "4. Do not repeat any example already listed as existing."
)


class GeneratedIntent(BaseModel):
    """One proposed intent, already validated against the intent model."""

    name: str
    description: str = ""
    tool: ToolRef = Field(default_factory=ToolRef)
    entity_schema: dict[str, Any] = Field(default_factory=dict)
    extraction_hints: str = ""
    #: Qualified name of a registered MCP tool, when one was offered and chosen.
    mcp_tool: str | None = None
    examples: list[str] = Field(default_factory=list)


class IntentGenerationResult(BaseModel):
    status: str  # ok | failed | unavailable
    detail: str | None = None
    provider: str = ""
    intents: list[GeneratedIntent] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class ExampleGenerationResult(BaseModel):
    status: str
    detail: str | None = None
    provider: str = ""
    examples: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class IntentGenerator:
    def __init__(self, client: LLMClient) -> None:
        self._client = client

    @property
    def available(self) -> bool:
        return self._client.configured

    @property
    def provider_name(self) -> str:
        return getattr(self._client, "name", "unknown")

    def _unavailable(self) -> str:
        return "no LLM provider configured (set OPENAI_API_KEY, or LLM_PROVIDER=mock)"

    async def generate_intents(
        self,
        domain: DomainConfig,
        brief: str = "",
        count: int = 5,
        examples_per_intent: int = 6,
        existing_names: list[str] | None = None,
        available_mcp_tools: list[dict[str, str]] | None = None,
    ) -> IntentGenerationResult:
        if not self._client.configured:
            return IntentGenerationResult(
                status="unavailable", detail=self._unavailable(), provider=self.provider_name
            )

        count = max(1, min(count, MAX_INTENTS))
        per_intent = max(0, min(examples_per_intent, MAX_EXAMPLES_PER_INTENT))
        user = json.dumps(
            {
                "task": "generate_domain_intents",
                "domain_name": domain.name,
                "description": (brief or domain.description).strip(),
                "domain_instructions": domain.system_instructions.strip(),
                "intent_count": count,
                "examples_per_intent": per_intent,
                "existing_intents": existing_names or [],
                "available_mcp_tools": available_mcp_tools or [],
            },
            ensure_ascii=False,
        )

        try:
            raw = await self._client.complete_json(INTENTS_SYSTEM_PROMPT, user)
        except LLMUnavailable as exc:
            logger.warning("intent generation unavailable: %s", exc)
            return IntentGenerationResult(
                status="failed", detail=str(exc), provider=self.provider_name
            )
        except Exception as exc:
            logger.exception("intent generation raised unexpectedly")
            return IntentGenerationResult(
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
                provider=self.provider_name,
            )

        proposed = raw.get("intents")
        if not isinstance(proposed, list) or not proposed:
            return IntentGenerationResult(
                status="failed",
                detail="provider returned no intents",
                provider=self.provider_name,
            )

        intents: list[GeneratedIntent] = []
        skipped: list[str] = []
        seen: set[str] = {n.casefold() for n in (existing_names or [])}

        for item in proposed[:count]:
            if not isinstance(item, dict):
                skipped.append("a proposed intent was not an object")
                continue
            name = _clean_intent_name(item.get("name"))
            if not name:
                skipped.append(f"unusable intent name: {item.get('name')!r}")
                continue
            if name.casefold() in seen:
                skipped.append(f"{name}: already exists in this domain")
                continue
            seen.add(name.casefold())

            schema, schema_notes = _clean_entity_schema(item.get("entity_schema"))
            skipped.extend(f"{name}.{note}" for note in schema_notes)

            # A tool the model invented cannot be bound: only names actually on
            # offer are accepted, and anything else is reported.
            offered = {t.get("qualified_name", "") for t in (available_mcp_tools or [])}
            proposed_tool = item.get("mcp_tool")
            mcp_tool = None
            if isinstance(proposed_tool, str) and proposed_tool.strip():
                candidate = proposed_tool.strip()
                if candidate in offered:
                    mcp_tool = candidate
                else:
                    skipped.append(f"{name}: MCP tool '{candidate}' is not registered")

            intents.append(
                GeneratedIntent(
                    name=name,
                    description=_clean_text(item.get("description"), MAX_DESCRIPTION_CHARS),
                    tool=ToolRef(name=_clean_tool_name(item.get("tool"), name)),
                    entity_schema=schema,
                    extraction_hints=_clean_text(item.get("extraction_hints"), MAX_HINT_CHARS),
                    mcp_tool=mcp_tool,
                    examples=_clean_examples(item.get("examples"), per_intent)[0],
                )
            )

        if not intents:
            return IntentGenerationResult(
                status="failed",
                detail="no usable intents in the response",
                provider=self.provider_name,
                skipped=skipped,
            )
        return IntentGenerationResult(
            status="ok", provider=self.provider_name, intents=intents, skipped=skipped
        )

    async def generate_examples(
        self,
        domain: DomainConfig,
        intent: IntentConfig,
        count: int = 6,
        existing: list[str] | None = None,
    ) -> ExampleGenerationResult:
        if not self._client.configured:
            return ExampleGenerationResult(
                status="unavailable", detail=self._unavailable(), provider=self.provider_name
            )

        count = max(1, min(count, MAX_EXAMPLES_PER_INTENT))
        existing = existing or []
        user = json.dumps(
            {
                "task": "generate_intent_examples",
                "domain_name": domain.name,
                "domain_description": domain.description.strip(),
                "intent_name": intent.name,
                "intent_description": intent.description.strip(),
                "extraction_hints": intent.extraction_hints.strip(),
                "entity_schema": intent.entity_schema,
                "example_count": count,
                "existing_examples": existing,
            },
            ensure_ascii=False,
        )

        try:
            raw = await self._client.complete_json(EXAMPLES_SYSTEM_PROMPT, user)
        except LLMUnavailable as exc:
            logger.warning("example generation unavailable: %s", exc)
            return ExampleGenerationResult(
                status="failed", detail=str(exc), provider=self.provider_name
            )
        except Exception as exc:
            logger.exception("example generation raised unexpectedly")
            return ExampleGenerationResult(
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
                provider=self.provider_name,
            )

        examples, skipped = _clean_examples(raw.get("examples"), count, existing)
        if not examples:
            return ExampleGenerationResult(
                status="failed",
                detail="no usable examples in the response",
                provider=self.provider_name,
                skipped=skipped,
            )
        return ExampleGenerationResult(
            status="ok", provider=self.provider_name, examples=examples, skipped=skipped
        )


# --------------------------------------------------------------------- cleaning


def _clean_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _clean_intent_name(value: object) -> str:
    """Coerce a proposed name to something the intent model will accept."""
    if not isinstance(value, str):
        return ""
    name = _NAME_CLEAN.sub("_", value.strip().upper()).strip("_")
    name = re.sub(r"_{2,}", "_", name)[:MAX_NAME_CHARS]
    return name if name and _NAME_OK.match(name) else ""


def _clean_tool_name(value: object, intent_name: str) -> str:
    """A tool name, derived from the intent when the model gives nothing usable."""
    if isinstance(value, str) and value.strip():
        tool = _TOOL_CLEAN.sub("_", value.strip().casefold()).strip("_")
        if tool:
            return tool[:200]
    return _TOOL_CLEAN.sub("_", intent_name.casefold()).strip("_")[:200]


def _clean_entity_schema(value: object) -> tuple[dict[str, Any], list[str]]:
    """Keep only fields with a declared, known type."""
    if value in (None, ""):
        return {}, []
    if not isinstance(value, dict):
        return {}, ["entity_schema was not an object"]

    schema: dict[str, Any] = {}
    notes: list[str] = []
    for field, spec in value.items():
        if not isinstance(field, str) or not field.strip():
            notes.append("dropped a field with no name")
            continue
        key = _NAME_CLEAN.sub("_", field.strip().casefold()).strip("_")
        if not key:
            notes.append(f"dropped unusable field name {field!r}")
            continue

        if isinstance(spec, str):
            spec = {"type": spec}
        if not isinstance(spec, dict):
            notes.append(f"dropped {key}: specification was not an object")
            continue

        declared = str(spec.get("type", "string")).strip().casefold()
        if declared not in _KNOWN_TYPES:
            notes.append(f"dropped {key}: unknown type {declared!r}")
            continue

        cleaned: dict[str, Any] = {"type": declared}
        if spec.get("required") is True:
            cleaned["required"] = True
        if declared == EntityType.ENUM:
            values = [str(v).strip() for v in spec.get("values", []) if str(v).strip()]
            if not values:
                notes.append(f"dropped {key}: enum with no values")
                continue
            cleaned["values"] = values
        if declared == EntityType.ARRAY:
            item_type = str((spec.get("items") or {}).get("type", "string")).casefold()
            cleaned["items"] = {"type": item_type if item_type in _KNOWN_TYPES else "string"}
        schema[key] = cleaned
    return schema, notes


def _clean_examples(
    value: object, limit: int, existing: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Trim, deduplicate, and drop anything that is not usable text.

    Deduplication uses the same normalisation the index does, so a generated
    example that only differs by case or spacing is treated as the duplicate it
    would become once stored.
    """
    if limit <= 0 or not isinstance(value, list):
        return [], [] if limit <= 0 else ["examples was not a list"]

    seen = {normalize_text(text) for text in (existing or [])}
    kept: list[str] = []
    notes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            notes.append("dropped a non-text example")
            continue
        text = " ".join(item.split())[:MAX_EXAMPLE_CHARS]
        if not text:
            continue
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            notes.append(f"dropped duplicate example: '{text}'")
            continue
        seen.add(normalized)
        kept.append(text)
        if len(kept) >= limit:
            break
    return kept, notes
