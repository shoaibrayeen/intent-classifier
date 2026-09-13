"""Persisting generated intents and examples.

Generation proposes; this writes. Kept as one service so the REST API and the
HTML UI take exactly the same path, rather than each re-implementing "skip what
already exists, then save the rest".
"""

from __future__ import annotations

import logging

from app.errors import ConflictError
from app.models.common import Status
from app.models.domain import DomainConfig
from app.models.example import ExampleCreate
from app.models.intent import (
    ExampleGenerateResponse,
    GeneratedIntentRead,
    IntentConfig,
    IntentCreate,
    IntentGenerateResponse,
    ToolRef,
)
from app.services.example_service import ExampleService
from app.services.intent_service import IntentService
from app.services.llm.intent_generator import IntentGenerator
from app.services.mcp_service import McpToolService

logger = logging.getLogger(__name__)


class IntentAuthoringService:
    def __init__(
        self,
        generator: IntentGenerator,
        intents: IntentService,
        examples: ExampleService,
        mcp: McpToolService | None = None,
    ) -> None:
        self._generator = generator
        self._intents = intents
        self._examples = examples
        self._mcp = mcp

    @property
    def available(self) -> bool:
        return self._generator.available

    @property
    def provider_name(self) -> str:
        return self._generator.provider_name

    async def generate_intents(
        self,
        domain: DomainConfig,
        brief: str = "",
        count: int = 5,
        examples_per_intent: int = 6,
        dry_run: bool = False,
    ) -> IntentGenerateResponse:
        existing = self._intents.list(domain.id)
        # Offer the registry so generated intents can be wired to real tools
        # straight away, rather than to a name nothing serves.
        offered = (
            [
                {
                    "qualified_name": tool.qualified_name,
                    "server": tool.server,
                    "name": tool.name,
                    "description": tool.description,
                }
                for tool in self._mcp.list()
                if tool.status == Status.ACTIVE
            ]
            if self._mcp
            else []
        )
        result = await self._generator.generate_intents(
            domain,
            brief=brief,
            count=count,
            examples_per_intent=examples_per_intent,
            existing_names=[i.name for i in existing],
            available_mcp_tools=offered,
        )
        response = IntentGenerateResponse(
            domain_id=domain.id,
            status=result.status,
            detail=result.detail,
            provider=result.provider,
            dry_run=dry_run,
            skipped=list(result.skipped),
        )
        if result.status != "ok":
            return response

        for proposal in result.intents:
            row = GeneratedIntentRead(**proposal.model_dump())
            if dry_run:
                response.intents.append(row)
                continue
            try:
                created = self._intents.create(
                    domain.id,
                    IntentCreate(
                        name=proposal.name,
                        description=proposal.description,
                        tool=ToolRef(
                            name=proposal.tool.name,
                            version=proposal.tool.version,
                            mcp_tool_id=proposal.mcp_tool,
                        ),
                        entity_schema=proposal.entity_schema,
                        extraction_hints=proposal.extraction_hints,
                    ),
                )
            except ConflictError as exc:
                # Another writer created it between listing and writing.
                response.skipped.append(f"{proposal.name}: {exc.message}")
                response.intents.append(row)
                continue

            row.intent_id = created.id
            row.created = True
            response.created_count += 1
            stored, notes = self._add_examples(domain.id, created, proposal.examples)
            row.examples = stored
            response.example_count += len(stored)
            response.skipped.extend(f"{created.name}: {note}" for note in notes)
            response.intents.append(row)

        return response

    async def generate_examples(
        self,
        domain: DomainConfig,
        intent: IntentConfig,
        count: int = 6,
        dry_run: bool = False,
    ) -> ExampleGenerateResponse:
        existing = [e.text for e in self._examples.list(domain.id, intent.id)]
        result = await self._generator.generate_examples(
            domain, intent, count=count, existing=existing
        )
        response = ExampleGenerateResponse(
            intent_id=intent.id,
            status=result.status,
            detail=result.detail,
            provider=result.provider,
            dry_run=dry_run,
            skipped=list(result.skipped),
            examples=list(result.examples),
        )
        if result.status != "ok" or dry_run:
            return response

        stored, notes = self._add_examples(domain.id, intent, result.examples)
        response.examples = stored
        response.created_count = len(stored)
        response.skipped.extend(notes)
        return response

    def _add_examples(
        self, domain_id: str, intent: IntentConfig, texts: list[str]
    ) -> tuple[list[str], list[str]]:
        """Write a batch, and report rather than raise when one is unusable.

        add_many rejects the whole batch if any example duplicates an existing
        one. Generated batches are expected to contain the occasional
        near-duplicate, so a single bad item must not discard the rest: the
        batch is retried without it.
        """
        if not texts:
            return [], []
        remaining = list(texts)
        notes: list[str] = []
        while remaining:
            try:
                created = self._examples.add_many(
                    domain_id, intent.id, [ExampleCreate(text=t) for t in remaining]
                )
                return [e.text for e in created], notes
            except ConflictError as exc:
                offender = _offending_text(exc.message, remaining)
                if offender is None:
                    notes.append(f"examples not saved: {exc.message}")
                    return [], notes
                notes.append(f"skipped duplicate example '{offender}'")
                remaining = [t for t in remaining if t != offender]
        return [], notes


def _offending_text(message: str, candidates: list[str]) -> str | None:
    """Recover which example a conflict was about, so the rest can still save."""
    return next((text for text in candidates if f"'{text}'" in message), None)
