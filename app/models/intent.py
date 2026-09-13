"""Intent: a named business operation within a domain."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.common import Status, name_key, new_id, now_ts

INTENT_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$"


class ToolRef(BaseModel):
    """The downstream capability an intent maps to.

    ``name``/``version`` is the caller's own label for it. ``mcp_tool_id``
    binds the intent to a registered MCP tool, which is what lets a
    classification report the exact MCP call that would satisfy it. It accepts
    a tool id, a qualified ``mcp__server__tool`` name, or ``server::tool``.
    """

    name: str = Field(default="", max_length=200)
    version: str = Field(default="v1", max_length=40)
    mcp_tool_id: str | None = Field(default=None, max_length=400)


class IntentBase(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=INTENT_NAME_PATTERN)
    description: str = Field(default="", max_length=2000)
    tool: ToolRef = Field(default_factory=ToolRef)
    entity_schema: dict[str, Any] = Field(default_factory=dict)
    status: Status = Status.ACTIVE
    #: Intent-specific guidance for the extractor, on top of the domain's.
    extraction_hints: str = Field(default="", max_length=2000)

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class IntentCreate(IntentBase):
    pass


class IntentUpdate(BaseModel):
    name: str | None = Field(
        default=None, min_length=1, max_length=120, pattern=INTENT_NAME_PATTERN
    )
    description: str | None = Field(default=None, max_length=2000)
    tool: ToolRef | None = None
    entity_schema: dict[str, Any] | None = None
    status: Status | None = None
    extraction_hints: str | None = Field(default=None, max_length=2000)


class IntentConfig(IntentBase):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id)
    domain_id: str
    created_at: float = Field(default_factory=now_ts)
    updated_at: float = Field(default_factory=now_ts)

    @property
    def name_key(self) -> str:
        return name_key(self.name)


class IntentRead(IntentConfig):
    example_count: int = 0


class IntentGenerateRequest(BaseModel):
    """Draft a domain's intents with the LLM.

    ``dry_run`` returns the proposal without writing anything, so auto mode can
    be previewed before it touches the catalogue.
    """

    brief: str | None = Field(default=None, max_length=2000)
    count: int = Field(default=5, ge=1, le=12)
    examples_per_intent: int = Field(default=6, ge=0, le=25)
    dry_run: bool = False


class GeneratedIntentRead(BaseModel):
    """One proposed intent, and what became of it."""

    name: str
    description: str = ""
    tool: ToolRef = Field(default_factory=ToolRef)
    entity_schema: dict[str, Any] = Field(default_factory=dict)
    extraction_hints: str = ""
    #: The registered MCP tool this intent was bound to, if any.
    mcp_tool: str | None = None
    examples: list[str] = Field(default_factory=list)
    intent_id: str | None = None
    created: bool = False


class IntentGenerateResponse(BaseModel):
    domain_id: str
    status: str
    detail: str | None = None
    provider: str = ""
    dry_run: bool = False
    intents: list[GeneratedIntentRead] = Field(default_factory=list)
    created_count: int = 0
    example_count: int = 0
    skipped: list[str] = Field(default_factory=list)


class ExampleGenerateRequest(BaseModel):
    count: int = Field(default=6, ge=1, le=25)
    dry_run: bool = False


class ExampleGenerateResponse(BaseModel):
    intent_id: str
    status: str
    detail: str | None = None
    provider: str = ""
    dry_run: bool = False
    examples: list[str] = Field(default_factory=list)
    created_count: int = 0
    skipped: list[str] = Field(default_factory=list)
