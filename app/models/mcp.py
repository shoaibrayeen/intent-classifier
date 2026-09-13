"""MCP tools: the downstream capabilities an intent can be wired to.

A tool is identified the way Model Context Protocol clients identify one: a
server, and a tool name within that server. The registry records what the tool
is and what arguments it accepts, so a classification can report the exact call
that would satisfy it.

Nothing here executes anything. The endpoint and transport are recorded so the
caller knows where the tool lives; this service never connects to it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from app.models.common import Status, name_key, new_id, now_ts

SERVER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$"
TOOL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$"


class Transport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    UNKNOWN = "unknown"


def qualified_name(server: str, tool: str) -> str:
    """The conventional client-side spelling of an MCP tool."""
    return f"mcp__{server}__{tool}"


class McpToolBase(BaseModel):
    server: str = Field(min_length=1, max_length=120, pattern=SERVER_PATTERN)
    name: str = Field(min_length=1, max_length=200, pattern=TOOL_PATTERN)
    description: str = Field(default="", max_length=2000)
    #: The tool's JSON Schema, as MCP reports it in tools/list.
    input_schema: dict[str, Any] = Field(default_factory=dict)
    transport: Transport = Transport.UNKNOWN
    #: Command line or URL. Recorded for the caller; never dialled here.
    endpoint: str = Field(default="", max_length=1000)
    status: Status = Status.ACTIVE

    @field_validator("server", "name", "description", "endpoint", mode="before")
    @classmethod
    def _strip(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v


class McpToolCreate(McpToolBase):
    pass


class McpToolUpdate(BaseModel):
    server: str | None = Field(default=None, min_length=1, max_length=120, pattern=SERVER_PATTERN)
    name: str | None = Field(default=None, min_length=1, max_length=200, pattern=TOOL_PATTERN)
    description: str | None = Field(default=None, max_length=2000)
    input_schema: dict[str, Any] | None = None
    transport: Transport | None = None
    endpoint: str | None = Field(default=None, max_length=1000)
    status: Status | None = None


class McpTool(McpToolBase):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(default_factory=new_id)
    created_at: float = Field(default_factory=now_ts)
    updated_at: float = Field(default_factory=now_ts)

    @computed_field
    @property
    def qualified_name(self) -> str:
        """Serialized too: clients match on this, so it must be in the JSON."""
        return qualified_name(self.server, self.name)

    @property
    def key(self) -> str:
        return f"{name_key(self.server)}::{name_key(self.name)}"

    @property
    def properties(self) -> dict[str, Any]:
        props = self.input_schema.get("properties")
        return props if isinstance(props, dict) else {}

    @property
    def required(self) -> list[str]:
        required = self.input_schema.get("required")
        return [str(r) for r in required] if isinstance(required, list) else []


class McpToolRead(McpTool):
    bound_intents: int = 0

    @classmethod
    def of(cls, tool: McpTool, bound_intents: int = 0) -> McpToolRead:
        # qualified_name is computed, so it must not be passed back in.
        return cls(**tool.model_dump(exclude={"qualified_name"}), bound_intents=bound_intents)


class McpImportRequest(BaseModel):
    """Load a server's catalogue from an MCP ``tools/list`` response.

    Accepts the raw result object, the bare list, or a JSON-RPC envelope, since
    all three are what people actually have to hand.
    """

    server: str = Field(min_length=1, max_length=120, pattern=SERVER_PATTERN)
    transport: Transport = Transport.UNKNOWN
    endpoint: str = Field(default="", max_length=1000)
    tools: Any = Field(description="tools/list result, its 'tools' array, or a JSON-RPC envelope")
    replace: bool = Field(
        default=False, description="Remove this server's other tools that the payload omits"
    )


class McpImportResponse(BaseModel):
    server: str
    created: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)


class McpCall(BaseModel):
    """The MCP invocation a classification resolves to. Never performed here."""

    tool_id: str
    server: str
    tool: str
    qualified_name: str
    description: str = ""
    transport: Transport = Transport.UNKNOWN
    endpoint: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    #: Entities the intent produced that this tool's schema does not accept.
    unmapped: list[str] = Field(default_factory=list)
    ready: bool = False
    #: Set when the intent names a tool the registry cannot resolve.
    unresolved: str | None = None
