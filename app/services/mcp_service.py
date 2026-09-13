"""The MCP tool registry, and mapping a classification onto an MCP call.

Registering a tool records what it is and what it accepts. Binding an intent to
one is what lets a classification report the exact call that would satisfy it.
This service never connects to an MCP server: the endpoint is recorded for the
caller, and invocation stays on their side of the line.
"""

from __future__ import annotations

import logging
from typing import Any

from app.errors import ConflictError, InvalidInputError, NotFoundError
from app.models.common import Status, now_ts
from app.models.intent import IntentConfig
from app.models.mcp import (
    McpCall,
    McpImportRequest,
    McpImportResponse,
    McpTool,
    McpToolCreate,
    McpToolRead,
    McpToolUpdate,
)
from app.repositories.chroma_mcp import ChromaMcpToolRepository

logger = logging.getLogger(__name__)

#: JSON Schema types that map onto the entity types intents declare.
_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object", "null"}


class McpToolService:
    def __init__(self, store, repo: ChromaMcpToolRepository) -> None:
        self._store = store
        self._repo = repo

    # ------------------------------------------------------------- registry
    def create(self, payload: McpToolCreate) -> McpTool:
        with self._store.lock:
            if self._repo.find_by_key(payload.server, payload.name) is not None:
                raise ConflictError(
                    f"MCP tool '{payload.name}' is already registered for server '{payload.server}'"
                )
            return self._repo.save(McpTool(**payload.model_dump()))

    def get(self, tool_id: str) -> McpTool:
        tool = self._repo.get(tool_id)
        if tool is None:
            raise NotFoundError(f"MCP tool '{tool_id}' not found")
        return tool

    def list(self, server: str | None = None) -> list[McpTool]:
        return self._repo.list(server)

    def update(self, tool_id: str, payload: McpToolUpdate) -> McpTool:
        with self._store.lock:
            tool = self.get(tool_id)
            changes = payload.model_dump(exclude_unset=True, exclude_none=True)
            server = changes.get("server", tool.server)
            name = changes.get("name", tool.name)
            if (server, name) != (tool.server, tool.name):
                clash = self._repo.find_by_key(server, name)
                if clash is not None and clash.id != tool_id:
                    raise ConflictError(f"MCP tool '{name}' already exists for server '{server}'")
            # Re-validate rather than model_copy, so a nested transport or
            # schema arriving as a plain value is checked before it is stored.
            updated = McpTool.model_validate(
                {**tool.model_dump(), **changes, "updated_at": now_ts()}
            )
            return self._repo.save(updated)

    def delete(self, tool_id: str) -> None:
        with self._store.lock:
            self.get(tool_id)
            self._repo.delete(tool_id)

    def resolve(self, reference: str) -> McpTool | None:
        """Find a tool by id, qualified name, or 'server::tool'."""
        if not reference:
            return None
        if tool := self._repo.get(reference):
            return tool
        text = reference.strip()
        if text.startswith("mcp__"):
            parts = text[5:].split("__", 1)
            if len(parts) == 2:
                return self._repo.find_by_key(parts[0], parts[1])
        for separator in ("::", "/", "."):
            if separator in text:
                server, _, name = text.partition(separator)
                if found := self._repo.find_by_key(server, name):
                    return found
        return None

    def read_all(self, bound_counts: dict[str, int] | None = None) -> list[McpToolRead]:
        counts = bound_counts or {}
        return [
            McpToolRead(
                **tool.model_dump(),
                qualified_name=tool.qualified_name,
                bound_intents=counts.get(tool.id, 0),
            )
            for tool in self._repo.list()
        ]

    # --------------------------------------------------------------- import
    def import_tools(self, payload: McpImportRequest) -> McpImportResponse:
        """Load a server's catalogue from an MCP tools/list payload."""
        entries = _extract_tools(payload.tools)
        if not entries:
            raise InvalidInputError(
                "no tools found: expected a tools/list result, its 'tools' array, "
                "or a JSON-RPC envelope containing one"
            )

        response = McpImportResponse(server=payload.server)
        seen: set[str] = set()
        with self._store.lock:
            for entry in entries:
                name = str(entry.get("name", "")).strip()
                if not name:
                    response.skipped.append("a tool had no name")
                    continue
                schema = entry.get("inputSchema") or entry.get("input_schema") or {}
                if not isinstance(schema, dict):
                    response.skipped.append(f"{name}: inputSchema was not an object")
                    schema = {}
                try:
                    fields = {
                        "server": payload.server,
                        "name": name,
                        "description": str(entry.get("description", ""))[:2000],
                        "input_schema": schema,
                        "transport": payload.transport,
                        "endpoint": payload.endpoint,
                    }
                    existing = self._repo.find_by_key(payload.server, name)
                    if existing is None:
                        self._repo.save(McpTool(**McpToolCreate(**fields).model_dump()))
                        response.created.append(name)
                    else:
                        self._repo.save(
                            McpTool.model_validate(
                                {**existing.model_dump(), **fields, "updated_at": now_ts()}
                            )
                        )
                        response.updated.append(name)
                    seen.add(name.casefold())
                except Exception as exc:
                    response.skipped.append(f"{name}: {type(exc).__name__}: {exc}")

            if payload.replace:
                for tool in self._repo.list(server=payload.server):
                    if tool.name.casefold() not in seen:
                        self._repo.delete(tool.id)
                        response.removed.append(tool.name)
        return response

    # ----------------------------------------------------------- resolution
    def build_call(self, intent: IntentConfig, entities: dict[str, Any]) -> McpCall | None:
        """Describe the MCP call this intent maps to. Nothing is invoked."""
        reference = (intent.tool.mcp_tool_id or "").strip()
        if not reference:
            return None

        tool = self.resolve(reference)
        if tool is None:
            # The binding outlived the tool. Say so rather than silently
            # dropping it: a caller waiting for a tool call needs to know why
            # there isn't one.
            return McpCall(
                tool_id=reference,
                server="",
                tool="",
                qualified_name=reference,
                unresolved=f"no registered MCP tool matches '{reference}'",
            )

        properties = tool.properties
        arguments = {k: v for k, v in entities.items() if k in properties} if properties else {}
        unmapped = sorted(set(entities) - set(arguments))
        missing = [name for name in tool.required if name not in arguments]
        disabled = tool.status != Status.ACTIVE

        return McpCall(
            tool_id=tool.id,
            server=tool.server,
            tool=tool.name,
            qualified_name=tool.qualified_name,
            description=tool.description,
            transport=tool.transport,
            endpoint=tool.endpoint,
            arguments=arguments,
            missing_required=missing,
            unmapped=unmapped,
            ready=not missing and not disabled,
            unresolved="this MCP tool is disabled" if disabled else None,
        )


def _extract_tools(payload: Any) -> list[dict]:
    """Pull the tool array out of whatever shape the caller pasted."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("tools", "result", "data"):
            if key in payload:
                found = _extract_tools(payload[key])
                if found:
                    return found
    return []


def schema_summary(tool: McpTool) -> list[dict[str, Any]]:
    """Flatten an input schema into rows a template can render."""
    required = set(tool.required)
    rows = []
    for name, spec in tool.properties.items():
        spec = spec if isinstance(spec, dict) else {}
        declared = str(spec.get("type", "string"))
        rows.append(
            {
                "name": name,
                "type": declared if declared in _JSON_TYPES else "string",
                "required": name in required,
                "description": str(spec.get("description", ""))[:200],
            }
        )
    return sorted(rows, key=lambda r: (not r["required"], r["name"]))
