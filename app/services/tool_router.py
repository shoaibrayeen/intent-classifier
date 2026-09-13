"""Intent to tool mapping.

This layer decides *what* should be called and with which arguments. It never
calls anything. Execution belongs to the caller, who owns the credentials, the
retry policy and the blast radius; a classifier that also invokes tools cannot
be safely reused by a UI, an API and an agent at once.
"""

from __future__ import annotations

from typing import Any

from app.models.classification import ToolCall
from app.models.intent import IntentConfig
from app.services.entity_schema import missing_required
from app.services.mcp_service import McpToolService


class ToolRouter:
    def __init__(self, mcp: McpToolService | None = None) -> None:
        self._mcp = mcp

    def route(self, intent: IntentConfig, entities: dict[str, Any]) -> ToolCall | None:
        """Build the call an intent maps to, or None when it maps to nothing.

        An intent bound to a registered MCP tool also carries the resolved MCP
        invocation, so the caller can see precisely which tool would run and
        with which arguments.
        """
        mcp_call = self._mcp.build_call(intent, entities) if self._mcp else None
        if not intent.tool.name:
            # An intent may be wired to MCP and nothing else; that is still a
            # call worth reporting.
            if mcp_call is None:
                return None
            return ToolCall(
                name=mcp_call.qualified_name,
                version=intent.tool.version,
                arguments=mcp_call.arguments,
                missing_required=mcp_call.missing_required,
                ready=mcp_call.ready,
                mcp=mcp_call,
            )

        schema = intent.entity_schema or {}
        # Entities have already been validated against the schema, but an intent
        # can be edited after extraction ran, so filter again rather than trust.
        arguments = {name: value for name, value in entities.items() if name in schema}
        missing = missing_required(arguments, schema)

        return ToolCall(
            name=intent.tool.name,
            version=intent.tool.version,
            arguments=arguments,
            missing_required=missing,
            ready=not missing and (mcp_call is None or mcp_call.ready),
            mcp=mcp_call,
        )
