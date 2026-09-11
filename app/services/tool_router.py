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


class ToolRouter:
    def route(self, intent: IntentConfig, entities: dict[str, Any]) -> ToolCall | None:
        """Build the call an intent maps to, or None when it maps to nothing."""
        if not intent.tool.name:
            return None

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
            ready=not missing,
        )
