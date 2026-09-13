"""The seam where real MCP invocation plugs in.

Today the service resolves an intent to an MCP call and reports it; nothing is
dialled. That boundary is deliberate, but it should not be a wall: when an MCP
client is wired up, it implements this protocol and is handed to the container,
and the resolved call already carries everything an invocation needs (server,
tool, transport, endpoint, validated arguments).

Nothing in the classification path calls this. Execution stays a decision the
caller makes, because the caller owns the credentials and the blast radius.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from app.models.mcp import McpCall

logger = logging.getLogger(__name__)


class McpExecutionResult:
    """What an invocation returned, or why it did not happen."""

    def __init__(
        self,
        status: str,
        content: Any = None,
        detail: str | None = None,
        is_error: bool = False,
    ) -> None:
        self.status = status  # ok | refused | failed | unavailable
        self.content = content
        self.detail = detail
        self.is_error = is_error

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"McpExecutionResult(status={self.status!r}, is_error={self.is_error})"


class McpExecutor(Protocol):
    """What a real MCP client would have to provide."""

    @property
    def available(self) -> bool: ...

    async def call(self, call: McpCall) -> McpExecutionResult: ...


class DescribeOnlyExecutor:
    """The default: describes the call, refuses to make it.

    Swapping this for a real client is the whole integration. Because a call is
    only ever refused here, a caller that wires one in cannot be surprised by
    this service having already run something.
    """

    name = "describe-only"

    @property
    def available(self) -> bool:
        return False

    async def call(self, call: McpCall) -> McpExecutionResult:
        logger.info(
            "MCP call described but not made: %s (ready=%s)", call.qualified_name, call.ready
        )
        return McpExecutionResult(
            status="refused",
            detail=(
                "no MCP client is wired up; this service resolves and reports the "
                "call, and the caller invokes it"
            ),
        )
