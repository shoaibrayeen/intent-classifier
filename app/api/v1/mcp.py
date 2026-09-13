"""MCP tool registry.

Registering a tool records what it is and what it accepts, so a classification
can report the exact MCP call that would satisfy an intent. Nothing here
connects to an MCP server: invocation stays with the caller.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.dependencies import get_container, require_read, require_write
from app.models.mcp import (
    McpImportRequest,
    McpImportResponse,
    McpTool,
    McpToolCreate,
    McpToolRead,
    McpToolUpdate,
)
from app.security.auth import Principal
from app.services.container import Container

router = APIRouter(prefix="/mcp/tools", tags=["mcp"])


def _bound_counts(container: Container) -> dict[str, int]:
    """How many intents point at each tool, across every domain."""
    counts: dict[str, int] = {}
    for domain in container.domains.list():
        for intent in container.intents.list(domain.id):
            reference = (intent.tool.mcp_tool_id or "").strip()
            if not reference:
                continue
            tool = container.mcp_tools.resolve(reference)
            if tool is not None:
                counts[tool.id] = counts.get(tool.id, 0) + 1
    return counts


@router.post("", response_model=McpTool, status_code=status.HTTP_201_CREATED)
def create_tool(
    payload: McpToolCreate,
    request: Request,
    principal: Principal = Depends(require_write),
    container: Container = Depends(get_container),
) -> McpTool:
    tool = container.mcp_tools.create(payload)
    container.audit.record(action="mcp.create", resource=f"mcp:{tool.qualified_name}")
    return tool


@router.get("", response_model=list[McpToolRead])
def list_tools(
    request: Request,
    server: str | None = None,
    principal: Principal = Depends(require_read),
    container: Container = Depends(get_container),
) -> list[McpToolRead]:
    counts = _bound_counts(container)
    tools = container.mcp_tools.list(server)
    return [McpToolRead.of(t, counts.get(t.id, 0)) for t in tools]


@router.get("/{tool_id}", response_model=McpToolRead)
def get_tool(
    tool_id: str,
    principal: Principal = Depends(require_read),
    container: Container = Depends(get_container),
) -> McpToolRead:
    tool = container.mcp_tools.get(tool_id)
    return McpToolRead.of(tool, _bound_counts(container).get(tool.id, 0))


@router.put("/{tool_id}", response_model=McpTool)
def update_tool(
    tool_id: str,
    payload: McpToolUpdate,
    request: Request,
    principal: Principal = Depends(require_write),
    container: Container = Depends(get_container),
) -> McpTool:
    tool = container.mcp_tools.update(tool_id, payload)
    container.audit.record(action="mcp.update", resource=f"mcp:{tool.qualified_name}")
    return tool


@router.delete("/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tool(
    tool_id: str,
    request: Request,
    principal: Principal = Depends(require_write),
    container: Container = Depends(get_container),
) -> None:
    """Deleting a tool leaves bound intents pointing at nothing.

    Those classifications then report the binding as unresolved rather than
    silently losing their tool call, which is the honest failure mode.
    """
    tool = container.mcp_tools.get(tool_id)
    container.mcp_tools.delete(tool_id)
    container.audit.record(action="mcp.delete", resource=f"mcp:{tool.qualified_name}")


@router.post("/import", response_model=McpImportResponse)
def import_tools(
    payload: McpImportRequest,
    request: Request,
    principal: Principal = Depends(require_write),
    container: Container = Depends(get_container),
) -> McpImportResponse:
    """Load a server's catalogue from an MCP ``tools/list`` response.

    Accepts the result object, its bare ``tools`` array, or a JSON-RPC
    envelope, because all three are what people have to hand. Existing tools
    for the server are updated in place, so re-importing after a server change
    is safe.
    """
    result = container.mcp_tools.import_tools(payload)
    container.audit.record(
        action="mcp.import",
        resource=f"mcp:{payload.server}",
        created=len(result.created),
        updated=len(result.updated),
        removed=len(result.removed),
    )
    return result
