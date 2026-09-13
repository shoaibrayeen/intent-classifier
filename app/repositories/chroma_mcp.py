"""Chroma-backed MCP tool registry.

The JSON Schema lives in the document because Chroma metadata cannot nest;
server, name and status stay flat so they can be filtered on.
"""

from __future__ import annotations

import logging

from app.db.chroma import DUMMY_EMBEDDING, ChromaStore, clean_metadata
from app.models.common import Status, name_key
from app.models.mcp import McpTool

logger = logging.getLogger(__name__)


class ChromaMcpToolRepository:
    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self._col = store.mcp_tools

    @staticmethod
    def _to_tool(document: str) -> McpTool | None:
        try:
            return McpTool.model_validate_json(document)
        except Exception:
            logger.exception("skipping an unreadable MCP tool record")
            return None

    def _metadata(self, tool: McpTool) -> dict:
        return clean_metadata(
            {
                "server": tool.server,
                "server_key": name_key(tool.server),
                "name": tool.name,
                "name_key": name_key(tool.name),
                "key": tool.key,
                "qualified_name": tool.qualified_name,
                "status": str(tool.status),
                "created_at": tool.created_at,
                "updated_at": tool.updated_at,
            }
        )

    def save(self, tool: McpTool) -> McpTool:
        with self._store.lock:
            self._col.upsert(
                ids=[tool.id],
                embeddings=[DUMMY_EMBEDDING],
                documents=[tool.model_dump_json()],
                metadatas=[self._metadata(tool)],
            )
        return tool

    def get(self, tool_id: str) -> McpTool | None:
        with self._store.lock:
            result = self._col.get(ids=[tool_id], include=["documents"])
        documents = result.get("documents") or []
        return self._to_tool(documents[0]) if documents else None

    def find_by_key(self, server: str, name: str) -> McpTool | None:
        where = {"$and": [{"server_key": name_key(server)}, {"name_key": name_key(name)}]}
        with self._store.lock:
            result = self._col.get(where=where, include=["documents"])
        documents = result.get("documents") or []
        return self._to_tool(documents[0]) if documents else None

    def list(self, server: str | None = None, active_only: bool = False) -> list[McpTool]:
        clauses = []
        if server:
            clauses.append({"server_key": name_key(server)})
        if active_only:
            clauses.append({"status": str(Status.ACTIVE)})
        where = None
        if len(clauses) == 1:
            where = clauses[0]
        elif clauses:
            where = {"$and": clauses}

        with self._store.lock:
            result = self._col.get(where=where, include=["documents"])
        tools = [t for t in (self._to_tool(d) for d in result.get("documents") or []) if t]
        tools.sort(key=lambda t: (t.server.casefold(), t.name.casefold()))
        return tools

    def delete(self, tool_id: str) -> None:
        with self._store.lock:
            self._col.delete(ids=[tool_id])
