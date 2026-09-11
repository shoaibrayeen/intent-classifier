"""BM25 index status reporting."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class IndexState(StrEnum):
    NOT_BUILT = "not_built"
    DIRTY = "dirty"
    READY = "ready"


class IndexStatus(BaseModel):
    domain_id: str
    state: IndexState
    version: int = 0
    built_at: str | None = None
    bm25_doc_count: int = 0
    dense_count: int = 0
    tokenizer_version: str = "simple_v1"
