"""Chroma-backed domain repository."""

from __future__ import annotations

from app.db.chroma import DUMMY_EMBEDDING, ChromaStore, clean_metadata
from app.models.common import name_key
from app.models.domain import DomainConfig


class ChromaDomainRepository:
    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self._col = store.domains

    @staticmethod
    def _metadata(domain: DomainConfig) -> dict[str, object]:
        return clean_metadata(
            {
                "name": domain.name,
                "name_key": domain.name_key,
                "status": str(domain.status),
                "created_at": domain.created_at,
                "updated_at": domain.updated_at,
            }
        )

    def get(self, domain_id: str) -> DomainConfig | None:
        with self._store.lock:
            result = self._col.get(ids=[domain_id], include=["documents"])
        documents = result["documents"] or []
        if not documents:
            return None
        return DomainConfig.model_validate_json(documents[0])

    def get_by_name(self, name: str) -> DomainConfig | None:
        with self._store.lock:
            result = self._col.get(where={"name_key": name_key(name)}, include=["documents"])
        documents = result["documents"] or []
        if not documents:
            return None
        return DomainConfig.model_validate_json(documents[0])

    def list(self) -> list[DomainConfig]:
        with self._store.lock:
            result = self._col.get(include=["documents"])
        domains = [DomainConfig.model_validate_json(doc) for doc in result["documents"] or []]
        return sorted(domains, key=lambda d: d.created_at)

    def save(self, domain: DomainConfig) -> DomainConfig:
        with self._store.lock:
            self._col.upsert(
                ids=[domain.id],
                embeddings=[DUMMY_EMBEDDING],
                documents=[domain.model_dump_json()],
                metadatas=[self._metadata(domain)],
            )
        return domain

    def delete(self, domain_id: str) -> None:
        with self._store.lock:
            self._col.delete(ids=[domain_id])
