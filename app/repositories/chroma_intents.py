"""Chroma-backed intent repository."""

from __future__ import annotations

from app.db.chroma import DUMMY_EMBEDDING, ChromaStore, clean_metadata
from app.models.common import name_key
from app.models.intent import IntentConfig


class ChromaIntentRepository:
    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self._col = store.intents

    @staticmethod
    def _metadata(intent: IntentConfig) -> dict[str, object]:
        return clean_metadata(
            {
                "domain_id": intent.domain_id,
                "name": intent.name,
                "name_key": intent.name_key,
                "status": str(intent.status),
                "tool_name": intent.tool.name,
                "created_at": intent.created_at,
                "updated_at": intent.updated_at,
            }
        )

    def _load(self, documents: list[str]) -> list[IntentConfig]:
        intents = [IntentConfig.model_validate_json(doc) for doc in documents]
        return sorted(intents, key=lambda i: i.created_at)

    def get(self, intent_id: str) -> IntentConfig | None:
        with self._store.lock:
            result = self._col.get(ids=[intent_id], include=["documents"])
        documents = result["documents"] or []
        if not documents:
            return None
        return IntentConfig.model_validate_json(documents[0])

    def get_by_name(self, domain_id: str, name: str) -> IntentConfig | None:
        where = {"$and": [{"domain_id": domain_id}, {"name_key": name_key(name)}]}
        with self._store.lock:
            result = self._col.get(where=where, include=["documents"])
        documents = result["documents"] or []
        if not documents:
            return None
        return IntentConfig.model_validate_json(documents[0])

    def list_for_domain(self, domain_id: str) -> list[IntentConfig]:
        with self._store.lock:
            result = self._col.get(where={"domain_id": domain_id}, include=["documents"])
        return self._load(result["documents"] or [])

    def list_all(self) -> list[IntentConfig]:
        with self._store.lock:
            result = self._col.get(include=["documents"])
        return self._load(result["documents"] or [])

    def save(self, intent: IntentConfig) -> IntentConfig:
        with self._store.lock:
            self._col.upsert(
                ids=[intent.id],
                embeddings=[DUMMY_EMBEDDING],
                documents=[intent.model_dump_json()],
                metadatas=[self._metadata(intent)],
            )
        return intent

    def delete(self, intent_id: str) -> None:
        with self._store.lock:
            self._col.delete(ids=[intent_id])

    def delete_for_domain(self, domain_id: str) -> None:
        with self._store.lock:
            ids = self._col.get(where={"domain_id": domain_id}, include=[])["ids"]
            if ids:
                self._col.delete(ids=ids)
