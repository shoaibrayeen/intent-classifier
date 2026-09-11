"""Example CRUD. Every mutation re-embeds and rebuilds the domain's BM25 index."""

from __future__ import annotations

from app.db.chroma import ChromaStore
from app.errors import ConflictError, NotFoundError
from app.models.example import Example, ExampleCreate, text_hash
from app.repositories.base import ExampleRepository
from app.services.embedding import Embedder
from app.services.index_manager import IndexManager
from app.services.intent_service import IntentService
from app.services.normalizer import normalize_text


class ExampleService:
    def __init__(
        self,
        store: ChromaStore,
        examples: ExampleRepository,
        intents: IntentService,
        embedder: Embedder,
        index_manager: IndexManager,
    ) -> None:
        self._store = store
        self._examples = examples
        self._intents = intents
        self._embedder = embedder
        self._indexes = index_manager

    def list(self, domain_id: str, intent_id: str) -> list[Example]:
        self._intents.get(domain_id, intent_id)
        return self._examples.list_for_intent(intent_id)

    def add(self, domain_id: str, intent_id: str, payload: ExampleCreate) -> Example:
        return self.add_many(domain_id, intent_id, [payload])[0]

    def add_many(
        self, domain_id: str, intent_id: str, payloads: list[ExampleCreate]
    ) -> list[Example]:
        """Add a batch, embedding once and rebuilding the index once."""
        intent = self._intents.get(domain_id, intent_id)
        with self._store.lock:
            created: list[Example] = []
            texts: list[str] = []
            seen: set[str] = set()
            for payload in payloads:
                normalized = normalize_text(payload.text)
                digest = text_hash(normalized)
                if digest in seen:
                    raise ConflictError(f"duplicate example in request: '{payload.text}'")
                if self._examples.find_by_hash(intent_id, digest) is not None:
                    raise ConflictError(f"example already exists for this intent: '{payload.text}'")
                seen.add(digest)
                created.append(
                    Example(
                        domain_id=domain_id,
                        intent_id=intent_id,
                        intent_name=intent.name,
                        text=payload.text,
                        text_hash=digest,
                    )
                )
                texts.append(normalized)

            embeddings = self._embedder.embed_many(texts)
            self._examples.add_many(created, embeddings)
            self._indexes.rebuild(domain_id)
            return created

    def delete(self, domain_id: str, intent_id: str, example_id: str) -> None:
        self._intents.get(domain_id, intent_id)
        with self._store.lock:
            example = self._examples.get(example_id)
            if example is None or example.intent_id != intent_id:
                raise NotFoundError(f"example '{example_id}' not found")
            self._examples.delete(example_id)
            self._indexes.rebuild(domain_id)
