"""Intent CRUD, including the index invalidation every mutation implies."""

from __future__ import annotations

from app.db.chroma import ChromaStore
from app.errors import ConflictError, NotFoundError
from app.models.common import now_ts
from app.models.intent import IntentConfig, IntentCreate, IntentRead, IntentUpdate
from app.repositories.base import ExampleRepository, IntentRepository
from app.services.index_manager import IndexManager


class IntentService:
    def __init__(
        self,
        store: ChromaStore,
        intents: IntentRepository,
        examples: ExampleRepository,
        index_manager: IndexManager,
    ) -> None:
        self._store = store
        self._intents = intents
        self._examples = examples
        self._indexes = index_manager

    def create(self, domain_id: str, payload: IntentCreate) -> IntentConfig:
        with self._store.lock:
            if self._intents.get_by_name(domain_id, payload.name) is not None:
                raise ConflictError(f"intent '{payload.name}' already exists in this domain")
            intent = IntentConfig(domain_id=domain_id, **payload.model_dump())
            saved = self._intents.save(intent)
            self._indexes.rebuild(domain_id)
            return saved

    def get(self, domain_id: str, intent_id: str) -> IntentConfig:
        intent = self._intents.get(intent_id)
        if intent is None or intent.domain_id != domain_id:
            raise NotFoundError(f"intent '{intent_id}' not found")
        return intent

    def find_by_name(self, domain_id: str, name: str) -> IntentConfig:
        intent = self._intents.get_by_name(domain_id, name)
        if intent is None:
            raise NotFoundError(f"intent '{name}' not found in this domain")
        return intent

    def list(self, domain_id: str) -> list[IntentRead]:
        intents = self._intents.list_for_domain(domain_id)
        counts = self._examples.count_by_intent(domain_id)
        return [
            IntentRead(**intent.model_dump(), example_count=counts.get(intent.id, 0))
            for intent in intents
        ]

    def read(self, domain_id: str, intent_id: str) -> IntentRead:
        intent = self.get(domain_id, intent_id)
        counts = self._examples.count_by_intent(domain_id)
        return IntentRead(**intent.model_dump(), example_count=counts.get(intent.id, 0))

    def update(self, domain_id: str, intent_id: str, payload: IntentUpdate) -> IntentConfig:
        with self._store.lock:
            intent = self.get(domain_id, intent_id)
            changes = payload.model_dump(exclude_unset=True, exclude_none=True)
            new_name = changes.get("name")
            if new_name and new_name != intent.name:
                existing = self._intents.get_by_name(domain_id, new_name)
                if existing is not None and existing.id != intent_id:
                    raise ConflictError(f"intent '{new_name}' already exists in this domain")
            updated = intent.model_copy(update={**changes, "updated_at": now_ts()})
            saved = self._intents.save(updated)
            if new_name and new_name != intent.name:
                # Examples carry a denormalized intent name for debug output.
                self._examples.rename_intent(intent_id, saved.name)
            self._indexes.rebuild(domain_id)
            return saved

    def delete(self, domain_id: str, intent_id: str) -> None:
        with self._store.lock:
            self.get(domain_id, intent_id)
            self._examples.delete_for_intent(intent_id)
            self._intents.delete(intent_id)
            self._indexes.rebuild(domain_id)
