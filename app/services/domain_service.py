"""Domain CRUD with the uniqueness and cascade rules Chroma cannot enforce."""

from __future__ import annotations

from app.db.chroma import ChromaStore
from app.errors import ConflictError, NotFoundError
from app.models.common import now_ts
from app.models.domain import DomainConfig, DomainCreate, DomainRead, DomainUpdate
from app.repositories.base import DomainRepository, ExampleRepository, IntentRepository
from app.services.index_manager import IndexManager


class DomainService:
    def __init__(
        self,
        store: ChromaStore,
        domains: DomainRepository,
        intents: IntentRepository,
        examples: ExampleRepository,
        index_manager: IndexManager,
    ) -> None:
        self._store = store
        self._domains = domains
        self._intents = intents
        self._examples = examples
        self._indexes = index_manager

    def create(self, payload: DomainCreate) -> DomainConfig:
        with self._store.lock:
            if self._domains.get_by_name(payload.name) is not None:
                raise ConflictError(f"domain '{payload.name}' already exists")
            domain = DomainConfig(**payload.model_dump())
            return self._domains.save(domain)

    def get(self, domain_id: str) -> DomainConfig:
        domain = self._domains.get(domain_id)
        if domain is None:
            raise NotFoundError(f"domain '{domain_id}' not found")
        return domain

    def resolve(self, domain_ref: str) -> DomainConfig:
        """Look a domain up by id, falling back to its name."""
        domain = self._domains.get(domain_ref) or self._domains.get_by_name(domain_ref)
        if domain is None:
            raise NotFoundError(f"domain '{domain_ref}' not found")
        return domain

    def list(self) -> list[DomainRead]:
        domains = self._domains.list()
        intents = self._intents.list_all()
        example_counts = self._examples.count_by_intent()

        reads: list[DomainRead] = []
        for domain in domains:
            domain_intents = [i for i in intents if i.domain_id == domain.id]
            reads.append(
                DomainRead(
                    **domain.model_dump(),
                    intent_count=len(domain_intents),
                    example_count=sum(example_counts.get(i.id, 0) for i in domain_intents),
                )
            )
        return reads

    def read(self, domain_id: str) -> DomainRead:
        domain = self.get(domain_id)
        intents = self._intents.list_for_domain(domain_id)
        counts = self._examples.count_by_intent(domain_id)
        return DomainRead(
            **domain.model_dump(),
            intent_count=len(intents),
            example_count=sum(counts.get(i.id, 0) for i in intents),
        )

    def update(self, domain_id: str, payload: DomainUpdate) -> DomainConfig:
        with self._store.lock:
            domain = self.get(domain_id)
            changes = payload.model_dump(exclude_unset=True, exclude_none=True)
            new_name = changes.get("name")
            if new_name and new_name != domain.name:
                existing = self._domains.get_by_name(new_name)
                if existing is not None and existing.id != domain_id:
                    raise ConflictError(f"domain '{new_name}' already exists")
            updated = domain.model_copy(update={**changes, "updated_at": now_ts()})
            saved = self._domains.save(updated)
            if "status" in changes:
                self._indexes.rebuild(domain_id)
            return saved

    def delete(self, domain_id: str) -> None:
        """Cascade: examples, then intents, then the domain itself."""
        with self._store.lock:
            self.get(domain_id)
            self._examples.delete_for_domain(domain_id)
            self._intents.delete_for_domain(domain_id)
            self._domains.delete(domain_id)
            self._indexes.drop(domain_id)
