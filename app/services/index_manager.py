"""Per-domain BM25 index lifecycle.

rank-bm25 is an in-memory library, so the index is built from Chroma and cached
per domain. Mutations mark a domain dirty and rebuild eagerly: corpora are small
(hundreds of short examples) and an eager rebuild keeps the first classify after
an edit fast and the reported status truthful.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rank_bm25 import BM25Okapi

from app.models.index import IndexState, IndexStatus
from app.repositories.base import ExampleRepository, IntentRepository
from app.services.normalizer import TOKENIZER_VERSION, tokenize

logger = logging.getLogger(__name__)


@dataclass
class Bm25Index:
    domain_id: str
    version: int
    built_at: datetime
    example_ids: list[str] = field(default_factory=list)
    intent_ids: list[str] = field(default_factory=list)
    intent_names: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    @property
    def doc_count(self) -> int:
        return len(self.example_ids)


class IndexManager:
    def __init__(self, examples: ExampleRepository, intents: IntentRepository) -> None:
        self._examples = examples
        self._intents = intents
        self._indexes: dict[str, Bm25Index] = {}
        self._dirty: set[str] = set()
        self._versions: dict[str, int] = {}
        self._lock = threading.RLock()

    # -- lifecycle ------------------------------------------------------
    def get(self, domain_id: str) -> Bm25Index:
        """Return a ready index, building it if missing or stale."""
        with self._lock:
            index = self._indexes.get(domain_id)
            if index is None or domain_id in self._dirty:
                return self.build(domain_id)
            return index

    def build(self, domain_id: str) -> Bm25Index:
        with self._lock:
            active_intents = [
                intent
                for intent in self._intents.list_for_domain(domain_id)
                if intent.status == "active"
            ]
            intent_names = {intent.id: intent.name for intent in active_intents}
            examples = self._examples.list_for_domain(domain_id, list(intent_names))

            version = self._versions.get(domain_id, 0) + 1
            self._versions[domain_id] = version

            corpus = [tokenize(example.text) for example in examples]
            # BM25Okapi divides by the average document length, so an empty
            # corpus (or a corpus of only unparseable text) must be skipped.
            usable = [tokens for tokens in corpus if tokens]
            bm25 = BM25Okapi(corpus) if usable else None

            index = Bm25Index(
                domain_id=domain_id,
                version=version,
                built_at=datetime.now(UTC),
                example_ids=[e.id for e in examples],
                intent_ids=[e.intent_id for e in examples],
                intent_names=[intent_names.get(e.intent_id, e.intent_name) for e in examples],
                texts=[e.text for e in examples],
                bm25=bm25,
            )
            self._indexes[domain_id] = index
            self._dirty.discard(domain_id)
            logger.info(
                "bm25 index built domain=%s version=%d docs=%d", domain_id, version, index.doc_count
            )
            return index

    def mark_dirty(self, domain_id: str) -> None:
        with self._lock:
            self._dirty.add(domain_id)

    def rebuild(self, domain_id: str) -> Bm25Index:
        """Mark dirty and rebuild immediately (used after every mutation)."""
        self.mark_dirty(domain_id)
        return self.build(domain_id)

    def warm(self, domain_ids: list[str]) -> int:
        """Build every domain's index up front.

        Without this the first classification after a restart pays the build
        cost, and index health reports "not_built" for a service that is in
        fact perfectly healthy.
        """
        for domain_id in domain_ids:
            self.build(domain_id)
        return len(domain_ids)

    def clear(self) -> None:
        """Forget every cached index (used when the store is reset)."""
        with self._lock:
            self._indexes.clear()
            self._dirty.clear()
            self._versions.clear()

    def drop(self, domain_id: str) -> None:
        with self._lock:
            self._indexes.pop(domain_id, None)
            self._dirty.discard(domain_id)
            self._versions.pop(domain_id, None)

    # -- reporting ------------------------------------------------------
    def status(self, domain_id: str) -> IndexStatus:
        with self._lock:
            index = self._indexes.get(domain_id)
            dense_count = self._examples.count_for_domain(domain_id)
            if index is None:
                state = IndexState.NOT_BUILT
            elif domain_id in self._dirty:
                state = IndexState.DIRTY
            else:
                state = IndexState.READY
            return IndexStatus(
                domain_id=domain_id,
                state=state,
                version=index.version if index else 0,
                built_at=index.built_at.isoformat() if index else None,
                bm25_doc_count=index.doc_count if index else 0,
                dense_count=dense_count,
                tokenizer_version=TOKENIZER_VERSION,
            )
