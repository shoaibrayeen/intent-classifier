"""Chroma-backed example repository (the retrieval corpus)."""

from __future__ import annotations

from collections import Counter

from app.db.chroma import ChromaStore, clean_metadata
from app.models.classification import DenseHit
from app.models.example import Example


class ChromaExampleRepository:
    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self._col = store.examples

    @staticmethod
    def _metadata(example: Example) -> dict[str, object]:
        return clean_metadata(
            {
                "example_id": example.id,
                "domain_id": example.domain_id,
                "intent_id": example.intent_id,
                "intent_name": example.intent_name,
                "text_hash": example.text_hash,
                "created_at": example.created_at,
            }
        )

    @staticmethod
    def _to_example(doc: str, meta: dict) -> Example:
        return Example(
            id=str(meta["example_id"]),
            domain_id=str(meta["domain_id"]),
            intent_id=str(meta["intent_id"]),
            intent_name=str(meta.get("intent_name", "")),
            text=doc,
            text_hash=str(meta.get("text_hash", "")),
            created_at=float(meta.get("created_at", 0.0)),
        )

    def _collect(self, result: dict) -> list[Example]:
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        examples = [self._to_example(d, m) for d, m in zip(documents, metadatas, strict=True)]
        return sorted(examples, key=lambda e: e.created_at)

    def get(self, example_id: str) -> Example | None:
        with self._store.lock:
            result = self._col.get(ids=[example_id], include=["documents", "metadatas"])
        examples = self._collect(result)
        return examples[0] if examples else None

    def list_for_intent(self, intent_id: str) -> list[Example]:
        with self._store.lock:
            result = self._col.get(
                where={"intent_id": intent_id}, include=["documents", "metadatas"]
            )
        return self._collect(result)

    def list_for_domain(self, domain_id: str, intent_ids: list[str] | None = None) -> list[Example]:
        if intent_ids is not None and not intent_ids:
            return []
        where: dict = {"domain_id": domain_id}
        if intent_ids is not None:
            where = {"$and": [{"domain_id": domain_id}, {"intent_id": {"$in": intent_ids}}]}
        with self._store.lock:
            result = self._col.get(where=where, include=["documents", "metadatas"])
        return self._collect(result)

    def find_by_hash(self, intent_id: str, text_hash: str) -> Example | None:
        where = {"$and": [{"intent_id": intent_id}, {"text_hash": text_hash}]}
        with self._store.lock:
            result = self._col.get(where=where, include=["documents", "metadatas"])
        examples = self._collect(result)
        return examples[0] if examples else None

    def add_many(self, examples: list[Example], embeddings: list[list[float]]) -> None:
        if not examples:
            return
        with self._store.lock:
            self._col.upsert(
                ids=[e.id for e in examples],
                embeddings=embeddings,
                documents=[e.text for e in examples],
                metadatas=[self._metadata(e) for e in examples],
            )

    def delete(self, example_id: str) -> None:
        with self._store.lock:
            self._col.delete(ids=[example_id])

    def delete_for_intent(self, intent_id: str) -> None:
        with self._store.lock:
            ids = self._col.get(where={"intent_id": intent_id}, include=[])["ids"]
            if ids:
                self._col.delete(ids=ids)

    def delete_for_domain(self, domain_id: str) -> None:
        with self._store.lock:
            ids = self._col.get(where={"domain_id": domain_id}, include=[])["ids"]
            if ids:
                self._col.delete(ids=ids)

    def count_by_intent(self, domain_id: str | None = None) -> dict[str, int]:
        where = {"domain_id": domain_id} if domain_id else None
        with self._store.lock:
            result = self._col.get(where=where, include=["metadatas"])
        counts = Counter(str(m["intent_id"]) for m in result.get("metadatas") or [])
        return dict(counts)

    def count_for_domain(self, domain_id: str) -> int:
        with self._store.lock:
            return len(self._col.get(where={"domain_id": domain_id}, include=[])["ids"])

    def rename_intent(self, intent_id: str, intent_name: str) -> None:
        """Refresh the denormalized intent name carried on every example."""
        with self._store.lock:
            result = self._col.get(where={"intent_id": intent_id}, include=["metadatas"])
            ids = result["ids"]
            if not ids:
                return
            metadatas = []
            for meta in result.get("metadatas") or []:
                updated = dict(meta)
                updated["intent_name"] = intent_name
                metadatas.append(updated)
            self._col.update(ids=ids, metadatas=metadatas)

    def query_dense(
        self, domain_id: str, vector: list[float], top_k: int, intent_ids: list[str]
    ) -> list[DenseHit]:
        if not intent_ids or top_k <= 0:
            return []
        with self._store.lock:
            available = len(self._col.get(where={"domain_id": domain_id}, include=[])["ids"])
            if available == 0:
                return []
            where = {"$and": [{"domain_id": domain_id}, {"intent_id": {"$in": intent_ids}}]}
            result = self._col.query(
                query_embeddings=[vector],
                n_results=min(top_k, available),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        hits: list[DenseHit] = []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for rank, (doc, meta, distance) in enumerate(
            zip(documents, metadatas, distances, strict=True), start=1
        ):
            # Collection uses cosine space, so Chroma returns 1 - cosine_similarity.
            hits.append(
                DenseHit(
                    example_id=str(meta["example_id"]),
                    intent_id=str(meta["intent_id"]),
                    intent_name=str(meta.get("intent_name", "")),
                    text=doc,
                    similarity=round(1.0 - float(distance), 6),
                    rank=rank,
                )
            )
        return hits
