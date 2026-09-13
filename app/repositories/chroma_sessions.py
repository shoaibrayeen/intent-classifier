"""Chroma-backed conversation memory.

One record per turn. The document is the user's text, because a follow-up is
read against the previous question; everything else is scalar metadata, with
entities JSON-encoded since Chroma metadata cannot nest.
"""

from __future__ import annotations

import json
import time

from app.db.chroma import DUMMY_EMBEDDING, ChromaStore, clean_metadata
from app.models.classification import SessionTurn
from app.models.common import new_id, now_ts


class ChromaSessionRepository:
    def __init__(self, store: ChromaStore) -> None:
        self._store = store
        self._col = store.sessions

    @staticmethod
    def _to_turn(doc: str, meta: dict) -> SessionTurn:
        try:
            entities = json.loads(str(meta.get("entities") or "{}"))
        except json.JSONDecodeError:
            entities = {}
        try:
            details = json.loads(str(meta.get("details") or "{}"))
        except json.JSONDecodeError:
            details = {}
        return SessionTurn(
            turn=int(meta.get("turn", 0)),
            text=doc,
            intent=str(meta.get("intent", "")),
            intent_id=str(meta["intent_id"]) if meta.get("intent_id") else None,
            confidence=float(meta.get("confidence", 0.0)),
            entities=entities if isinstance(entities, dict) else {},
            created_at=float(meta.get("created_at", 0.0)),
            details=details if isinstance(details, dict) else {},
        )

    def history(self, session_id: str, domain_id: str, limit: int) -> list[SessionTurn]:
        """Earlier turns in this session for this domain, oldest first."""
        where = {"$and": [{"session_id": session_id}, {"domain_id": domain_id}]}
        with self._store.lock:
            result = self._col.get(where=where, include=["documents", "metadatas"])
        turns = [
            self._to_turn(doc, meta)
            for doc, meta in zip(
                result.get("documents") or [], result.get("metadatas") or [], strict=True
            )
        ]
        turns.sort(key=lambda t: (t.turn, t.created_at))
        return turns[-limit:] if limit > 0 else turns

    def append(
        self,
        session_id: str,
        domain_id: str,
        turn: SessionTurn,
        principal: str = "",
    ) -> None:
        with self._store.lock:
            self._col.add(
                ids=[new_id()],
                embeddings=[DUMMY_EMBEDDING],
                documents=[turn.text],
                metadatas=[
                    clean_metadata(
                        {
                            "session_id": session_id,
                            "domain_id": domain_id,
                            "turn": turn.turn,
                            "intent": turn.intent,
                            "intent_id": turn.intent_id or "",
                            "confidence": turn.confidence,
                            "entities": json.dumps(turn.entities, ensure_ascii=False, default=str),
                            "details": json.dumps(turn.details, ensure_ascii=False, default=str),
                            "principal": principal,
                            "created_at": turn.created_at or now_ts(),
                        }
                    )
                ],
            )

    def all_turns(self, session_id: str) -> list[SessionTurn]:
        with self._store.lock:
            result = self._col.get(
                where={"session_id": session_id}, include=["documents", "metadatas"]
            )
        turns = [
            self._to_turn(doc, meta)
            for doc, meta in zip(
                result.get("documents") or [], result.get("metadatas") or [], strict=True
            )
        ]
        turns.sort(key=lambda t: (t.turn, t.created_at))
        return turns

    def delete(self, session_id: str) -> int:
        with self._store.lock:
            ids = self._col.get(where={"session_id": session_id}, include=[])["ids"]
            if ids:
                self._col.delete(ids=ids)
        return len(ids)

    def prune(self, session_id: str, keep: int) -> int:
        """Drop the oldest turns beyond ``keep``. Returns how many were removed."""
        with self._store.lock:
            result = self._col.get(where={"session_id": session_id}, include=["metadatas"])
            rows = sorted(
                zip(result["ids"], result.get("metadatas") or [], strict=True),
                key=lambda pair: (int(pair[1].get("turn", 0)), float(pair[1].get("created_at", 0))),
            )
            stale = [row_id for row_id, _ in rows[:-keep]] if keep > 0 else [r for r, _ in rows]
            if stale:
                self._col.delete(ids=stale)
        return len(stale)

    def expire(self, older_than_seconds: float) -> int:
        """Remove every turn older than the cutoff, across all sessions."""
        cutoff = time.time() - older_than_seconds
        with self._store.lock:
            result = self._col.get(where={"created_at": {"$lt": cutoff}}, include=[])
            ids = result["ids"]
            if ids:
                self._col.delete(ids=ids)
        return len(ids)
