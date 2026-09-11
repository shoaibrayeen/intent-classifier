"""ChromaDB access layer.

Chroma is the single store for this service: two config collections (domains,
intents) plus one collection of embedded intent examples. Everything goes
through one client instance guarded by a re-entrant lock, because embedded
Chroma is not safe for concurrent writers.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.config import Settings

logger = logging.getLogger(__name__)

DOMAINS_COLLECTION = "domains"
INTENTS_COLLECTION = "intents"
EXAMPLES_COLLECTION = "intent_examples"
SESSIONS_COLLECTION = "session_turns"

#: Config collections hold no vectors, but Chroma requires an embedding on every
#: add when the collection has no embedding function. A constant 1-d vector is
#: the cheapest way to satisfy that.
DUMMY_EMBEDDING: list[float] = [0.0]


class ChromaStore:
    """Owns the Chroma client, the three collections, and the global write lock."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.lock = threading.RLock()
        chroma_settings = ChromaSettings(anonymized_telemetry=False)

        if settings.chroma_mode == "ephemeral":
            self.client = chromadb.EphemeralClient(settings=chroma_settings)
        else:
            path = Path(settings.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path), settings=chroma_settings)

        logger.info("chroma client ready (mode=%s)", settings.chroma_mode)

        self.domains: Collection = self.client.get_or_create_collection(
            DOMAINS_COLLECTION, embedding_function=None
        )
        self.intents: Collection = self.client.get_or_create_collection(
            INTENTS_COLLECTION, embedding_function=None
        )
        self.examples: Collection = self.client.get_or_create_collection(
            EXAMPLES_COLLECTION,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        # Conversation turns live in the same store as everything else, so a
        # single directory (or a single volume) is the whole deployment.
        self.sessions: Collection = self.client.get_or_create_collection(
            SESSIONS_COLLECTION, embedding_function=None
        )

    def heartbeat(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:  # pragma: no cover - only on a broken store
            logger.exception("chroma heartbeat failed")
            return False

    def reset_all(self) -> None:
        """Drop every record. Used by the seed script's --reset and by tests."""
        with self.lock:
            for collection in (self.sessions, self.examples, self.intents, self.domains):
                ids = collection.get(include=[])["ids"]
                if ids:
                    collection.delete(ids=ids)


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata accepts only str/int/float/bool. Coerce and drop ``None``.

    Note that Chroma *merges* metadata on upsert/update rather than replacing it,
    so callers must always write the complete key set.
    """
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, bool | int | float | str):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned
