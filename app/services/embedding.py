"""FastEmbed wrapper.

The model is loaded once at startup and reused. ``bge-small-en-v1.5`` returns
L2-normalized 384-dimensional vectors, so Chroma's cosine distance is directly
``1 - cosine_similarity``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from fastembed import TextEmbedding

from app.config import Settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, settings: Settings) -> None:
        cache_dir = Path(settings.model_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = settings.model_name
        logger.info("loading embedding model %s (cache=%s)", self.model_name, cache_dir)
        self._model = TextEmbedding(model_name=self.model_name, cache_dir=str(cache_dir))
        self.dimension = len(self.embed_one("dimension probe"))
        logger.info("embedding model ready (dim=%d)", self.dimension)

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        return [vector.tolist() for vector in self._model.embed(items)]

    def embed_one(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
