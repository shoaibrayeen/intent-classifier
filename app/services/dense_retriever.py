"""Dense (semantic) retrieval over the Chroma example collection."""

from __future__ import annotations

from app.models.classification import DenseHit
from app.repositories.base import ExampleRepository


def retrieve(
    examples: ExampleRepository,
    domain_id: str,
    vector: list[float],
    top_k: int,
    intent_ids: list[str],
) -> list[DenseHit]:
    """Top-k nearest examples, restricted to the domain's active intents."""
    return examples.query_dense(domain_id, vector, top_k, intent_ids)
