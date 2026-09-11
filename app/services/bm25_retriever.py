"""Sparse (lexical) retrieval over a domain's BM25 index."""

from __future__ import annotations

from app.models.classification import Bm25Hit
from app.services.index_manager import Bm25Index


def retrieve(index: Bm25Index, tokens: list[str], top_k: int) -> list[Bm25Hit]:
    """Top-k examples by BM25 score. Zero-scoring documents are dropped."""
    if index.bm25 is None or not tokens or index.doc_count == 0 or top_k <= 0:
        return []

    scores = index.bm25.get_scores(tokens)
    ordered = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)

    hits: list[Bm25Hit] = []
    for position in ordered[:top_k]:
        score = float(scores[position])
        if score <= 0.0:
            continue
        hits.append(
            Bm25Hit(
                example_id=index.example_ids[position],
                intent_id=index.intent_ids[position],
                intent_name=index.intent_names[position],
                text=index.texts[position],
                score=round(score, 6),
                rank=len(hits) + 1,
            )
        )
    return hits
