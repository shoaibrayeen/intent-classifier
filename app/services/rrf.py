"""Reciprocal Rank Fusion.

Ranks, not scores, are fused: dense cosine similarity and BM25 scores live on
incomparable scales, so combining them numerically is not meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """Fuse ranked id lists into ``{id: rrf_score}``.

    Each list contributes ``1 / (k + rank)`` for the ids it contains, with
    ``rank`` starting at 1.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k + rank)
    return fused


def max_possible_score(k: int = 60, top_n: int = 3, n_lists: int = 2) -> float:
    """Ceiling used to normalize an aggregated intent score into [0, 1].

    Reached when one intent's examples occupy ranks 1..top_n in every list.
    """
    return n_lists * sum(1.0 / (k + rank) for rank in range(1, top_n + 1))
