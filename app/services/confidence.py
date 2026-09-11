"""Confidence scoring.

The fused RRF value is a ranking quantity, not a probability, so it is never
returned as confidence directly. Confidence combines four bounded signals:

* ``s_dense``   how semantically close the best matching example is
* ``s_rrf``     the aggregated fusion score against its theoretical ceiling
* ``s_margin``  how far the top intent is ahead of the runner-up
* ``s_support`` how many retrieved examples back the top intent

BM25's raw score is deliberately excluded: its scale depends on corpus size and
document length, so it is not comparable across domains. Its evidence already
enters through the fusion ranking.
"""

from __future__ import annotations

from app.config import Settings
from app.models.classification import ConfidenceBreakdown, IntentScore
from app.services.rrf import max_possible_score
from app.services.strategies import RetrievalStrategy


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def compute(
    ranked: list[IntentScore],
    settings: Settings,
    support_target: int = 3,
    strategy: RetrievalStrategy | None = None,
) -> ConfidenceBreakdown:
    if not ranked:
        return ConfidenceBreakdown(
            s_dense=0.0,
            s_rrf=0.0,
            s_margin=0.0,
            s_support=0.0,
            confidence=0.0,
            agg_top=0.0,
            agg_second=0.0,
            best_similarity=0.0,
        )

    top = ranked[0]
    agg_top = top.score
    agg_second = ranked[1].score if len(ranked) > 1 else 0.0

    # A single-retriever strategy can only ever reach half the hybrid ceiling,
    # so it is normalized against its own maximum rather than being penalised
    # for evidence it was never configured to collect.
    rrf_k = strategy.rrf_k if strategy else settings.rrf_k
    agg_top_n = strategy.agg_top_n if strategy else settings.agg_top_n
    n_lists = 2
    if strategy is not None and not (strategy.use_dense and strategy.use_bm25):
        n_lists = 1
    ceiling = max_possible_score(rrf_k, agg_top_n, n_lists=n_lists)
    s_rrf = _clamp(agg_top / ceiling) if ceiling > 0 else 0.0
    s_margin = _clamp((agg_top - agg_second) / agg_top) if agg_top > 0 else 0.0

    span = settings.dense_sim_ceil - settings.dense_sim_floor
    if strategy is not None and not strategy.use_dense:
        # Without dense retrieval there is no similarity to judge; redistributing
        # its weight is more honest than scoring every result as zero.
        s_dense = _clamp((s_rrf + s_margin) / 2)
    else:
        s_dense = (
            _clamp((top.best_similarity - settings.dense_sim_floor) / span) if span > 0 else 0.0
        )

    s_support = _clamp(min(top.supporting_examples, support_target) / support_target)

    confidence = (
        settings.w_dense * s_dense
        + settings.w_rrf * s_rrf
        + settings.w_margin * s_margin
        + settings.w_support * s_support
    )

    return ConfidenceBreakdown(
        s_dense=round(s_dense, 4),
        s_rrf=round(s_rrf, 4),
        s_margin=round(s_margin, 4),
        s_support=round(s_support, 4),
        confidence=round(_clamp(confidence), 4),
        agg_top=round(agg_top, 6),
        agg_second=round(agg_second, 6),
        best_similarity=round(top.best_similarity, 6),
    )
