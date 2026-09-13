"""Retrieval strategy variants and their assignment.

Changing retrieval is the highest-risk change this service can make, so
variants are explicit, named, and assigned deterministically. The same request
id always lands on the same variant, which makes a reported result
reproducible: re-send the id and you get the same pipeline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from app.config import Settings

HYBRID_RRF = "hybrid_rrf"
DENSE_ONLY = "dense_only"
BM25_ONLY = "bm25_only"
AUTO = "auto"

#: Tried in order by AUTO, after hybrid has already failed to resolve.
#: Hybrid is measurably the best single strategy, so it always answers first and
#: the rest exist only to rescue a query it could not place.
#:
#: Every fallback must use dense retrieval. The minimum-similarity floor is what
#: rejects an off-domain query, and it can only be applied when there is a
#: similarity to judge: measured here, a lexical-only pass matched "what is the
#: weather today" to CONTRACT_EXPIRY at 0.86 confidence on one shared word.
#: A rescue that accepts anything is not a rescue.
AUTO_FALLBACKS: tuple[str, ...] = ("hybrid_wide", DENSE_ONLY)


@dataclass(frozen=True)
class RetrievalStrategy:
    """One configuration of the retrieval pipeline."""

    name: str
    use_dense: bool = True
    use_bm25: bool = True
    rrf_k: int = 60
    agg_top_n: int = 3
    retrieval_top_k: int = 10
    description: str = ""

    def for_settings(self, settings: Settings) -> RetrievalStrategy:
        """Baseline variants inherit the service defaults."""
        if self.name != HYBRID_RRF:
            return self
        return replace(
            self,
            rrf_k=settings.rrf_k,
            agg_top_n=settings.agg_top_n,
            retrieval_top_k=settings.retrieval_top_k,
        )


BUILTIN_STRATEGIES: dict[str, RetrievalStrategy] = {
    HYBRID_RRF: RetrievalStrategy(
        name=HYBRID_RRF,
        description="Dense and BM25 fused with Reciprocal Rank Fusion. The default.",
    ),
    DENSE_ONLY: RetrievalStrategy(
        name=DENSE_ONLY,
        use_bm25=False,
        description="Semantic retrieval alone. Loses exact-term matches.",
    ),
    BM25_ONLY: RetrievalStrategy(
        name=BM25_ONLY,
        use_dense=False,
        description="Lexical retrieval alone. Loses paraphrase.",
    ),
    "hybrid_rrf_k20": RetrievalStrategy(
        name="hybrid_rrf_k20",
        rrf_k=20,
        description="Fusion with a smaller k, which sharpens the top ranks.",
    ),
    "hybrid_top1": RetrievalStrategy(
        name="hybrid_top1",
        agg_top_n=1,
        description="Score each intent by its single best example only.",
    ),
    "hybrid_wide": RetrievalStrategy(
        name="hybrid_wide",
        retrieval_top_k=20,
        description="Retrieve twice as many candidates before fusing.",
    ),
}

#: AUTO is a selection policy, not a retrieval configuration, so it is resolved
#: by the classification service rather than being a strategy in its own right.
AUTO_STRATEGY = RetrievalStrategy(
    name=AUTO,
    description=(
        "Answer with hybrid fusion; if it cannot place the query, retry the "
        "alternates and report whichever resolved it."
    ),
)


class StrategySelector:
    """Resolves which strategy a request runs, and why."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.ab_testing_enabled and len(self.variants) > 1

    @property
    def variants(self) -> list[str]:
        configured = [
            name.strip()
            for name in self._settings.ab_variants.split(",")
            if name.strip() and name.strip() in BUILTIN_STRATEGIES
        ]
        return configured or [HYBRID_RRF]

    def get(self, name: str) -> RetrievalStrategy:
        if name == AUTO:
            return AUTO_STRATEGY
        strategy = BUILTIN_STRATEGIES.get(name, BUILTIN_STRATEGIES[HYBRID_RRF])
        return strategy.for_settings(self._settings)

    def auto_sequence(self) -> list[RetrievalStrategy]:
        """Hybrid first, then the rescues, in the order AUTO should try them.

        Candidates without dense retrieval are dropped rather than trusted: see
        AUTO_FALLBACKS. Filtering here as well as in the list means a future
        edit to that tuple cannot quietly reopen the hole.
        """
        fallbacks = [self.get(name) for name in AUTO_FALLBACKS]
        return [self.get(HYBRID_RRF), *(s for s in fallbacks if s.use_dense)]

    def select(self, request_id: str, override: str | None = None) -> RetrievalStrategy:
        """An explicit override always wins, so a caller can reproduce a result."""
        if override:
            return self.get(override)
        if not self.enabled:
            return self.get(self._settings.default_strategy)

        variants = self.variants
        digest = hashlib.sha256(request_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % len(variants)
        return self.get(variants[bucket])
