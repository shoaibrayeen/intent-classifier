"""The intent classification engine.

    query
      -> normalize
      -> dense retrieval (FastEmbed + Chroma)  ||  sparse retrieval (BM25)
      -> reciprocal rank fusion over examples
      -> aggregate examples into intents
      -> confidence scoring
      -> intent or UNKNOWN

Entity extraction and tool routing happen *after* this, and only for a
confident result. This class stays synchronous and free of network calls so it
can run in a worker thread without blocking the event loop.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.config import Settings
from app.models.classification import (
    UNKNOWN_INTENT,
    Bm25Hit,
    ClassifyDebug,
    ConfidenceBreakdown,
    DenseHit,
    IntentScore,
    StageTimings,
    UnknownReason,
)
from app.models.common import Status
from app.models.domain import DomainConfig
from app.models.intent import IntentConfig
from app.repositories.base import ExampleRepository, IntentRepository
from app.services import aggregator, bm25_retriever, confidence, dense_retriever
from app.services.embedding import Embedder
from app.services.index_manager import IndexManager
from app.services.normalizer import normalize_text, tokenize
from app.services.rrf import reciprocal_rank_fusion
from app.services.strategies import RetrievalStrategy
from app.services.tracing_helpers import stage

logger = logging.getLogger(__name__)

EMPTY_BREAKDOWN = ConfidenceBreakdown(
    s_dense=0.0,
    s_rrf=0.0,
    s_margin=0.0,
    s_support=0.0,
    confidence=0.0,
    agg_top=0.0,
    agg_second=0.0,
    best_similarity=0.0,
)


@dataclass
class ClassificationOutcome:
    """The engine's verdict, before entities and tool routing are attached."""

    domain: DomainConfig
    intent: IntentConfig | None
    confidence: float
    reason: UnknownReason | None
    ranked: list[IntentScore]
    strategy: str
    timings: StageTimings
    normalized_text: str
    tokens: list[str]
    dense_hits: list[DenseHit]
    bm25_hits: list[Bm25Hit]
    breakdown: ConfidenceBreakdown
    index_version: int

    @property
    def intent_name(self) -> str:
        return self.intent.name if self.intent else UNKNOWN_INTENT

    @property
    def matched(self) -> bool:
        return self.intent is not None

    def to_debug(self) -> ClassifyDebug:
        return ClassifyDebug(
            normalized_text=self.normalized_text,
            tokens=self.tokens,
            dense_hits=self.dense_hits,
            bm25_hits=self.bm25_hits,
            rrf_intents=self.ranked,
            confidence_breakdown=self.breakdown,
            index_version=self.index_version,
            strategy=self.strategy,
            timings=self.timings,
        )


class IntentClassifier:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        examples: ExampleRepository,
        intents: IntentRepository,
        index_manager: IndexManager,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._examples = examples
        self._intents = intents
        self._indexes = index_manager

    def classify(
        self, domain: DomainConfig, text: str, strategy: RetrievalStrategy
    ) -> ClassificationOutcome:
        settings = self._settings
        started = time.perf_counter()
        timings = StageTimings()

        normalized = normalize_text(text)
        tokens = tokenize(text)

        active_intents = [
            intent
            for intent in self._intents.list_for_domain(domain.id)
            if intent.status == Status.ACTIVE
        ]
        intent_names = {intent.id: intent.name for intent in active_intents}
        index = self._indexes.get(domain.id)

        def finish(
            intent: IntentConfig | None,
            confidence_value: float,
            reason: UnknownReason | None,
            ranked: list[IntentScore],
            dense_hits: list[DenseHit],
            bm25_hits: list[Bm25Hit],
            breakdown: ConfidenceBreakdown,
        ) -> ClassificationOutcome:
            timings.total_ms = round((time.perf_counter() - started) * 1000, 3)
            return ClassificationOutcome(
                domain=domain,
                intent=intent,
                confidence=confidence_value,
                reason=reason,
                ranked=ranked,
                strategy=strategy.name,
                timings=timings,
                normalized_text=normalized,
                tokens=tokens,
                dense_hits=dense_hits,
                bm25_hits=bm25_hits,
                breakdown=breakdown,
                index_version=index.version,
            )

        if not active_intents or index.doc_count == 0:
            return finish(None, 0.0, UnknownReason.NO_EXAMPLES, [], [], [], EMPTY_BREAKDOWN)

        dense_hits: list[DenseHit] = []
        bm25_hits: list[Bm25Hit] = []

        if strategy.use_dense:
            with stage("embed") as elapsed:
                vector = self._embedder.embed_one(normalized)
            timings.embed_ms = elapsed()
            with stage("dense_retrieval") as elapsed:
                dense_hits = dense_retriever.retrieve(
                    self._examples,
                    domain.id,
                    vector,
                    strategy.retrieval_top_k,
                    list(intent_names),
                )
            timings.dense_ms = elapsed()

        if strategy.use_bm25:
            with stage("bm25_retrieval") as elapsed:
                bm25_hits = bm25_retriever.retrieve(index, tokens, strategy.retrieval_top_k)
            timings.bm25_ms = elapsed()

        with stage("fusion") as elapsed:
            fused = reciprocal_rank_fusion(
                [[hit.example_id for hit in dense_hits], [hit.example_id for hit in bm25_hits]],
                k=strategy.rrf_k,
            )
            example_intents = {hit.example_id: hit.intent_id for hit in dense_hits}
            example_intents.update({hit.example_id: hit.intent_id for hit in bm25_hits})
            dense_similarity = {hit.example_id: hit.similarity for hit in dense_hits}

            ranked = aggregator.aggregate(
                fused, example_intents, intent_names, dense_similarity, top_n=strategy.agg_top_n
            )
            breakdown = confidence.compute(ranked, settings, strategy=strategy)
        timings.fusion_ms = elapsed()

        reason: UnknownReason | None = None
        intent: IntentConfig | None = None
        if not ranked:
            reason = UnknownReason.NO_EXAMPLES
        elif breakdown.best_similarity < settings.min_dense_similarity and strategy.use_dense:
            reason = UnknownReason.LOW_SIMILARITY
        elif breakdown.confidence < settings.confidence_threshold:
            reason = UnknownReason.LOW_CONFIDENCE
        else:
            intent = next((i for i in active_intents if i.id == ranked[0].intent_id), None)
            if intent is None:
                reason = UnknownReason.NO_EXAMPLES

        return finish(
            intent, breakdown.confidence, reason, ranked, dense_hits, bm25_hits, breakdown
        )
