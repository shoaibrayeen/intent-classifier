"""The intent classification engine.

query
  -> normalize
  -> dense retrieval (FastEmbed + Chroma)  ||  sparse retrieval (BM25)
  -> reciprocal rank fusion over examples
  -> aggregate examples into intents
  -> confidence scoring
  -> intent or UNKNOWN
"""

from __future__ import annotations

import logging

from app.config import Settings
from app.models.classification import (
    UNKNOWN_INTENT,
    ClassifyDebug,
    ClassifyResponse,
    ConfidenceBreakdown,
    UnknownReason,
)
from app.models.common import Status
from app.models.domain import DomainConfig
from app.repositories.base import ExampleRepository, IntentRepository
from app.services import aggregator, bm25_retriever, confidence, dense_retriever
from app.services.embedding import Embedder
from app.services.index_manager import IndexManager
from app.services.normalizer import normalize_text, tokenize
from app.services.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


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

    def classify(self, domain: DomainConfig, text: str, debug: bool = False) -> ClassifyResponse:
        settings = self._settings
        normalized = normalize_text(text)
        tokens = tokenize(text)

        active_intents = [
            intent
            for intent in self._intents.list_for_domain(domain.id)
            if intent.status == Status.ACTIVE
        ]
        intent_names = {intent.id: intent.name for intent in active_intents}
        index = self._indexes.get(domain.id)

        empty_breakdown = ConfidenceBreakdown(
            s_dense=0.0,
            s_rrf=0.0,
            s_margin=0.0,
            s_support=0.0,
            confidence=0.0,
            agg_top=0.0,
            agg_second=0.0,
            best_similarity=0.0,
        )

        if not active_intents or index.doc_count == 0:
            return self._response(
                domain=domain,
                intent_id=None,
                confidence_value=0.0,
                reason=UnknownReason.NO_EXAMPLES,
                ranked=[],
                debug=self._debug(normalized, tokens, [], [], [], empty_breakdown, index.version)
                if debug
                else None,
            )

        vector = self._embedder.embed_one(normalized)
        dense_hits = dense_retriever.retrieve(
            self._examples, domain.id, vector, settings.retrieval_top_k, list(intent_names)
        )
        bm25_hits = bm25_retriever.retrieve(index, tokens, settings.retrieval_top_k)

        fused = reciprocal_rank_fusion(
            [[hit.example_id for hit in dense_hits], [hit.example_id for hit in bm25_hits]],
            k=settings.rrf_k,
        )

        example_intents = {hit.example_id: hit.intent_id for hit in dense_hits}
        example_intents.update({hit.example_id: hit.intent_id for hit in bm25_hits})
        dense_similarity = {hit.example_id: hit.similarity for hit in dense_hits}

        ranked = aggregator.aggregate(
            fused, example_intents, intent_names, dense_similarity, top_n=settings.agg_top_n
        )
        breakdown = confidence.compute(ranked, settings)

        reason: UnknownReason | None = None
        intent_id: str | None = None
        if not ranked:
            reason = UnknownReason.NO_EXAMPLES
        elif breakdown.best_similarity < settings.min_dense_similarity:
            reason = UnknownReason.LOW_SIMILARITY
        elif breakdown.confidence < settings.confidence_threshold:
            reason = UnknownReason.LOW_CONFIDENCE
        else:
            intent_id = ranked[0].intent_id

        return self._response(
            domain=domain,
            intent_id=intent_id,
            confidence_value=breakdown.confidence,
            reason=reason,
            ranked=ranked,
            debug=self._debug(
                normalized, tokens, dense_hits, bm25_hits, ranked, breakdown, index.version
            )
            if debug
            else None,
        )

    # -- helpers --------------------------------------------------------
    def _debug(
        self, normalized, tokens, dense_hits, bm25_hits, ranked, breakdown, index_version
    ) -> ClassifyDebug:
        return ClassifyDebug(
            normalized_text=normalized,
            tokens=tokens,
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            rrf_intents=ranked,
            confidence_breakdown=breakdown,
            index_version=index_version,
        )

    def _response(
        self, domain, intent_id, confidence_value, reason, ranked, debug
    ) -> ClassifyResponse:
        intent_name = UNKNOWN_INTENT
        tool = None
        if intent_id:
            intent = self._intents.get(intent_id)
            if intent is not None:
                intent_name = intent.name
                tool = intent.tool.model_dump() if intent.tool.name else None
        return ClassifyResponse(
            domain_id=domain.id,
            domain=domain.name,
            intent=intent_name,
            intent_id=intent_id,
            confidence=confidence_value,
            entities={},
            tool=tool,
            reason=reason,
            top_intents=ranked[:5],
            debug=debug,
        )
