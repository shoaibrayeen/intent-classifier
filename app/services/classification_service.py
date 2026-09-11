"""Orchestration: classify, then extract entities, then map to a tool.

This is the async seam. Retrieval is CPU-bound and synchronous, so it runs in a
worker thread; entity extraction is a network call, so it runs on the event
loop. Mixing the two in one handler is what this class exists to get right.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.models.classification import (
    ClassifyRequest,
    ClassifyResponse,
    EntityExtractionInfo,
)
from app.observability import metrics, tracing
from app.observability.audit import AuditLog
from app.observability.context import get_request_id
from app.services.classifier import ClassificationOutcome, IntentClassifier
from app.services.domain_service import DomainService
from app.services.llm.entity_extractor import EntityExtractor
from app.services.strategies import StrategySelector
from app.services.tool_router import ToolRouter

logger = logging.getLogger(__name__)


class ClassificationService:
    def __init__(
        self,
        settings: Settings,
        classifier: IntentClassifier,
        domains: DomainService,
        extractor: EntityExtractor,
        router: ToolRouter,
        strategies: StrategySelector,
        audit: AuditLog,
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._domains = domains
        self._extractor = extractor
        self._router = router
        self._strategies = strategies
        self._audit = audit

    async def classify(self, payload: ClassifyRequest, debug: bool = False) -> ClassifyResponse:
        started = time.perf_counter()
        request_id = get_request_id()
        domain = self._domains.resolve(payload.domain)
        strategy = self._strategies.select(request_id or payload.text, payload.variant)

        with tracing.span("classify", domain=domain.name, strategy=strategy.name):
            outcome: ClassificationOutcome = await run_in_threadpool(
                self._classifier.classify, domain, payload.text, strategy
            )

        entities: dict = {}
        extraction: EntityExtractionInfo | None = None

        # Extraction runs only for a confident result. Pulling arguments out of a
        # request the engine did not understand produces confident nonsense.
        if outcome.matched and outcome.intent is not None:
            wanted = (
                payload.extract_entities
                if payload.extract_entities is not None
                else self._settings.entity_extraction_enabled
            )
            if wanted:
                extraction_started = time.perf_counter()
                with tracing.span("entity_extraction", intent=outcome.intent.name):
                    entities, extraction = await self._extractor.extract(
                        payload.text,
                        outcome.intent.entity_schema,
                        today=datetime.now(UTC).date().isoformat(),
                    )
                elapsed = time.perf_counter() - extraction_started
                outcome.timings.extraction_ms = round(elapsed * 1000, 3)
                metrics.stage_latency.labels(stage="entity_extraction").observe(elapsed)
                metrics.extractions.labels(status=extraction.status).inc()
            elif payload.extract_entities is False:
                extraction = EntityExtractionInfo(
                    status="skipped", detail="disabled for this request"
                )
            else:
                extraction = EntityExtractionInfo(
                    status="disabled", detail="entity extraction is turned off"
                )

        tool = self._router.route(outcome.intent, entities) if outcome.intent else None

        total_seconds = time.perf_counter() - started
        outcome.timings.total_ms = round(total_seconds * 1000, 3)

        outcome_label = "matched" if outcome.matched else "unknown"
        metrics.classifications.labels(outcome=outcome_label, strategy=strategy.name).inc()
        metrics.classification_latency.labels(strategy=strategy.name).observe(total_seconds)
        metrics.confidence_observed.observe(outcome.confidence)

        self._audit.record(
            action="classify",
            resource=f"domain:{domain.name}",
            outcome=outcome_label,
            intent=outcome.intent_name,
            confidence=outcome.confidence,
            strategy=strategy.name,
            reason=str(outcome.reason) if outcome.reason else None,
            entity_count=len(entities),
            tool=tool.name if tool else None,
            latency_ms=outcome.timings.total_ms,
            query=self._audit.record_query(payload.text),
        )

        return ClassifyResponse(
            domain_id=domain.id,
            domain=domain.name,
            intent=outcome.intent_name,
            intent_id=outcome.intent.id if outcome.intent else None,
            confidence=outcome.confidence,
            entities=entities,
            entity_extraction=extraction,
            tool=tool,
            reason=outcome.reason,
            top_intents=outcome.ranked[:5],
            latency_ms=outcome.timings.total_ms,
            request_id=request_id or None,
            debug=outcome.to_debug() if debug else None,
        )
