"""Orchestration: classify, then extract entities, then map to a tool.

This is the async seam. Retrieval is CPU-bound and synchronous, so it runs in a
worker thread; entity extraction is a network call, so it runs on the event
loop. Mixing the two in one handler is what this class exists to get right.

With a session id the turn is also read against the conversation so far: a
follow-up that cannot stand alone is retried with the previous question, and
entities named earlier carry into an intent that accepts them.
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
    ContextInfo,
    EntityExtractionInfo,
    SessionTurn,
)
from app.observability import metrics, tracing
from app.observability.audit import AuditLog
from app.observability.context import get_principal, get_request_id
from app.services.classifier import ClassificationOutcome, IntentClassifier
from app.services.domain_service import DomainService
from app.services.llm.entity_extractor import EntityExtractor
from app.services.session_service import SessionService
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
        sessions: SessionService,
    ) -> None:
        self._settings = settings
        self._classifier = classifier
        self._domains = domains
        self._extractor = extractor
        self._router = router
        self._strategies = strategies
        self._audit = audit
        self._sessions = sessions

    async def classify(self, payload: ClassifyRequest, debug: bool = False) -> ClassifyResponse:
        started = time.perf_counter()
        request_id = get_request_id()
        domain = self._domains.resolve(payload.domain)
        strategy = self._strategies.select(request_id or payload.text, payload.variant)

        # --- conversation so far -------------------------------------------
        history: list[SessionTurn] = []
        context: ContextInfo | None = None
        session_active = bool(payload.session_id) and self._sessions.enabled
        if session_active:
            history = self._sessions.history(payload.session_id, domain.id)
            if not payload.use_context:
                history = []
            context = ContextInfo(
                session_id=payload.session_id,
                turn=self._sessions.next_turn_number(
                    self._sessions.all_turns(payload.session_id)
                    if not payload.use_context
                    else history
                ),
                previous_turns=len(history),
                previous_intent=history[-1].intent if history else None,
            )

        # --- retrieval -------------------------------------------------------
        with tracing.span("classify", domain=domain.name, strategy=strategy.name):
            outcome: ClassificationOutcome = await run_in_threadpool(
                self._classifier.classify, domain, payload.text, strategy
            )

        # A follow-up like "when do they expire" has no anchor on its own.
        # Retry with the previous question, and keep the result only if it
        # actually resolves -- context may rescue a turn, never overrule one.
        if context is not None and not outcome.matched:
            contextual_text = self._sessions.contextual_text(history, payload.text)
            if contextual_text:
                with tracing.span("classify.contextual", domain=domain.name):
                    retried: ClassificationOutcome = await run_in_threadpool(
                        self._classifier.classify, domain, contextual_text, strategy
                    )
                if retried.matched:
                    outcome = retried
                    context.used_for_retrieval = True
                    context.retrieval_text = contextual_text

        # --- entities ----------------------------------------------------------
        entities: dict = {}
        extraction: EntityExtractionInfo | None = None

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
                        domain=domain,
                        intent=outcome.intent,
                        history=history,
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

            if context is not None and history:
                entities, carried = self._sessions.carry_over(
                    history, entities, outcome.intent.entity_schema
                )
                context.carried_entities = carried

        tool = self._router.route(outcome.intent, entities) if outcome.intent else None

        total_seconds = time.perf_counter() - started
        outcome.timings.total_ms = round(total_seconds * 1000, 3)

        # --- remember this turn --------------------------------------------------
        if context is not None:
            self._sessions.record(
                payload.session_id,
                domain.id,
                SessionTurn(
                    turn=context.turn,
                    text=payload.text,
                    intent=outcome.intent_name,
                    intent_id=outcome.intent.id if outcome.intent else None,
                    confidence=outcome.confidence,
                    entities=entities,
                ),
                principal=get_principal(),
            )

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
            session_id=payload.session_id,
            turn=context.turn if context else None,
            context_used=context.used_for_retrieval if context else False,
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
            context=context,
            debug=outcome.to_debug() if debug else None,
        )
