"""Liveness, readiness and index health."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from app.dependencies import get_container, require_admin, require_read
from app.observability import metrics, tracing
from app.security.auth import Principal
from app.services.container import Container

router = APIRouter(tags=["operations"])


@router.get("/health")
def health(container: Container = Depends(get_container)) -> dict:
    """Unauthenticated on purpose: a health probe should not need a credential."""
    return {
        "status": "ok",
        "version": container.settings.app_version,
        "chroma": "ok" if container.store.heartbeat() else "unavailable",
        "chroma_mode": container.settings.chroma_mode,
        "model": container.embedder.model_name,
        "model_loaded": True,
        "embedding_dimension": container.embedder.dimension,
        "entity_extraction": {
            "enabled": container.entity_extractor.enabled,
            "provider_configured": container.entity_extractor.available,
            "provider": container.entity_extractor.provider_name,
            "model": container.settings.openai_model,
        },
        "sessions_enabled": container.sessions.enabled,
        "auth_enabled": container.authenticator.enabled,
        "tracing_enabled": tracing.enabled(),
        "ab_testing_enabled": container.strategies.enabled,
        "strategy": container.settings.default_strategy,
    }


@router.get("/health/index")
def index_health(
    principal: Principal = Depends(require_read),
    container: Container = Depends(get_container),
) -> dict:
    """Per-domain index health: are the two indexes in step?"""
    domains = []
    healthy = True
    for domain in container.domains.list():
        if not principal.may_access(domain.id, domain.name):
            continue
        status = container.index_manager.status(domain.id)
        in_sync = status.bm25_doc_count == status.dense_count
        healthy = healthy and in_sync and status.state != "dirty"
        metrics.index_version.labels(domain=domain.name).set(status.version)
        metrics.index_documents.labels(domain=domain.name).set(status.bm25_doc_count)
        domains.append(
            {
                "domain": domain.name,
                "domain_id": domain.id,
                **status.model_dump(),
                "in_sync": in_sync,
            }
        )
    return {"status": "ok" if healthy else "degraded", "domains": domains}


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics(container: Container = Depends(get_container)) -> Response:
    if not container.settings.metrics_enabled:
        return Response(status_code=404)
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@router.get("/audit")
def audit_tail(
    limit: int = 100,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    return {
        "enabled": container.audit.enabled,
        "path": str(container.audit.path),
        "entries": container.audit.tail(min(max(limit, 1), 1000)),
    }


@router.get("/strategies")
def strategies(
    request: Request,
    principal: Principal = Depends(require_read),
    container: Container = Depends(get_container),
) -> dict:
    from app.services.strategies import BUILTIN_STRATEGIES

    return {
        "ab_testing_enabled": container.strategies.enabled,
        "default": container.settings.default_strategy,
        "assigned_variants": container.strategies.variants,
        "available": [
            {
                "name": strategy.name,
                "description": strategy.description,
                "use_dense": strategy.use_dense,
                "use_bm25": strategy.use_bm25,
                "rrf_k": strategy.rrf_k,
                "agg_top_n": strategy.agg_top_n,
                "retrieval_top_k": strategy.retrieval_top_k,
            }
            for strategy in BUILTIN_STRATEGIES.values()
        ],
    }
