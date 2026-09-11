"""Classification endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_container, get_domain_service
from app.models.classification import ClassifyRequest, ClassifyResponse
from app.services.container import Container
from app.services.domain_service import DomainService

router = APIRouter(prefix="/classify", tags=["classify"])


def _classify(
    payload: ClassifyRequest, container: Container, domains: DomainService, debug: bool
) -> ClassifyResponse:
    domain = domains.resolve(payload.domain)
    return container.classifier.classify(domain, payload.text, debug=debug)


@router.post("", response_model=ClassifyResponse, response_model_exclude={"debug"})
def classify(
    payload: ClassifyRequest,
    container: Container = Depends(get_container),
    domains: DomainService = Depends(get_domain_service),
) -> ClassifyResponse:
    return _classify(payload, container, domains, debug=False)


@router.post("/debug", response_model=ClassifyResponse)
def classify_debug(
    payload: ClassifyRequest,
    container: Container = Depends(get_container),
    domains: DomainService = Depends(get_domain_service),
) -> ClassifyResponse:
    """Same pipeline, with the retrieval trace attached."""
    return _classify(payload, container, domains, debug=True)
