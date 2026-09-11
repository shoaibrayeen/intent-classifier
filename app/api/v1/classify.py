"""Classification endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.dependencies import (
    authorize_domain,
    get_classification_service,
    get_domain_service,
    require_classify,
)
from app.models.classification import ClassifyRequest, ClassifyResponse
from app.security.auth import Principal
from app.services.classification_service import ClassificationService
from app.services.domain_service import DomainService

router = APIRouter(prefix="/classify", tags=["classify"])


@router.post("", response_model=ClassifyResponse, response_model_exclude={"debug"})
async def classify(
    payload: ClassifyRequest,
    request: Request,
    principal: Principal = Depends(require_classify),
    service: ClassificationService = Depends(get_classification_service),
    domains: DomainService = Depends(get_domain_service),
) -> ClassifyResponse:
    domain = domains.resolve(payload.domain)
    authorize_domain(request, domain.id, domain.name)
    return await service.classify(payload, debug=False)


@router.post("/debug", response_model=ClassifyResponse)
async def classify_debug(
    payload: ClassifyRequest,
    request: Request,
    principal: Principal = Depends(require_classify),
    service: ClassificationService = Depends(get_classification_service),
    domains: DomainService = Depends(get_domain_service),
) -> ClassifyResponse:
    """The same decision, with the full retrieval trace attached."""
    domain = domains.resolve(payload.domain)
    authorize_domain(request, domain.id, domain.name)
    return await service.classify(payload, debug=True)
