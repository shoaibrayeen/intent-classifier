"""Index maintenance endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.dependencies import (
    authorize_domain,
    get_domain_service,
    get_index_manager,
    require_read,
    require_write,
)
from app.models.index import IndexStatus
from app.security.auth import Principal
from app.services.domain_service import DomainService
from app.services.index_manager import IndexManager

router = APIRouter(prefix="/domains/{domain_id}", tags=["index"])


@router.post("/reindex", response_model=IndexStatus)
def reindex(
    domain_id: str,
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    indexes: IndexManager = Depends(get_index_manager),
) -> IndexStatus:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    indexes.rebuild(domain_id)
    status = indexes.status(domain_id)
    request.app.state.container.audit.record(
        action="index.rebuild", resource=f"domain:{domain.name}", version=status.version
    )
    return status


@router.get("/index/status", response_model=IndexStatus)
def index_status(
    domain_id: str,
    request: Request,
    principal: Principal = Depends(require_read),
    domains: DomainService = Depends(get_domain_service),
    indexes: IndexManager = Depends(get_index_manager),
) -> IndexStatus:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return indexes.status(domain_id)
