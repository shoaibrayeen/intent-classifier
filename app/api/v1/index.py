"""Index maintenance endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_domain_service, get_index_manager
from app.models.index import IndexStatus
from app.services.domain_service import DomainService
from app.services.index_manager import IndexManager

router = APIRouter(prefix="/domains/{domain_id}", tags=["index"])


@router.post("/reindex", response_model=IndexStatus)
def reindex(
    domain_id: str,
    domains: DomainService = Depends(get_domain_service),
    indexes: IndexManager = Depends(get_index_manager),
) -> IndexStatus:
    domains.get(domain_id)
    indexes.rebuild(domain_id)
    return indexes.status(domain_id)


@router.get("/index/status", response_model=IndexStatus)
def index_status(
    domain_id: str,
    domains: DomainService = Depends(get_domain_service),
    indexes: IndexManager = Depends(get_index_manager),
) -> IndexStatus:
    domains.get(domain_id)
    return indexes.status(domain_id)
