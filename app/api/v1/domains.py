"""Domain endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.dependencies import authorize_domain, get_domain_service, require_read, require_write
from app.models.domain import DomainConfig, DomainCreate, DomainRead, DomainUpdate
from app.security.auth import Principal
from app.services.domain_service import DomainService

router = APIRouter(prefix="/domains", tags=["domains"])


@router.post("", response_model=DomainConfig, status_code=status.HTTP_201_CREATED)
def create_domain(
    payload: DomainCreate,
    request: Request,
    principal: Principal = Depends(require_write),
    service: DomainService = Depends(get_domain_service),
) -> DomainConfig:
    domain = service.create(payload)
    request.app.state.container.audit.record(
        action="domain.create", resource=f"domain:{domain.name}"
    )
    return domain


@router.get("", response_model=list[DomainRead])
def list_domains(
    request: Request,
    principal: Principal = Depends(require_read),
    service: DomainService = Depends(get_domain_service),
) -> list[DomainRead]:
    return [d for d in service.list() if principal.may_access(d.id, d.name)]


@router.get("/{domain_id}", response_model=DomainRead)
def get_domain(
    domain_id: str,
    request: Request,
    principal: Principal = Depends(require_read),
    service: DomainService = Depends(get_domain_service),
) -> DomainRead:
    domain = service.read(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return domain


@router.put("/{domain_id}", response_model=DomainConfig)
def update_domain(
    domain_id: str,
    payload: DomainUpdate,
    request: Request,
    principal: Principal = Depends(require_write),
    service: DomainService = Depends(get_domain_service),
) -> DomainConfig:
    existing = service.get(domain_id)
    authorize_domain(request, existing.id, existing.name)
    domain = service.update(domain_id, payload)
    request.app.state.container.audit.record(
        action="domain.update", resource=f"domain:{domain.name}"
    )
    return domain


@router.delete("/{domain_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_domain(
    domain_id: str,
    request: Request,
    principal: Principal = Depends(require_write),
    service: DomainService = Depends(get_domain_service),
) -> None:
    existing = service.get(domain_id)
    authorize_domain(request, existing.id, existing.name)
    service.delete(domain_id)
    request.app.state.container.audit.record(
        action="domain.delete", resource=f"domain:{existing.name}"
    )
