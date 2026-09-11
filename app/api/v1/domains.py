"""Domain endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import get_domain_service
from app.models.domain import DomainConfig, DomainCreate, DomainRead, DomainUpdate
from app.services.domain_service import DomainService

router = APIRouter(prefix="/domains", tags=["domains"])


@router.post("", response_model=DomainConfig, status_code=status.HTTP_201_CREATED)
def create_domain(
    payload: DomainCreate, service: DomainService = Depends(get_domain_service)
) -> DomainConfig:
    return service.create(payload)


@router.get("", response_model=list[DomainRead])
def list_domains(service: DomainService = Depends(get_domain_service)) -> list[DomainRead]:
    return service.list()


@router.get("/{domain_id}", response_model=DomainRead)
def get_domain(domain_id: str, service: DomainService = Depends(get_domain_service)) -> DomainRead:
    return service.read(domain_id)


@router.put("/{domain_id}", response_model=DomainConfig)
def update_domain(
    domain_id: str,
    payload: DomainUpdate,
    service: DomainService = Depends(get_domain_service),
) -> DomainConfig:
    return service.update(domain_id, payload)


@router.delete("/{domain_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_domain(domain_id: str, service: DomainService = Depends(get_domain_service)) -> None:
    service.delete(domain_id)
