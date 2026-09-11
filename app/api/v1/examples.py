"""Training-example endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.dependencies import (
    authorize_domain,
    get_domain_service,
    get_example_service,
    require_read,
    require_write,
)
from app.models.example import Example, ExampleCreate
from app.security.auth import Principal
from app.services.domain_service import DomainService
from app.services.example_service import ExampleService

router = APIRouter(prefix="/domains/{domain_id}/intents/{intent_id}/examples", tags=["examples"])


@router.post("", response_model=Example, status_code=status.HTTP_201_CREATED)
def add_example(
    domain_id: str,
    intent_id: str,
    payload: ExampleCreate,
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> Example:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.add(domain_id, intent_id, payload)


@router.post("/bulk", response_model=list[Example], status_code=status.HTTP_201_CREATED)
def add_examples(
    domain_id: str,
    intent_id: str,
    payload: list[ExampleCreate],
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> list[Example]:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.add_many(domain_id, intent_id, payload)


@router.get("", response_model=list[Example])
def list_examples(
    domain_id: str,
    intent_id: str,
    request: Request,
    principal: Principal = Depends(require_read),
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> list[Example]:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.list(domain_id, intent_id)


@router.delete("/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_example(
    domain_id: str,
    intent_id: str,
    example_id: str,
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> None:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    service.delete(domain_id, intent_id, example_id)
