"""Training-example endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import get_domain_service, get_example_service
from app.models.example import Example, ExampleCreate
from app.services.domain_service import DomainService
from app.services.example_service import ExampleService

router = APIRouter(prefix="/domains/{domain_id}/intents/{intent_id}/examples", tags=["examples"])


@router.post("", response_model=Example, status_code=status.HTTP_201_CREATED)
def add_example(
    domain_id: str,
    intent_id: str,
    payload: ExampleCreate,
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> Example:
    domains.get(domain_id)
    return service.add(domain_id, intent_id, payload)


@router.post("/bulk", response_model=list[Example], status_code=status.HTTP_201_CREATED)
def add_examples(
    domain_id: str,
    intent_id: str,
    payload: list[ExampleCreate],
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> list[Example]:
    domains.get(domain_id)
    return service.add_many(domain_id, intent_id, payload)


@router.get("", response_model=list[Example])
def list_examples(
    domain_id: str,
    intent_id: str,
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> list[Example]:
    domains.get(domain_id)
    return service.list(domain_id, intent_id)


@router.delete("/{example_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_example(
    domain_id: str,
    intent_id: str,
    example_id: str,
    domains: DomainService = Depends(get_domain_service),
    service: ExampleService = Depends(get_example_service),
) -> None:
    domains.get(domain_id)
    service.delete(domain_id, intent_id, example_id)
