"""Intent endpoints, nested under their domain."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import get_domain_service, get_intent_service
from app.models.intent import IntentConfig, IntentCreate, IntentRead, IntentUpdate
from app.services.domain_service import DomainService
from app.services.intent_service import IntentService

router = APIRouter(prefix="/domains/{domain_id}/intents", tags=["intents"])


@router.post("", response_model=IntentConfig, status_code=status.HTTP_201_CREATED)
def create_intent(
    domain_id: str,
    payload: IntentCreate,
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> IntentConfig:
    domains.get(domain_id)
    return service.create(domain_id, payload)


@router.get("", response_model=list[IntentRead])
def list_intents(
    domain_id: str,
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> list[IntentRead]:
    domains.get(domain_id)
    return service.list(domain_id)


@router.get("/{intent_id}", response_model=IntentRead)
def get_intent(
    domain_id: str,
    intent_id: str,
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> IntentRead:
    domains.get(domain_id)
    return service.read(domain_id, intent_id)


@router.put("/{intent_id}", response_model=IntentConfig)
def update_intent(
    domain_id: str,
    intent_id: str,
    payload: IntentUpdate,
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> IntentConfig:
    domains.get(domain_id)
    return service.update(domain_id, intent_id, payload)


@router.delete("/{intent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_intent(
    domain_id: str,
    intent_id: str,
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> None:
    domains.get(domain_id)
    service.delete(domain_id, intent_id)
