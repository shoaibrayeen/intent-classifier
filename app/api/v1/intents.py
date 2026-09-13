"""Intent endpoints, nested under their domain."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.dependencies import (
    authorize_domain,
    get_domain_service,
    get_intent_service,
    require_read,
    require_write,
)
from app.errors import UnavailableError
from app.models.intent import (
    ExampleGenerateRequest,
    ExampleGenerateResponse,
    IntentConfig,
    IntentCreate,
    IntentGenerateRequest,
    IntentGenerateResponse,
    IntentRead,
    IntentUpdate,
)
from app.security.auth import Principal
from app.services.domain_service import DomainService
from app.services.intent_service import IntentService

router = APIRouter(prefix="/domains/{domain_id}/intents", tags=["intents"])


@router.post("", response_model=IntentConfig, status_code=status.HTTP_201_CREATED)
def create_intent(
    domain_id: str,
    payload: IntentCreate,
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> IntentConfig:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.create(domain_id, payload)


@router.get("", response_model=list[IntentRead])
def list_intents(
    domain_id: str,
    request: Request,
    principal: Principal = Depends(require_read),
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> list[IntentRead]:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.list(domain_id)


@router.get("/{intent_id}", response_model=IntentRead)
def get_intent(
    domain_id: str,
    intent_id: str,
    request: Request,
    principal: Principal = Depends(require_read),
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> IntentRead:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.read(domain_id, intent_id)


@router.put("/{intent_id}", response_model=IntentConfig)
def update_intent(
    domain_id: str,
    intent_id: str,
    payload: IntentUpdate,
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> IntentConfig:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    return service.update(domain_id, intent_id, payload)


@router.delete("/{intent_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_intent(
    domain_id: str,
    intent_id: str,
    request: Request,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> None:
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    service.delete(domain_id, intent_id)
    request.app.state.container.audit.record(action="intent.delete", resource=f"intent:{intent_id}")


@router.post("/generate", response_model=IntentGenerateResponse)
async def generate_intents(
    domain_id: str,
    request: Request,
    payload: IntentGenerateRequest | None = None,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
) -> IntentGenerateResponse:
    """Draft this domain's intents with the LLM, with their examples.

    Auto mode. Intents are created and their examples indexed, so the domain is
    classifiable straight away; everything is then editable exactly like a
    hand-written intent. Names that already exist in the domain are skipped and
    reported rather than overwritten. Pass ``dry_run: true`` to see the
    proposal without writing anything.
    """
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    container = request.app.state.container
    authoring = container.intent_authoring
    if not authoring.available:
        raise UnavailableError(
            "no LLM provider is configured; set OPENAI_API_KEY or LLM_PROVIDER=mock, "
            "or create intents manually"
        )

    payload = payload or IntentGenerateRequest()
    response = await authoring.generate_intents(
        domain,
        brief=(payload.brief or "").strip(),
        count=payload.count,
        examples_per_intent=payload.examples_per_intent,
        dry_run=payload.dry_run,
    )
    container.audit.record(
        action="intent.generate",
        resource=f"domain:{domain.name}",
        outcome=response.status,
        dry_run=payload.dry_run,
        intents_created=response.created_count,
        examples_created=response.example_count,
    )
    return response


@router.post("/{intent_id}/examples/generate", response_model=ExampleGenerateResponse)
async def generate_examples(
    domain_id: str,
    intent_id: str,
    request: Request,
    payload: ExampleGenerateRequest | None = None,
    principal: Principal = Depends(require_write),
    domains: DomainService = Depends(get_domain_service),
    service: IntentService = Depends(get_intent_service),
) -> ExampleGenerateResponse:
    """Draft more training examples for one intent.

    Examples are what make an intent findable, so this is the useful companion
    to writing an intent by hand. Anything that duplicates an existing example
    is dropped and reported.
    """
    domain = domains.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    intent = service.get(domain_id, intent_id)
    container = request.app.state.container
    authoring = container.intent_authoring
    if not authoring.available:
        raise UnavailableError(
            "no LLM provider is configured; set OPENAI_API_KEY or LLM_PROVIDER=mock, "
            "or add examples manually"
        )

    payload = payload or ExampleGenerateRequest()
    response = await authoring.generate_examples(
        domain, intent, count=payload.count, dry_run=payload.dry_run
    )
    container.audit.record(
        action="example.generate",
        resource=f"intent:{intent.name}",
        outcome=response.status,
        dry_run=payload.dry_run,
        examples_created=response.created_count,
    )
    return response
