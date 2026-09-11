"""Domain endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status
from starlette.concurrency import run_in_threadpool

from app.dependencies import authorize_domain, get_domain_service, require_read, require_write
from app.errors import InvalidInputError, UnavailableError
from app.models.domain import (
    DomainConfig,
    DomainCreate,
    DomainRead,
    DomainUpdate,
    InstructionGenerateRequest,
    InstructionGenerateResponse,
)
from app.security.auth import Principal
from app.services.domain_service import DomainService
from app.services.llm.instruction_generator import brief_for

router = APIRouter(prefix="/domains", tags=["domains"])


@router.post("", response_model=DomainConfig, status_code=status.HTTP_201_CREATED)
async def create_domain(
    payload: DomainCreate,
    request: Request,
    principal: Principal = Depends(require_write),
    service: DomainService = Depends(get_domain_service),
) -> DomainConfig:
    """Create a domain.

    With ``generate_instructions: true`` the extraction instructions are
    drafted from the name and description by the configured LLM provider and
    saved with the domain. Explicit instructions in the payload win over
    generation. Requires a provider (``OPENAI_API_KEY`` or ``LLM_PROVIDER=mock``).
    """
    container = request.app.state.container
    generator = container.instruction_generator

    wants_generation = payload.generate_instructions and not (
        payload.system_instructions.strip() or payload.user_instructions.strip()
    )
    if wants_generation and not generator.available:
        # Checked before anything is written, so a config problem cannot leave
        # a half-configured domain behind.
        raise UnavailableError(
            "generate_instructions was requested but no LLM provider is configured; "
            "set OPENAI_API_KEY (or LLM_PROVIDER=mock), or create the domain without it"
        )

    domain = await run_in_threadpool(service.create, payload)
    generation_status = None
    if wants_generation:
        result = await generator.generate(domain.name, brief_for(domain))
        generation_status = result.status
        if result.status == "ok":
            domain = await run_in_threadpool(
                service.update,
                domain.id,
                DomainUpdate(
                    system_instructions=result.system_instructions,
                    user_instructions=result.user_instructions,
                ),
            )
        # A provider failure never destroys the creation: the domain exists,
        # instructions stay empty, and the caller can see both in the response.

    container.audit.record(
        action="domain.create",
        resource=f"domain:{domain.name}",
        instructions_generated=generation_status,
    )
    return domain


@router.post(
    "/{domain_id}/instructions/generate",
    response_model=InstructionGenerateResponse,
)
async def generate_instructions(
    domain_id: str,
    request: Request,
    payload: InstructionGenerateRequest | None = None,
    principal: Principal = Depends(require_write),
    service: DomainService = Depends(get_domain_service),
) -> InstructionGenerateResponse:
    """Draft this domain's extraction instructions from its name and description.

    The draft is saved immediately and returned for review; edit it with
    ``PUT /domains/{domainId}`` or on the domain page. ``brief`` overrides the
    stored description as the generator's input.
    """
    domain = service.get(domain_id)
    authorize_domain(request, domain.id, domain.name)
    container = request.app.state.container
    generator = container.instruction_generator

    if not generator.available:
        raise UnavailableError(
            "no LLM provider is configured; set OPENAI_API_KEY or LLM_PROVIDER=mock"
        )
    brief = brief_for(domain, payload.brief if payload else None)
    if not brief:
        raise InvalidInputError(
            "this domain has no description; provide a 'brief' describing what "
            "users do in this domain"
        )

    result = await generator.generate(domain.name, brief)
    saved = False
    if result.status == "ok":
        await run_in_threadpool(
            service.update,
            domain_id,
            DomainUpdate(
                system_instructions=result.system_instructions,
                user_instructions=result.user_instructions,
            ),
        )
        saved = True

    container.audit.record(
        action="domain.instructions.generate",
        resource=f"domain:{domain.name}",
        outcome=result.status,
    )
    return InstructionGenerateResponse(
        domain_id=domain_id,
        status=result.status,
        detail=result.detail,
        provider=result.provider,
        system_instructions=result.system_instructions,
        user_instructions=result.user_instructions,
        saved=saved,
    )


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
