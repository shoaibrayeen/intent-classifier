"""FastAPI dependencies: container access, authentication, authorization."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Request

from app.observability import metrics
from app.observability.context import set_principal
from app.security.auth import (
    ANONYMOUS,
    AuthenticationError,
    AuthorizationError,
    Principal,
    Scope,
)
from app.services.classification_service import ClassificationService
from app.services.container import Container
from app.services.domain_service import DomainService
from app.services.example_service import ExampleService
from app.services.index_manager import IndexManager
from app.services.intent_service import IntentService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_domain_service(request: Request) -> DomainService:
    return get_container(request).domains


def get_intent_service(request: Request) -> IntentService:
    return get_container(request).intents


def get_example_service(request: Request) -> ExampleService:
    return get_container(request).examples


def get_index_manager(request: Request) -> IndexManager:
    return get_container(request).index_manager


def get_classification_service(request: Request) -> ClassificationService:
    return get_container(request).classification


def get_principal(request: Request) -> Principal:
    """Authenticate once per request and remember who it was."""
    cached = getattr(request.state, "principal", None)
    if cached is not None:
        return cached
    container = get_container(request)
    try:
        principal = container.authenticator.authenticate(request)
    except AuthenticationError as exc:
        metrics.auth_failures.labels(reason="unauthenticated").inc()
        container.audit.record(
            action="auth", resource=request.url.path, outcome="denied", detail=exc.message
        )
        raise
    request.state.principal = principal
    set_principal(principal.label)
    return principal


def require_scope(scope: str) -> Callable[[Principal], Principal]:
    """Dependency factory enforcing one scope."""

    def dependency(request: Request, principal: Principal = Depends(get_principal)) -> Principal:
        if not principal.may(scope):
            metrics.auth_failures.labels(reason="scope").inc()
            get_container(request).audit.record(
                action="authz",
                resource=request.url.path,
                outcome="denied",
                required_scope=scope,
            )
            raise AuthorizationError(f"this key does not hold the '{scope}' scope")
        return principal

    return dependency


require_classify = require_scope(Scope.CLASSIFY)
require_read = require_scope(Scope.READ)
require_write = require_scope(Scope.WRITE)
require_admin = require_scope(Scope.ADMIN)


def authorize_domain(request: Request, domain_id: str, domain_name: str = "") -> None:
    """Enforce the per-key domain allowlist."""
    principal = getattr(request.state, "principal", None) or ANONYMOUS
    if principal.may_access(domain_id, domain_name):
        return
    metrics.auth_failures.labels(reason="domain").inc()
    get_container(request).audit.record(
        action="authz", resource=f"domain:{domain_id}", outcome="denied"
    )
    raise AuthorizationError("this key is not allowed to access that domain")
