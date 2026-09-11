"""Authentication and domain-level authorization.

API keys, not OAuth: this is a service-to-service component, and an internal
platform without an identity provider is better served by a credential the
caller can rotate than by a login flow nobody operates.

Keys are configured as a single string so they can come from an environment
variable or a secret mount:

    API_KEYS="ops-key:*:admin, contract-svc:contract:classify"
             ^ secret   ^ domains      ^ scopes

``*`` means every domain, or every scope. A key is referred to in logs by its
label (the part before the first colon is the secret, so the label is derived
from a short fingerprint) and never by its value.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from enum import StrEnum

from fastapi import Request

from app.config import Settings
from app.errors import AppError

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


class Scope(StrEnum):
    CLASSIFY = "classify"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


#: admin implies everything; write implies read.
_IMPLIED: dict[str, set[str]] = {
    Scope.ADMIN: {Scope.ADMIN, Scope.WRITE, Scope.READ, Scope.CLASSIFY},
    Scope.WRITE: {Scope.WRITE, Scope.READ, Scope.CLASSIFY},
    Scope.READ: {Scope.READ, Scope.CLASSIFY},
    Scope.CLASSIFY: {Scope.CLASSIFY},
}


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"


class AuthorizationError(AppError):
    status_code = 403
    code = "forbidden"


@dataclass(frozen=True)
class Principal:
    """Who is calling, and what they may touch."""

    label: str
    secret: str = field(repr=False, default="")
    domains: frozenset[str] = frozenset({"*"})
    scopes: frozenset[str] = frozenset({Scope.ADMIN})
    anonymous: bool = False

    def may(self, scope: str) -> bool:
        if self.anonymous:
            return True
        return any(scope in _IMPLIED.get(held, {held}) for held in self.scopes)

    def may_access(self, domain_id: str, domain_name: str = "") -> bool:
        if self.anonymous or "*" in self.domains:
            return True
        return domain_id in self.domains or (
            bool(domain_name) and domain_name.casefold() in self.domains
        )


ANONYMOUS = Principal(label="anonymous", anonymous=True)


def _fingerprint(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:8]


def parse_api_keys(raw: str) -> list[Principal]:
    """Parse the API_KEYS setting. Malformed entries are skipped loudly."""
    principals: list[Principal] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split(":")]
        secret = parts[0]
        if not secret:
            logger.warning("skipping an API key entry with an empty secret")
            continue
        domains = parts[1] if len(parts) > 1 and parts[1] else "*"
        scopes = parts[2] if len(parts) > 2 and parts[2] else Scope.ADMIN

        domain_set = frozenset(d.strip().casefold() for d in domains.split("|") if d.strip())
        scope_set = frozenset(s.strip().casefold() for s in scopes.split("|") if s.strip())
        unknown = scope_set - set(_IMPLIED)
        if unknown:
            logger.warning("API key %s declares unknown scopes: %s", _fingerprint(secret), unknown)
        principals.append(
            Principal(
                label=f"key-{_fingerprint(secret)}",
                secret=secret,
                domains=domain_set or frozenset({"*"}),
                scopes=(scope_set & set(_IMPLIED)) or frozenset({Scope.CLASSIFY}),
            )
        )
    return principals


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._principals = parse_api_keys(settings.api_keys)
        if settings.auth_enabled and not self._principals:
            logger.error(
                "AUTH_ENABLED is true but API_KEYS is empty: every request will be rejected"
            )

    @property
    def enabled(self) -> bool:
        return self._settings.auth_enabled

    @property
    def key_count(self) -> int:
        return len(self._principals)

    def authenticate(self, request: Request) -> Principal:
        if not self.enabled:
            return ANONYMOUS

        presented = request.headers.get(API_KEY_HEADER) or _bearer(request)
        if not presented:
            raise AuthenticationError(f"missing {API_KEY_HEADER} header")

        for principal in self._principals:
            # Constant-time comparison: a timing oracle on an API key is a real
            # way to recover one character at a time.
            if hmac.compare_digest(principal.secret, presented):
                return principal
        raise AuthenticationError("invalid API key")


def _bearer(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""
