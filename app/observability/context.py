"""Per-request context.

A request id ties together the access log, the audit record, the trace and the
classification response, so a user reporting "this query returned the wrong
intent" can be traced end to end from that one value.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_principal: ContextVar[str] = ContextVar("principal", default="anonymous")

REQUEST_ID_HEADER = "X-Request-ID"


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


def set_principal(value: str) -> None:
    _principal.set(value)


def get_principal() -> str:
    return _principal.get()
