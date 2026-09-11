"""OpenTelemetry tracing, off by default.

Tracing is opt-in because a collector is infrastructure, not a library: with no
endpoint configured the exporter would retry in the background and add latency
to every request. When disabled, `span()` is a no-op context manager, so call
sites read the same either way.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

_tracer: Any = None
_enabled = False


def setup(settings: Settings) -> bool:
    """Configure the exporter. Returns whether tracing is active."""
    global _tracer, _enabled
    if not settings.tracing_enabled:
        _enabled = False
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {"service.name": settings.app_name, "service.version": settings.app_version}
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(settings.app_name)
        _enabled = True
        logger.info("tracing enabled, exporting to %s", settings.otlp_endpoint)
    except Exception:
        logger.exception("tracing could not be initialised; continuing without it")
        _enabled = False
    return _enabled


def enabled() -> bool:
    return _enabled


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    if not _enabled or _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield


def shutdown() -> None:
    global _enabled
    if not _enabled:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        logger.exception("tracing shutdown failed")
    finally:
        _enabled = False
