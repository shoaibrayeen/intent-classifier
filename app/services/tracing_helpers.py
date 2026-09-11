"""Timing plus tracing for one pipeline stage, in a single statement."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator

from app.observability import metrics, tracing


@contextlib.contextmanager
def stage(name: str) -> Iterator[Callable[[], float]]:
    """Time a stage, record it, and yield a getter for the elapsed ms."""
    started = time.perf_counter()
    elapsed_ms = 0.0

    def elapsed() -> float:
        return elapsed_ms

    with tracing.span(name):
        try:
            yield elapsed
        finally:
            seconds = time.perf_counter() - started
            elapsed_ms = round(seconds * 1000, 3)
            metrics.stage_latency.labels(stage=name).observe(seconds)
