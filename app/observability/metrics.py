"""Prometheus metrics.

Deliberately few series. Labels are bounded values only: intent names and
domain ids are unbounded in principle, so they are not used as labels.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

http_requests = Counter(
    "intent_http_requests_total",
    "HTTP requests handled.",
    ["method", "path", "status"],
)

http_latency = Histogram(
    "intent_http_request_duration_seconds",
    "End-to-end HTTP request latency.",
    ["method", "path"],
    buckets=LATENCY_BUCKETS,
)

classifications = Counter(
    "intent_classifications_total",
    "Classification requests by outcome.",
    ["outcome", "strategy"],
)

classification_latency = Histogram(
    "intent_classification_duration_seconds",
    "Classification latency, excluding HTTP overhead.",
    ["strategy"],
    buckets=LATENCY_BUCKETS,
)

stage_latency = Histogram(
    "intent_stage_duration_seconds",
    "Latency of one pipeline stage.",
    ["stage"],
    buckets=LATENCY_BUCKETS,
)

confidence_observed = Histogram(
    "intent_confidence",
    "Confidence of every classification, including UNKNOWN results.",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

extractions = Counter(
    "intent_entity_extractions_total",
    "Entity extraction attempts by status.",
    ["status"],
)

index_version = Gauge(
    "intent_index_version",
    "Current BM25 index version per domain.",
    ["domain"],
)

index_documents = Gauge(
    "intent_index_documents",
    "Documents in a domain's BM25 index.",
    ["domain"],
)

auth_failures = Counter(
    "intent_auth_failures_total",
    "Rejected requests by reason.",
    ["reason"],
)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
