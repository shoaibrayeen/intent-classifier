"""Request / response models for the classification API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

UNKNOWN_INTENT = "UNKNOWN"


class UnknownReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    LOW_SIMILARITY = "low_similarity"
    NO_EXAMPLES = "no_examples"


class ClassifyRequest(BaseModel):
    """``domain`` accepts either a domain id or a domain name."""

    domain: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=2000)
    extract_entities: bool | None = Field(
        default=None,
        description=(
            "Override entity extraction for this request. None follows the "
            "service default. Extraction never runs for an UNKNOWN result."
        ),
    )
    variant: str | None = Field(
        default=None,
        max_length=60,
        description="Force a retrieval strategy variant instead of the assigned one.",
    )

    @field_validator("domain", "text", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class IntentScore(BaseModel):
    intent: str
    intent_id: str
    score: float
    supporting_examples: int
    best_similarity: float


class ConfidenceBreakdown(BaseModel):
    s_dense: float
    s_rrf: float
    s_margin: float
    s_support: float
    confidence: float
    agg_top: float
    agg_second: float
    best_similarity: float


class DenseHit(BaseModel):
    example_id: str
    intent_id: str
    intent_name: str
    text: str
    similarity: float
    rank: int


class Bm25Hit(BaseModel):
    example_id: str
    intent_id: str
    intent_name: str
    text: str
    score: float
    rank: int


class EntityExtractionInfo(BaseModel):
    """Why the entities field looks the way it does."""

    status: Literal["ok", "skipped", "disabled", "unavailable", "failed"]
    model: str | None = None
    detail: str | None = None
    rejected: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    """The call an intent maps to. This service never performs it."""

    name: str
    version: str = "v1"
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    ready: bool = True


class StageTimings(BaseModel):
    """Wall-clock milliseconds per pipeline stage."""

    embed_ms: float = 0.0
    dense_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    extraction_ms: float = 0.0
    total_ms: float = 0.0


class ClassifyDebug(BaseModel):
    normalized_text: str
    tokens: list[str]
    dense_hits: list[DenseHit]
    bm25_hits: list[Bm25Hit]
    rrf_intents: list[IntentScore]
    confidence_breakdown: ConfidenceBreakdown
    index_version: int
    strategy: str = "hybrid_rrf"
    timings: StageTimings = Field(default_factory=StageTimings)


class ClassifyResponse(BaseModel):
    domain_id: str
    domain: str
    intent: str
    intent_id: str | None = None
    confidence: float
    entities: dict[str, Any] = Field(default_factory=dict)
    entity_extraction: EntityExtractionInfo | None = None
    tool: ToolCall | None = None
    reason: UnknownReason | None = None
    top_intents: list[IntentScore] = Field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str | None = None
    debug: ClassifyDebug | None = None
