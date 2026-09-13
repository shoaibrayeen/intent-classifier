"""Application configuration, loaded from environment / .env.

Every setting carries its own description and grouping, so
``scripts/build_properties.py`` can render docs/properties.html straight from
this model. A setting documented anywhere else could drift; one documented here
cannot.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def setting(
    default: Any,
    description: str,
    *,
    group: str,
    required_when: str | None = None,
    **constraints: Any,
) -> Any:
    """A configuration field that documents itself.

    ``required_when`` names the condition that makes a value mandatory. Nothing
    here is required to start the service: the defaults run it end to end.
    """
    extra: dict[str, Any] = {"group": group}
    if required_when:
        extra["required_when"] = required_when
    return Field(default=default, description=description, json_schema_extra=extra, **constraints)


CORE = "Core"
STORAGE = "Storage"
EMBEDDING = "Embedding model"
RETRIEVAL = "Retrieval"
CONFIDENCE = "Confidence and UNKNOWN"
LLM = "Entity extraction and generation"
SESSIONS = "Multi-turn sessions"
STRATEGY = "Retrieval strategy"
SECURITY = "Security"
OBSERVABILITY = "Observability"


class Settings(BaseSettings):
    """All runtime configuration. Every value has a working default.

    The service starts and classifies with no ``.env`` at all. A value belongs
    in ``.env`` only once you want it to differ from the default below.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- core ----------------------------------------------------------
    app_name: str = setting(
        "intent-classifier", "Service name, used in logs and traces.", group=CORE
    )
    app_version: str = setting("0.1.0", "Reported by /api/v1/health.", group=CORE)
    log_level: str = setting(
        "INFO", "Python logging level: DEBUG, INFO, WARNING, ERROR.", group=CORE
    )

    # --- storage -------------------------------------------------------
    chroma_mode: Literal["persistent", "ephemeral"] = setting(
        "persistent",
        "persistent keeps the catalogue on disk; ephemeral is in-memory, used by the tests.",
        group=STORAGE,
    )
    chroma_path: str = setting(
        "./data/chroma", "Where the embedded Chroma store lives.", group=STORAGE
    )
    index_dir: str = setting(
        "./data/indexes",
        "Reserved for persisted BM25 artefacts. Indexes are rebuilt from Chroma today.",
        group=STORAGE,
    )

    # --- embedding model ----------------------------------------------
    model_name: str = setting(
        "BAAI/bge-small-en-v1.5",
        "FastEmbed model. Changing it invalidates existing vectors: reindex every domain after.",
        group=EMBEDDING,
    )
    model_cache_dir: str = setting(
        "./data/models",
        "Where the model is cached. The default OS temp directory is purged, so this is set.",
        group=EMBEDDING,
    )

    # --- retrieval -----------------------------------------------------
    retrieval_top_k: int = setting(
        10, "Examples each retriever returns before fusion.", group=RETRIEVAL, ge=1, le=100
    )
    rrf_k: int = setting(
        60,
        "Reciprocal Rank Fusion constant. Smaller sharpens the top ranks.",
        group=RETRIEVAL,
        ge=1,
    )
    agg_top_n: int = setting(
        3,
        "How many of an intent's best examples are summed into its score.",
        group=RETRIEVAL,
        ge=1,
        le=20,
    )

    # --- confidence / unknown -----------------------------------------
    confidence_threshold: float = setting(
        0.55,
        "Below this a result is UNKNOWN. Raise it to refuse more, lower it to guess more.",
        group=CONFIDENCE,
        ge=0.0,
        le=1.0,
    )
    min_dense_similarity: float = setting(
        0.60,
        "Hard floor on best cosine similarity. This is what rejects off-domain queries.",
        group=CONFIDENCE,
        ge=0.0,
        le=1.0,
    )
    dense_sim_floor: float = setting(
        0.50, "Similarity that maps to a dense score of 0.", group=CONFIDENCE, ge=0.0, le=1.0
    )
    dense_sim_ceil: float = setting(
        0.90, "Similarity that maps to a dense score of 1.", group=CONFIDENCE, ge=0.0, le=1.0
    )
    w_dense: float = setting(0.35, "Weight of the dense signal in confidence.", group=CONFIDENCE)
    w_rrf: float = setting(0.25, "Weight of the fusion signal in confidence.", group=CONFIDENCE)
    w_margin: float = setting(
        0.25, "Weight of the lead over the runner-up intent.", group=CONFIDENCE
    )
    w_support: float = setting(
        0.15, "Weight of how many examples back the top intent.", group=CONFIDENCE
    )

    # --- LLM -----------------------------------------------------------
    llm_provider: Literal["auto", "openai", "mock", "none"] = setting(
        "auto",
        "auto uses OpenAI when a key is set. mock is an offline rule-based provider needing "
        "no key. none disables generation entirely.",
        group=LLM,
    )
    openai_api_key: str = setting(
        "",
        "OpenAI API key. Classification never needs one; extraction and generation do.",
        group=LLM,
        required_when="LLM_PROVIDER resolves to openai",
    )
    openai_base_url: str = setting(
        "https://api.openai.com/v1",
        "Override to point at a compatible endpoint.",
        group=LLM,
    )
    openai_model: str = setting("gpt-4o-mini", "Model used for extraction and drafting.", group=LLM)
    entity_extraction_enabled: bool = setting(
        False,
        "Extract entities after a confident match. Off by default: it adds a network call.",
        group=LLM,
    )
    llm_timeout_seconds: float = setting(
        10.0, "How long a provider call may take before giving up.", group=LLM, gt=0
    )
    llm_max_retries: int = setting(1, "Retries on a failed provider call.", group=LLM, ge=0, le=5)
    evaluation_dataset_path: str = setting(
        "",
        "A JSON file with a 'cases' array to evaluate against. Empty uses the set that "
        "ships with the application.",
        group=CORE,
    )
    seed_on_startup: bool = setting(
        False,
        "Load the demo catalogue when the store is empty. Seeding runs in the server's own "
        "process because embedded Chroma is single-writer. docker-compose turns this on.",
        group=CORE,
    )

    # --- sessions ------------------------------------------------------
    sessions_enabled: bool = setting(
        True, "Record conversation turns when a request carries a session id.", group=SESSIONS
    )
    session_history_turns: int = setting(
        5, "Earlier turns shown to the extractor as history.", group=SESSIONS, ge=0, le=50
    )
    session_max_turns: int = setting(
        50, "Turns kept per session; older ones are pruned on write.", group=SESSIONS, ge=1, le=1000
    )
    session_ttl_seconds: int = setting(
        7 * 24 * 3600, "Turns older than this are dropped.", group=SESSIONS, ge=60
    )
    context_retrieval_enabled: bool = setting(
        True,
        "Retry an UNKNOWN follow-up with the previous question prepended.",
        group=SESSIONS,
    )
    context_rescue_margin: float = setting(
        0.10,
        "How far above the threshold a contextual retry must score. The retry includes the "
        "previous question, which alone can carry the match, so a rescue has to prove itself.",
        group=SESSIONS,
        ge=0.0,
        le=0.5,
    )
    context_followup_max_words: int = setting(
        3,
        "A fragment this short counts as referential even without a pronoun.",
        group=SESSIONS,
        ge=0,
        le=20,
    )
    entity_carry_over_enabled: bool = setting(
        True,
        "Carry entities from earlier turns into a new intent that accepts them.",
        group=SESSIONS,
    )

    # --- strategy ------------------------------------------------------
    default_strategy: str = setting(
        "hybrid_rrf",
        "hybrid_rrf is the measured best. auto answers with hybrid and falls back only when "
        "hybrid cannot place a query, reporting which one resolved it.",
        group=STRATEGY,
    )
    auto_rescue_margin: float = setting(
        0.10,
        "How far above the threshold an auto fallback must score. Without it, auto trades "
        "UNKNOWN detection for accuracy.",
        group=STRATEGY,
        ge=0.0,
        le=0.5,
    )
    ab_testing_enabled: bool = setting(
        False,
        "Assign a strategy per request by hashing the request id, so it stays reproducible.",
        group=STRATEGY,
    )
    ab_variants: str = setting(
        "hybrid_rrf,dense_only", "Comma-separated strategies to assign between.", group=STRATEGY
    )

    # --- security ------------------------------------------------------
    auth_enabled: bool = setting(
        False, "Require an API key on every request except /health.", group=SECURITY
    )
    api_keys: str = setting(
        "",
        'Comma-separated "secret:domains:scopes" entries. "*" means all. '
        "Scopes nest: admin > write > read > classify.",
        group=SECURITY,
        required_when="AUTH_ENABLED is true",
    )
    ui_auth_exempt: bool = setting(
        False,
        "Let the HTML UI through without a key, for use behind an authenticating proxy.",
        group=SECURITY,
    )

    # --- observability -------------------------------------------------
    metrics_enabled: bool = setting(True, "Serve Prometheus metrics.", group=OBSERVABILITY)
    audit_log_enabled: bool = setting(
        True, "Record an append-only trail of changes and classifications.", group=OBSERVABILITY
    )
    audit_log_path: str = setting(
        "./data/audit/audit.jsonl", "Where the audit trail is written.", group=OBSERVABILITY
    )
    audit_log_query_text: bool = setting(
        False,
        "Record the raw query text. Off by default: user queries can carry personal data.",
        group=OBSERVABILITY,
    )
    tracing_enabled: bool = setting(
        False,
        "Export OpenTelemetry spans. Off by default: with no collector the exporter retries "
        "in the background and adds latency.",
        group=OBSERVABILITY,
    )
    otlp_endpoint: str = setting(
        "http://localhost:4317", "Where traces are exported.", group=OBSERVABILITY
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
