"""Application configuration, loaded from environment / .env."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Every value has a working default.

    The only value an operator must supply is ``OPENAI_API_KEY``, and only once
    entity extraction is enabled (Phase 3).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "intent-classifier"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # --- storage -------------------------------------------------------
    chroma_mode: Literal["persistent", "ephemeral"] = "persistent"
    chroma_path: str = "./data/chroma"
    index_dir: str = "./data/indexes"

    # --- embedding model ----------------------------------------------
    model_name: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: str = "./data/models"

    # --- retrieval -----------------------------------------------------
    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    agg_top_n: int = Field(default=3, ge=1, le=20)

    # --- confidence / unknown -----------------------------------------
    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    min_dense_similarity: float = Field(default=0.60, ge=0.0, le=1.0)
    dense_sim_floor: float = Field(default=0.50, ge=0.0, le=1.0)
    dense_sim_ceil: float = Field(default=0.90, ge=0.0, le=1.0)

    w_dense: float = 0.35
    w_rrf: float = 0.25
    w_margin: float = 0.25
    w_support: float = 0.15

    # --- LLM / entity extraction ---------------------------------------
    #: auto = openai when a key is set, otherwise none. mock = offline
    #: rule-based provider for local runs and tests (no key needed).
    llm_provider: Literal["auto", "openai", "mock", "none"] = "auto"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    entity_extraction_enabled: bool = False
    llm_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)

    #: Load the demo catalogue at startup when the store is empty. Seeding
    #: in-process matters for the container: embedded Chroma is single-writer,
    #: so seeding from a second process while the server holds the store leaves
    #: the server unable to read the new vectors until it restarts.
    seed_on_startup: bool = False

    # --- multi-turn sessions --------------------------------------------
    sessions_enabled: bool = True
    #: Earlier turns shown to the extractor as conversation history.
    session_history_turns: int = Field(default=5, ge=0, le=50)
    #: Turns kept per session; older ones are pruned on write.
    session_max_turns: int = Field(default=50, ge=1, le=1000)
    #: Turns older than this are dropped, checked lazily on write.
    session_ttl_seconds: int = Field(default=7 * 24 * 3600, ge=60)
    #: When a short follow-up cannot be classified on its own, retry it with
    #: the previous question prepended.
    context_retrieval_enabled: bool = True
    #: Carry entities from earlier turns into a new intent that accepts them.
    entity_carry_over_enabled: bool = True

    # --- A/B testing of retrieval strategies ---------------------------
    ab_testing_enabled: bool = False
    default_strategy: str = "hybrid_rrf"
    ab_variants: str = "hybrid_rrf,dense_only"

    # --- security -------------------------------------------------------
    auth_enabled: bool = False
    #: "secret:domains:scopes" entries, comma separated. See app/security/auth.py
    api_keys: str = ""
    #: The HTML UI is a human surface; when auth is on it needs a key too,
    #: unless this is set (useful behind an authenticating proxy).
    ui_auth_exempt: bool = False

    # --- observability ---------------------------------------------------
    metrics_enabled: bool = True
    audit_log_enabled: bool = True
    audit_log_path: str = "./data/audit/audit.jsonl"
    #: User queries can carry personal data, so their text is opt-in.
    audit_log_query_text: bool = False
    tracing_enabled: bool = False
    otlp_endpoint: str = "http://localhost:4317"


@lru_cache
def get_settings() -> Settings:
    return Settings()
