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
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    entity_extraction_enabled: bool = False
    llm_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0, le=5)

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
