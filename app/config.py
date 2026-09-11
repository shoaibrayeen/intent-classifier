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

    # --- LLM (wired but unused until entity extraction is enabled) -----
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    entity_extraction_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
