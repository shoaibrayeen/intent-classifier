"""Liveness / readiness."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_container
from app.services.container import Container

router = APIRouter(tags=["health"])


@router.get("/health")
def health(container: Container = Depends(get_container)) -> dict:
    return {
        "status": "ok",
        "version": container.settings.app_version,
        "chroma": "ok" if container.store.heartbeat() else "unavailable",
        "chroma_mode": container.settings.chroma_mode,
        "model": container.embedder.model_name,
        "model_loaded": True,
        "embedding_dimension": container.embedder.dimension,
        "entity_extraction_enabled": container.entity_extractor.enabled,
    }
