"""API v1 router assembly."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import classify, domains, examples, health, index, intents, sessions

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(classify.router)
api_router.include_router(domains.router)
api_router.include_router(intents.router)
api_router.include_router(examples.router)
api_router.include_router(index.router)
api_router.include_router(sessions.router)
