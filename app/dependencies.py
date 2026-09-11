"""FastAPI dependency accessors over the application container."""

from __future__ import annotations

from fastapi import Request

from app.services.container import Container
from app.services.domain_service import DomainService
from app.services.example_service import ExampleService
from app.services.index_manager import IndexManager
from app.services.intent_service import IntentService


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_domain_service(request: Request) -> DomainService:
    return get_container(request).domains


def get_intent_service(request: Request) -> IntentService:
    return get_container(request).intents


def get_example_service(request: Request) -> ExampleService:
    return get_container(request).examples


def get_index_manager(request: Request) -> IndexManager:
    return get_container(request).index_manager
