"""Wiring: one place that builds every collaborator the app needs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.db.chroma import ChromaStore
from app.observability.audit import AuditLog
from app.repositories.chroma_domains import ChromaDomainRepository
from app.repositories.chroma_examples import ChromaExampleRepository
from app.repositories.chroma_intents import ChromaIntentRepository
from app.repositories.chroma_sessions import ChromaSessionRepository
from app.security.auth import Authenticator
from app.services.classification_service import ClassificationService
from app.services.classifier import IntentClassifier
from app.services.domain_service import DomainService
from app.services.embedding import Embedder
from app.services.example_service import ExampleService
from app.services.index_manager import IndexManager
from app.services.intent_service import IntentService
from app.services.llm.client import LLMClient, OpenAIChatClient, StubLLMClient
from app.services.llm.entity_extractor import EntityExtractor
from app.services.llm.mock import MockLLMClient
from app.services.session_service import SessionService
from app.services.strategies import StrategySelector
from app.services.tool_router import ToolRouter

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    store: ChromaStore
    embedder: Embedder
    domains: DomainService
    intents: IntentService
    examples: ExampleService
    classifier: IntentClassifier
    classification: ClassificationService
    index_manager: IndexManager
    entity_extractor: EntityExtractor
    tool_router: ToolRouter
    strategies: StrategySelector
    audit: AuditLog
    authenticator: Authenticator
    sessions: SessionService


def build_container(settings: Settings, llm_client: LLMClient | None = None) -> Container:
    store = ChromaStore(settings)
    domain_repo = ChromaDomainRepository(store)
    intent_repo = ChromaIntentRepository(store)
    example_repo = ChromaExampleRepository(store)

    index_manager = IndexManager(example_repo, intent_repo)
    embedder = Embedder(settings)

    domain_service = DomainService(store, domain_repo, intent_repo, example_repo, index_manager)
    intent_service = IntentService(store, intent_repo, example_repo, index_manager)
    example_service = ExampleService(store, example_repo, intent_service, embedder, index_manager)
    classifier = IntentClassifier(settings, embedder, example_repo, intent_repo, index_manager)

    if llm_client is None:
        llm_client = select_llm_client(settings)
    entity_extractor = EntityExtractor(settings, llm_client)
    tool_router = ToolRouter()
    strategies = StrategySelector(settings)
    audit = AuditLog(settings)
    authenticator = Authenticator(settings)
    sessions = SessionService(settings, ChromaSessionRepository(store))

    classification = ClassificationService(
        settings=settings,
        classifier=classifier,
        domains=domain_service,
        extractor=entity_extractor,
        router=tool_router,
        strategies=strategies,
        audit=audit,
        sessions=sessions,
    )

    return Container(
        settings=settings,
        store=store,
        embedder=embedder,
        domains=domain_service,
        intents=intent_service,
        examples=example_service,
        classifier=classifier,
        classification=classification,
        index_manager=index_manager,
        entity_extractor=entity_extractor,
        tool_router=tool_router,
        strategies=strategies,
        audit=audit,
        authenticator=authenticator,
        sessions=sessions,
    )


def select_llm_client(settings: Settings) -> LLMClient:
    """Pick the extraction provider from LLM_PROVIDER.

    ``auto`` follows the key: OpenAI when one is set, otherwise nothing. ``mock``
    is the offline rule-based provider, so the whole flow runs without a key.
    """
    provider = settings.llm_provider
    if provider == "auto":
        provider = "openai" if settings.openai_api_key else "none"
    if provider == "mock":
        logger.warning("LLM_PROVIDER=mock: entity extraction uses offline rules, not a model")
        return MockLLMClient()
    if provider == "openai":
        if not settings.openai_api_key:
            logger.error("LLM_PROVIDER=openai but OPENAI_API_KEY is empty; extraction disabled")
            return StubLLMClient()
        return OpenAIChatClient(settings)
    return StubLLMClient()
