"""Wiring: one place that builds every collaborator the app needs."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.db.chroma import ChromaStore
from app.repositories.chroma_domains import ChromaDomainRepository
from app.repositories.chroma_examples import ChromaExampleRepository
from app.repositories.chroma_intents import ChromaIntentRepository
from app.services.classifier import IntentClassifier
from app.services.domain_service import DomainService
from app.services.embedding import Embedder
from app.services.example_service import ExampleService
from app.services.index_manager import IndexManager
from app.services.intent_service import IntentService
from app.services.llm.client import OpenAIChatClient
from app.services.llm.entity_extractor import EntityExtractor


@dataclass
class Container:
    settings: Settings
    store: ChromaStore
    embedder: Embedder
    domains: DomainService
    intents: IntentService
    examples: ExampleService
    classifier: IntentClassifier
    index_manager: IndexManager
    entity_extractor: EntityExtractor


def build_container(settings: Settings) -> Container:
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

    llm_client = OpenAIChatClient(settings)
    entity_extractor = EntityExtractor(settings, llm_client)

    return Container(
        settings=settings,
        store=store,
        embedder=embedder,
        domains=domain_service,
        intents=intent_service,
        examples=example_service,
        classifier=classifier,
        index_manager=index_manager,
        entity_extractor=entity_extractor,
    )
