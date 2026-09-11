"""Shared fixtures. Tests always run against an in-memory Chroma store."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

os.environ["CHROMA_MODE"] = "ephemeral"
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENTITY_EXTRACTION_ENABLED"] = "false"
os.environ["AUTH_ENABLED"] = "false"
os.environ["AUDIT_LOG_ENABLED"] = "false"
os.environ.setdefault("MODEL_CACHE_DIR", "./data/models")

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def make_settings(**overrides) -> Settings:
    """Test defaults: in-memory store, no auth, no audit file, no LLM."""
    base = {
        "chroma_mode": "ephemeral",
        "model_cache_dir": "./data/models",
        "auth_enabled": False,
        "audit_log_enabled": False,
        "entity_extraction_enabled": False,
        "openai_api_key": "",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def app_factory():
    """Build an app with custom settings and/or a scripted LLM client."""
    created = []

    def build(settings_overrides: dict | None = None, llm_client=None):
        app = create_app(make_settings(**(settings_overrides or {})), llm_client=llm_client)
        client = TestClient(app)
        client.__enter__()
        app.state.container.store.reset_all()
        app.state.container.index_manager.clear()
        created.append(client)
        return app, client

    yield build
    for client in created:
        client.__exit__(None, None, None)


@pytest.fixture
def client(settings: Settings):
    """A fresh app per test.

    Chroma caches one in-memory system per settings object, so every ephemeral
    client in this process shares the same store. Clearing it on entry is what
    actually isolates the tests.
    """
    app = create_app(settings)
    with TestClient(app) as test_client:
        app.state.container.store.reset_all()
        app.state.container.index_manager.clear()
        yield test_client


def seed_contract_domain(client: TestClient) -> dict:
    """A small contract domain with two intents and their examples."""
    domain = client.post(
        "/api/v1/domains", json={"name": "contract", "description": "Contracts"}
    ).json()

    search = client.post(
        f"/api/v1/domains/{domain['id']}/intents",
        json={
            "name": "CONTRACT_SEARCH",
            "description": "Search contracts",
            "tool": {"name": "search_contracts", "version": "v1"},
            "entity_schema": {
                "counterparty": {"type": "string", "required": True},
                "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
            },
        },
    ).json()
    expiry = client.post(
        f"/api/v1/domains/{domain['id']}/intents",
        json={
            "name": "CONTRACT_EXPIRY",
            "description": "Contracts by expiry",
            "tool": {"name": "get_expiring_contracts", "version": "v1"},
        },
    ).json()

    # A realistic number of examples per intent. A four-example corpus is not
    # enough for the UNKNOWN rule to behave the way it does in practice.
    search_examples = [
        "Find all contracts with Microsoft",
        "Show me Microsoft agreements",
        "Give me all active contracts",
        "Search contracts for Acme Corp",
        "Which contracts do we have with Salesforce?",
        "List agreements signed with Oracle",
        "Look up vendor contracts in the EU region",
        "Show master service agreements for Deloitte",
    ]
    expiry_examples = [
        "Which contracts expire this year?",
        "Show agreements expiring next month",
        "What contracts are up for renewal?",
        "When does the Microsoft contract expire?",
        "List contracts ending before December",
        "Find agreements that terminate in 2026",
        "Contracts expiring in the next 90 days",
        "What agreements are due for renewal in Q3?",
    ]
    client.post(
        f"/api/v1/domains/{domain['id']}/intents/{search['id']}/examples/bulk",
        json=[{"text": text} for text in search_examples],
    )
    client.post(
        f"/api/v1/domains/{domain['id']}/intents/{expiry['id']}/examples/bulk",
        json=[{"text": text} for text in expiry_examples],
    )

    return {"domain": domain, "search": search, "expiry": expiry}


@pytest.fixture
def contract_domain(client: TestClient) -> dict:
    return seed_contract_domain(client)
