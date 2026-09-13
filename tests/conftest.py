"""Shared fixtures. Tests always run against an in-memory Chroma store."""

from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

os.environ["CHROMA_MODE"] = "ephemeral"
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENTITY_EXTRACTION_ENABLED"] = "false"
os.environ["AUTH_ENABLED"] = "false"
os.environ["AUDIT_LOG_ENABLED"] = "false"
os.environ.setdefault("MODEL_CACHE_DIR", "./data/models")

from app.config import Settings  # noqa: E402
from app.db.chroma import ChromaStore  # noqa: E402
from app.main import create_app  # noqa: E402

_reset_store: ChromaStore | None = None


def reset_shared_store() -> None:
    """Clear the ephemeral store every app in this process shares.

    One store is reused rather than constructing a client per app: each extra
    client adds a reference to Chroma's shared native system, and more live
    references make the interpreter-shutdown race in its Rust bindings more
    likely to fire.
    """
    global _reset_store
    if _reset_store is None:
        _reset_store = ChromaStore(make_settings())
    _reset_store.reset_all()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Shut Chroma down, then leave without running interpreter finalization.

    chromadb 1.5.x on macOS ARM64 intermittently aborts with 'recursive_mutex
    lock failed' while its Rust runtime is destroyed during interpreter exit.
    Every test has already run and reported by this point, but an aborting
    process still returns 134 and fails CI. Stopping the cached systems first
    helps and is not sufficient on its own: measured over repeated runs it
    still aborted roughly one time in eight.

    So after stopping them we flush and call os._exit, which skips the
    finalization that triggers the crash. The trade-off is that atexit handlers
    do not run, so this is skipped when coverage is active (it writes its data
    at exit) and can be disabled with INTENT_CLASSIFIER_SOFT_EXIT=1.
    """
    global _reset_store
    try:
        import gc

        from chromadb.api.shared_system_client import SharedSystemClient

        # Drop our own references first, so stopping the systems below is not
        # racing against objects that still hold handles into the bindings.
        _reset_store = None
        gc.collect()

        for system in list(SharedSystemClient._identifier_to_system.values()):
            try:
                system.stop()
            except Exception:
                pass
        SharedSystemClient.clear_system_cache()
        gc.collect()
    except Exception:
        pass

    if os.environ.get("INTENT_CLASSIFIER_SOFT_EXIT") or "coverage" in sys.modules:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exitstatus))


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
        settings = make_settings(**(settings_overrides or {}))
        # Ephemeral Chroma clients share one underlying system per settings
        # identifier, so records leak between apps built in the same process.
        # Clear it BEFORE the new app's lifespan runs: clearing afterwards
        # would also wipe anything startup did, such as seeding.
        reset_shared_store()
        app = create_app(settings, llm_client=llm_client)
        client = TestClient(app)
        client.__enter__()
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
