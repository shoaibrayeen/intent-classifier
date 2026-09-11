"""Phase 3 and 4 end to end: classification, extraction, tool arguments."""

from app.services.llm.client import LLMUnavailable, ScriptedLLMClient

MICROSOFT = "Find all active contracts with Microsoft"


def build(app_factory, llm_client=None, **overrides):
    from tests.conftest import seed_contract_domain

    settings = {"entity_extraction_enabled": True, "openai_api_key": "test-key", **overrides}
    app, client = app_factory(settings, llm_client=llm_client)
    fixture = seed_contract_domain(client)
    return client, fixture


def test_entities_flow_into_the_tool_arguments(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft", "status": "ACTIVE"})
    client, _ = build(app_factory, llm)

    body = client.post("/api/v1/classify", json={"domain": "contract", "text": MICROSOFT}).json()

    assert body["intent"] == "CONTRACT_SEARCH"
    assert body["entities"] == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert body["tool"]["name"] == "search_contracts"
    assert body["tool"]["arguments"] == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert body["tool"]["ready"] is True
    assert body["entity_extraction"]["status"] == "ok"


def test_a_missing_required_argument_marks_the_call_not_ready(app_factory):
    llm = ScriptedLLMClient(default={"status": "ACTIVE"})
    client, _ = build(app_factory, llm)

    body = client.post("/api/v1/classify", json={"domain": "contract", "text": MICROSOFT}).json()

    assert body["tool"]["missing_required"] == ["counterparty"]
    assert body["tool"]["ready"] is False


def test_invented_entities_never_reach_the_tool(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft", "drop_table": "users"})
    client, _ = build(app_factory, llm)

    body = client.post("/api/v1/classify", json={"domain": "contract", "text": MICROSOFT}).json()

    assert body["tool"]["arguments"] == {"counterparty": "Microsoft"}
    assert any("drop_table" in note for note in body["entity_extraction"]["rejected"])


def test_extraction_is_skipped_for_an_unknown_intent(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft"})
    client, _ = build(app_factory, llm)

    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "what is the weather today"}
    ).json()

    assert body["intent"] == "UNKNOWN"
    assert body["entities"] == {}
    assert body["tool"] is None
    # The provider was never called for a request the engine did not understand.
    assert llm.calls == []


def test_a_provider_outage_does_not_break_classification(app_factory):
    llm = ScriptedLLMClient(raises=LLMUnavailable("provider is down"))
    client, _ = build(app_factory, llm)

    response = client.post("/api/v1/classify", json={"domain": "contract", "text": MICROSOFT})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "CONTRACT_SEARCH"
    assert body["entities"] == {}
    assert body["entity_extraction"]["status"] == "failed"
    assert body["tool"]["name"] == "search_contracts"


def test_a_request_can_turn_extraction_off(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft"})
    client, _ = build(app_factory, llm)

    body = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": MICROSOFT, "extract_entities": False},
    ).json()

    assert body["entities"] == {}
    assert llm.calls == []


def test_a_request_can_turn_extraction_on_when_the_default_is_off(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft"})
    client, _ = build(app_factory, llm, entity_extraction_enabled=False)

    body = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": MICROSOFT, "extract_entities": True},
    ).json()

    assert body["entities"] == {"counterparty": "Microsoft"}


def test_the_global_default_off_means_no_provider_call(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft"})
    client, _ = build(app_factory, llm, entity_extraction_enabled=False)

    body = client.post("/api/v1/classify", json={"domain": "contract", "text": MICROSOFT}).json()

    assert body["entities"] == {}
    assert body["entity_extraction"]["status"] == "disabled"
    assert llm.calls == []


def test_without_a_provider_classification_still_works(app_factory):
    client, _ = build(app_factory, None, entity_extraction_enabled=True, openai_api_key="")

    body = client.post("/api/v1/classify", json={"domain": "contract", "text": MICROSOFT}).json()

    assert body["intent"] == "CONTRACT_SEARCH"
    assert body["entities"] == {}
    assert body["entity_extraction"]["status"] == "unavailable"


def test_extraction_timing_is_reported_separately(app_factory):
    llm = ScriptedLLMClient(default={"counterparty": "Microsoft"})
    client, _ = build(app_factory, llm)

    body = client.post(
        "/api/v1/classify/debug", json={"domain": "contract", "text": MICROSOFT}
    ).json()

    timings = body["debug"]["timings"]
    assert timings["total_ms"] > 0
    assert timings["embed_ms"] > 0
    assert timings["extraction_ms"] >= 0
