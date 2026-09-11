"""Multi-turn sessions and domain-scoped instructions, end to end with the mock provider."""

from tests.conftest import seed_contract_domain


def mock_app(app_factory, **overrides):
    app, client = app_factory(
        {"llm_provider": "mock", "entity_extraction_enabled": True, **overrides}
    )
    fixture = seed_contract_domain(client)
    client.put(
        f"/api/v1/domains/{fixture['domain']['id']}",
        json={
            "system_instructions": "A counterparty is the other party on the contract.",
            "user_instructions": "Users name vendors by brand.",
        },
    )
    return app, client, fixture


def classify(client, text, session="sess-1", **extra):
    return client.post(
        "/api/v1/classify/debug",
        json={"domain": "contract", "text": text, "session_id": session, **extra},
    ).json()


def test_the_mock_provider_runs_the_whole_flow_without_a_key(app_factory):
    _, client, _ = mock_app(app_factory)
    body = classify(client, "Find all active contracts with Microsoft", session=None)

    assert body["intent"] == "CONTRACT_SEARCH"
    assert body["entities"] == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert body["entity_extraction"]["status"] == "ok"
    assert body["entity_extraction"]["model"] == "mock"
    assert body["tool"]["arguments"] == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert body["tool"]["ready"] is True
    assert body["context"] is None


def test_a_follow_up_is_read_against_the_previous_turn(app_factory):
    _, client, _ = mock_app(app_factory)

    first = classify(client, "Show me all contracts with Microsoft")
    assert first["intent"] == "CONTRACT_SEARCH"
    assert first["context"]["turn"] == 1
    assert first["context"]["previous_turns"] == 0

    # "and for Oracle" has nothing to retrieve on. Alone it is UNKNOWN; read
    # against the previous question it is the same search for a new vendor.
    alone = classify(client, "and for Oracle", session=None)
    assert alone["intent"] == "UNKNOWN"

    second = classify(client, "and for Oracle")
    assert second["intent"] == "CONTRACT_SEARCH"
    assert second["context"]["turn"] == 2
    assert second["context"]["previous_intent"] == "CONTRACT_SEARCH"
    assert second["context"]["used_for_retrieval"] is True
    assert second["context"]["retrieval_text"] == (
        "Show me all contracts with Microsoft and for Oracle"
    )
    # The value named in this turn wins over the one carried from the last.
    assert second["entities"]["counterparty"] == "Oracle"
    assert second["context"]["carried_entities"] == []


def test_entities_carry_forward_into_an_intent_that_accepts_them(app_factory):
    app, client, fixture = mock_app(app_factory)
    # CONTRACT_EXPIRY needs to accept counterparty for carry-over to apply.
    client.put(
        f"/api/v1/domains/{fixture['domain']['id']}/intents/{fixture['expiry']['id']}",
        json={"entity_schema": {"counterparty": {"type": "string"}, "period": {"type": "string"}}},
    )

    classify(client, "Show me all contracts with Microsoft")
    second = classify(client, "which of them expire next month")

    assert second["intent"] == "CONTRACT_EXPIRY"
    assert second["entities"]["counterparty"] == "Microsoft"
    assert second["context"]["carried_entities"] == ["counterparty"]
    assert second["tool"]["arguments"]["counterparty"] == "Microsoft"


def test_a_new_value_in_the_turn_beats_the_carried_one(app_factory):
    _, client, _ = mock_app(app_factory)
    classify(client, "Show me all contracts with Microsoft")
    second = classify(client, "Now show me Oracle agreements")

    assert second["entities"]["counterparty"] == "Oracle"
    assert second["context"]["carried_entities"] == []


def test_a_follow_up_that_stands_alone_does_not_need_context(app_factory):
    """Retrieval is good enough that many follow-ups resolve by themselves;
    context is a rescue, not a default path."""
    _, client, _ = mock_app(app_factory)
    classify(client, "Show me all contracts with Microsoft")
    second = classify(client, "when do they expire")

    assert second["intent"] == "CONTRACT_EXPIRY"
    assert second["context"]["used_for_retrieval"] is False
    assert second["context"]["previous_turns"] == 1


def test_context_never_overrules_a_confident_answer(app_factory):
    _, client, _ = mock_app(app_factory)
    classify(client, "Which contracts expire this year?")
    second = classify(client, "Find all contracts with Microsoft")

    assert second["intent"] == "CONTRACT_SEARCH"
    assert second["context"]["used_for_retrieval"] is False


def test_sessions_are_isolated_from_each_other(app_factory):
    _, client, _ = mock_app(app_factory)
    classify(client, "Show me all contracts with Microsoft", session="alice")
    other = classify(client, "and for Oracle", session="bob")

    assert other["intent"] == "UNKNOWN"
    assert other["context"]["previous_turns"] == 0


def test_use_context_false_treats_the_turn_as_standalone(app_factory):
    _, client, _ = mock_app(app_factory)
    classify(client, "Show me all contracts with Microsoft")
    second = classify(client, "and for Oracle", use_context=False)

    assert second["intent"] == "UNKNOWN"
    assert second["context"]["turn"] == 2  # still recorded in the session


def test_the_session_can_be_read_and_cleared(app_factory):
    _, client, _ = mock_app(app_factory)
    classify(client, "Show me all contracts with Microsoft")
    classify(client, "when do they expire")

    view = client.get("/api/v1/sessions/sess-1").json()
    assert [t["turn"] for t in view["turns"]] == [1, 2]
    assert view["last_intent"] == "CONTRACT_EXPIRY"
    assert view["entities_in_play"]["counterparty"] == "Microsoft"

    assert client.delete("/api/v1/sessions/sess-1").status_code == 204
    assert client.get("/api/v1/sessions/sess-1").status_code == 404
    # With the history gone, the follow-up has nothing to lean on again.
    assert classify(client, "and for Oracle")["intent"] == "UNKNOWN"


def test_domain_instructions_and_history_reach_the_provider(app_factory):
    app, client, fixture = mock_app(app_factory)
    client.put(
        f"/api/v1/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}",
        json={"extraction_hints": "'live' means ACTIVE."},
    )
    mock = app.state.container.entity_extractor._client

    classify(client, "Show me all contracts with Microsoft")
    classify(client, "Find Oracle agreements")

    system, user = mock.calls[-1]
    assert "A counterparty is the other party on the contract." in system
    assert "'live' means ACTIVE." in system
    assert "Users name vendors by brand." in user
    assert "conversation_history" in user and "Microsoft" in user


def test_turns_are_stored_in_chroma_alongside_everything_else(app_factory):
    app, client, _ = mock_app(app_factory)
    classify(client, "Show me all contracts with Microsoft")

    store = app.state.container.store
    names = {c.name for c in store.client.list_collections()}
    assert "session_turns" in names
    assert store.sessions.count() == 1


def test_sessions_can_be_disabled(app_factory):
    _, client, _ = mock_app(app_factory, sessions_enabled=False)
    body = classify(client, "Show me all contracts with Microsoft")
    assert body["context"] is None


def test_invalid_session_ids_are_rejected(app_factory):
    _, client, _ = mock_app(app_factory)
    response = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "x", "session_id": "has spaces"}
    )
    assert response.status_code == 422


def test_domain_instructions_survive_a_round_trip(app_factory):
    _, client, fixture = mock_app(app_factory)
    domain = client.get(f"/api/v1/domains/{fixture['domain']['id']}").json()
    assert domain["system_instructions"].startswith("A counterparty")
    assert domain["user_instructions"] == "Users name vendors by brand."


def test_health_names_the_provider(app_factory):
    _, client, _ = mock_app(app_factory)
    body = client.get("/api/v1/health").json()
    assert body["entity_extraction"]["provider"] == "mock"
    assert body["entity_extraction"]["provider_configured"] is True
    assert body["sessions_enabled"] is True
