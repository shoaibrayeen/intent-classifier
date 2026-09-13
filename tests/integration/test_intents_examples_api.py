def test_intent_carries_tool_and_entity_schema(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]

    intent = client.get(f"/api/v1/domains/{domain_id}/intents/{intent_id}").json()
    assert intent["tool"]["name"] == "search_contracts"
    assert intent["entity_schema"] == {
        "counterparty": {"type": "string", "required": True},
        "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
    }
    assert intent["example_count"] == 8


def test_duplicate_intent_name_in_one_domain_is_rejected(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    duplicate = client.post(
        f"/api/v1/domains/{domain_id}/intents", json={"name": "CONTRACT_SEARCH"}
    )
    assert duplicate.status_code == 409


def test_same_intent_name_is_allowed_in_a_different_domain(client, contract_domain):
    other = client.post("/api/v1/domains", json={"name": "legal"}).json()
    created = client.post(
        f"/api/v1/domains/{other['id']}/intents", json={"name": "CONTRACT_SEARCH"}
    )
    assert created.status_code == 201


def test_intent_from_another_domain_is_not_visible(client, contract_domain):
    other = client.post("/api/v1/domains", json={"name": "legal"}).json()
    intent_id = contract_domain["search"]["id"]
    response = client.get(f"/api/v1/domains/{other['id']}/intents/{intent_id}")
    assert response.status_code == 404


def test_duplicate_example_text_is_rejected(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]
    base = f"/api/v1/domains/{domain_id}/intents/{intent_id}/examples"

    # differs only by case and spacing, so it normalizes to an existing example
    duplicate = client.post(base, json={"text": "find all CONTRACTS   with Microsoft"})
    assert duplicate.status_code == 409


def test_adding_an_example_rebuilds_the_index(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["expiry"]["id"]

    before = client.get(f"/api/v1/domains/{domain_id}/index/status").json()
    client.post(
        f"/api/v1/domains/{domain_id}/intents/{intent_id}/examples",
        json={"text": "list contracts terminating in December"},
    )
    after = client.get(f"/api/v1/domains/{domain_id}/index/status").json()

    assert after["version"] > before["version"]
    assert after["bm25_doc_count"] == before["bm25_doc_count"] + 1
    assert after["state"] == "ready"


def test_deleting_an_example_removes_it_from_both_indexes(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]
    base = f"/api/v1/domains/{domain_id}/intents/{intent_id}/examples"

    example_id = client.get(base).json()[0]["id"]
    assert client.delete(f"{base}/{example_id}").status_code == 204

    status = client.get(f"/api/v1/domains/{domain_id}/index/status").json()
    assert status["bm25_doc_count"] == 15
    assert status["dense_count"] == 15


def test_deleting_an_intent_removes_its_examples(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]

    assert client.delete(f"/api/v1/domains/{domain_id}/intents/{intent_id}").status_code == 204
    status = client.get(f"/api/v1/domains/{domain_id}/index/status").json()
    assert status["dense_count"] == 8


def test_renaming_an_intent_updates_its_examples(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]

    client.put(f"/api/v1/domains/{domain_id}/intents/{intent_id}", json={"name": "CONTRACT_LOOKUP"})
    result = client.post(
        "/api/v1/classify/debug",
        json={"domain": "contract", "text": "Find all contracts with Microsoft"},
    ).json()

    assert result["intent"] == "CONTRACT_LOOKUP"
    assert all(hit["intent_name"] != "CONTRACT_SEARCH" for hit in result["debug"]["dense_hits"])


def test_reindex_endpoint_reports_a_ready_index(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    status = client.post(f"/api/v1/domains/{domain_id}/reindex").json()
    assert status["state"] == "ready"
    assert status["bm25_doc_count"] == 16
    assert status["tokenizer_version"] == "simple_v1"


def test_updating_the_tool_mapping_keeps_a_valid_model(client, contract_domain):
    """Regression: model_copy does not re-validate, so a nested tool supplied
    as a dict used to be stored unvalidated and broke every later read."""
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]

    updated = client.put(
        f"/api/v1/domains/{domain_id}/intents/{intent_id}",
        json={"tool": {"name": "new_tool", "version": "v3"}},
    )
    assert updated.status_code == 200
    assert updated.json()["tool"] == {
        "name": "new_tool",
        "version": "v3",
        "mcp_tool_id": None,
    }

    # The record must still be readable, and the UI must still render it.
    reread = client.get(f"/api/v1/domains/{domain_id}/intents/{intent_id}").json()
    assert reread["tool"]["name"] == "new_tool"
    assert client.get(f"/ui/domains/{domain_id}/intents/{intent_id}").status_code == 200
    assert client.get(f"/ui/domains/{domain_id}").status_code == 200

    # And classification must still route to it.
    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()
    assert body["tool"]["name"] == "new_tool"
