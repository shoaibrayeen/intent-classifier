import pytest


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Find all contracts with Microsoft", "CONTRACT_SEARCH"),
        ("Show me Microsoft agreements", "CONTRACT_SEARCH"),
        ("Which agreements are we running with Acme?", "CONTRACT_SEARCH"),
        ("Which contracts expire this year?", "CONTRACT_EXPIRY"),
        ("What agreements come up for renewal soon?", "CONTRACT_EXPIRY"),
    ],
)
def test_paraphrases_reach_the_right_intent(client, contract_domain, query, expected):
    response = client.post("/api/v1/classify", json={"domain": "contract", "text": query})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == expected
    assert body["confidence"] >= 0.55


def test_response_carries_the_configured_tool(client, contract_domain):
    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()
    tool = body["tool"]
    assert tool["name"] == "search_contracts"
    assert tool["version"] == "v1"
    # No extraction ran, so there are no arguments yet, and nothing is required.
    assert tool["arguments"] == {}
    assert tool["missing_required"] == []
    assert tool["ready"] is True


def test_domain_can_be_addressed_by_id_or_name(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    by_name = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Show me Microsoft agreements"}
    ).json()
    by_id = client.post(
        "/api/v1/classify", json={"domain": domain_id, "text": "Show me Microsoft agreements"}
    ).json()
    assert by_name["intent"] == by_id["intent"] == "CONTRACT_SEARCH"


@pytest.mark.parametrize(
    "query",
    [
        "What's the weather today?",
        "Book me a table for two at eight",
        "How do I reset my router?",
    ],
)
def test_out_of_domain_queries_return_unknown(client, contract_domain, query):
    body = client.post("/api/v1/classify", json={"domain": "contract", "text": query}).json()
    assert body["intent"] == "UNKNOWN"
    assert body["intent_id"] is None
    assert body["reason"] in {"low_confidence", "low_similarity"}


def test_domain_without_examples_returns_unknown_with_a_clear_reason(client):
    domain = client.post("/api/v1/domains", json={"name": "empty"}).json()
    client.post(f"/api/v1/domains/{domain['id']}/intents", json={"name": "SOMETHING"})

    body = client.post("/api/v1/classify", json={"domain": "empty", "text": "anything"}).json()
    assert body["intent"] == "UNKNOWN"
    assert body["reason"] == "no_examples"
    assert body["confidence"] == 0.0


def test_blank_query_is_rejected(client, contract_domain):
    response = client.post("/api/v1/classify", json={"domain": "contract", "text": "   "})
    assert response.status_code == 422


def test_unknown_domain_returns_404(client):
    response = client.post("/api/v1/classify", json={"domain": "ghost", "text": "hello"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_plain_classify_hides_the_retrieval_trace(client, contract_domain):
    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find Microsoft contracts"}
    ).json()
    assert "debug" not in body


def test_debug_endpoint_exposes_every_stage(client, contract_domain):
    body = client.post(
        "/api/v1/classify/debug",
        json={"domain": "contract", "text": "Find Microsoft contracts"},
    ).json()
    debug = body["debug"]

    assert debug["normalized_text"] == "find microsoft contracts"
    assert debug["tokens"] == ["find", "microsoft", "contracts"]
    assert debug["dense_hits"] and debug["bm25_hits"]
    assert debug["rrf_intents"][0]["intent"] == body["intent"]
    assert 0.0 <= debug["confidence_breakdown"]["confidence"] <= 1.0
    assert debug["index_version"] >= 1


def test_disabled_intent_is_not_returned(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    search_id = contract_domain["search"]["id"]
    client.put(f"/api/v1/domains/{domain_id}/intents/{search_id}", json={"status": "disabled"})

    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()
    assert body["intent"] != "CONTRACT_SEARCH"


def test_domains_are_isolated_from_each_other(client, contract_domain):
    employee = client.post("/api/v1/domains", json={"name": "employee"}).json()
    intent = client.post(
        f"/api/v1/domains/{employee['id']}/intents", json={"name": "EMPLOYEE_SEARCH"}
    ).json()
    client.post(
        f"/api/v1/domains/{employee['id']}/intents/{intent['id']}/examples/bulk",
        json=[
            {"text": "Find all engineers in Bangalore"},
            {"text": "Show me employees in finance"},
        ],
    )

    body = client.post(
        "/api/v1/classify/debug",
        json={"domain": "employee", "text": "Show me Microsoft agreements"},
    ).json()
    assert all(hit["intent_name"] == "EMPLOYEE_SEARCH" for hit in body["debug"]["dense_hits"])
