"""Auto mode for intents, end to end: API, persistence, UI, and both modes together."""

BRIEF = "vendor agreements, renewals and obligations"


def mock_client(app_factory, **overrides):
    _, client = app_factory({"llm_provider": "mock", **overrides})
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()
    return client, domain


def test_generated_intents_are_saved_with_their_examples(app_factory):
    client, domain = mock_client(app_factory)

    body = client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate",
        json={"count": 4, "examples_per_intent": 6},
    ).json()

    assert body["status"] == "ok"
    assert body["created_count"] == 4
    assert body["example_count"] == 24
    assert all(i["created"] and i["intent_id"] for i in body["intents"])

    stored = client.get(f"/api/v1/domains/{domain['id']}/intents").json()
    assert len(stored) == 4
    assert all(i["example_count"] == 6 for i in stored)
    assert all(i["tool"]["name"] for i in stored)


def test_a_generated_catalogue_classifies_immediately(app_factory):
    """The point of generating examples too: the domain is usable at once."""
    client, domain = mock_client(app_factory)
    client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate",
        json={"count": 4, "examples_per_intent": 8},
    )

    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "find all contracts for Microsoft"}
    ).json()
    assert body["intent"] == "CONTRACT_SEARCH"
    assert body["tool"]["name"] == "search_contracts"

    off_domain = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "what is the weather today"}
    ).json()
    assert off_domain["intent"] == "UNKNOWN"


def test_the_index_is_rebuilt_so_health_is_in_step(app_factory):
    client, domain = mock_client(app_factory)
    client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate",
        json={"count": 3, "examples_per_intent": 5},
    )

    health = client.get("/api/v1/health/index").json()
    row = next(d for d in health["domains"] if d["domain_id"] == domain["id"])
    assert row["in_sync"] is True
    assert row["bm25_doc_count"] == 15


def test_dry_run_writes_nothing(app_factory):
    client, domain = mock_client(app_factory)

    body = client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate",
        json={"count": 3, "dry_run": True},
    ).json()

    assert body["status"] == "ok"
    assert body["dry_run"] is True
    assert body["created_count"] == 0
    assert len(body["intents"]) == 3
    assert all(not i["created"] for i in body["intents"])
    assert client.get(f"/api/v1/domains/{domain['id']}/intents").json() == []


def test_existing_intents_are_never_overwritten(app_factory):
    client, domain = mock_client(app_factory)
    manual = client.post(
        f"/api/v1/domains/{domain['id']}/intents",
        json={"name": "CONTRACT_SEARCH", "description": "mine", "tool": {"name": "my_tool"}},
    ).json()

    body = client.post(f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 3}).json()

    assert "CONTRACT_SEARCH" not in [i["name"] for i in body["intents"]]
    untouched = client.get(f"/api/v1/domains/{domain['id']}/intents/{manual['id']}").json()
    assert untouched["description"] == "mine"
    assert untouched["tool"]["name"] == "my_tool"


def test_generating_twice_adds_new_intents_rather_than_duplicates(app_factory):
    client, domain = mock_client(app_factory)
    first = client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 3}
    ).json()
    second = client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 3}
    ).json()

    names_first = {i["name"] for i in first["intents"]}
    names_second = {i["name"] for i in second["intents"]}
    assert not (names_first & names_second)

    stored = client.get(f"/api/v1/domains/{domain['id']}/intents").json()
    assert len({i["name"] for i in stored}) == len(stored) == 6


def test_generated_intents_remain_editable(app_factory):
    """Auto mode is a drafting aid, not a separate kind of object."""
    client, domain = mock_client(app_factory)
    body = client.post(f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 1}).json()
    intent_id = body["intents"][0]["intent_id"]

    edited = client.put(
        f"/api/v1/domains/{domain['id']}/intents/{intent_id}",
        json={"description": "refined by hand", "tool": {"name": "my_tool", "version": "v2"}},
    ).json()
    assert edited["description"] == "refined by hand"
    assert edited["tool"] == {"name": "my_tool", "version": "v2", "mcp_tool_id": None}

    assert client.delete(f"/api/v1/domains/{domain['id']}/intents/{intent_id}").status_code == 204


def test_zero_examples_per_intent_is_allowed(app_factory):
    client, domain = mock_client(app_factory)
    body = client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate",
        json={"count": 2, "examples_per_intent": 0},
    ).json()

    assert body["created_count"] == 2
    assert body["example_count"] == 0


def test_generation_without_a_provider_is_503(app_factory):
    _, client = app_factory({"llm_provider": "none"})
    domain = client.post("/api/v1/domains", json={"name": "contract"}).json()

    response = client.post(f"/api/v1/domains/{domain['id']}/intents/generate")
    assert response.status_code == 503
    assert "manually" in response.json()["error"]["message"]


def test_generation_on_a_missing_domain_is_404(app_factory):
    client, _ = mock_client(app_factory)
    assert client.post("/api/v1/domains/ghost/intents/generate").status_code == 404


def test_counts_outside_the_allowed_range_are_rejected(app_factory):
    client, domain = mock_client(app_factory)
    for payload in [{"count": 0}, {"count": 99}, {"examples_per_intent": -1}]:
        response = client.post(f"/api/v1/domains/{domain['id']}/intents/generate", json=payload)
        assert response.status_code == 422, payload


def test_write_scope_is_required(app_factory):
    _, client = app_factory(
        {"llm_provider": "mock", "auth_enabled": True, "api_keys": "admin:*:admin, reader:*:read"}
    )
    domain = client.post(
        "/api/v1/domains", json={"name": "contract"}, headers={"X-API-Key": "admin"}
    ).json()
    denied = client.post(
        f"/api/v1/domains/{domain['id']}/intents/generate", headers={"X-API-Key": "reader"}
    )
    assert denied.status_code == 403


# ------------------------------------------------------- example generation
def test_examples_can_be_generated_for_a_hand_written_intent(app_factory):
    """The manual path stays first class: write the intent, generate the examples."""
    client, domain = mock_client(app_factory)
    intent = client.post(
        f"/api/v1/domains/{domain['id']}/intents",
        json={"name": "CONTRACT_SEARCH", "tool": {"name": "search_contracts"}},
    ).json()

    body = client.post(
        f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples/generate",
        json={"count": 6},
    ).json()

    assert body["status"] == "ok"
    assert body["created_count"] == 6
    stored = client.get(f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples").json()
    assert len(stored) == 6

    classified = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "find all contracts"}
    ).json()
    assert classified["intent"] == "CONTRACT_SEARCH"


def test_generating_examples_twice_does_not_duplicate(app_factory):
    client, domain = mock_client(app_factory)
    intent = client.post(
        f"/api/v1/domains/{domain['id']}/intents", json={"name": "CONTRACT_SEARCH"}
    ).json()
    url = f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples/generate"

    client.post(url, json={"count": 6})
    client.post(url, json={"count": 6})

    stored = client.get(f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples").json()
    texts = [e["text"].casefold() for e in stored]
    assert len(texts) == len(set(texts))


def test_example_dry_run_writes_nothing(app_factory):
    client, domain = mock_client(app_factory)
    intent = client.post(
        f"/api/v1/domains/{domain['id']}/intents", json={"name": "CONTRACT_SEARCH"}
    ).json()

    body = client.post(
        f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples/generate",
        json={"count": 4, "dry_run": True},
    ).json()

    assert len(body["examples"]) == 4
    assert body["created_count"] == 0
    assert (
        client.get(f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples").json() == []
    )


def test_example_generation_on_a_missing_intent_is_404(app_factory):
    client, domain = mock_client(app_factory)
    assert (
        client.post(f"/api/v1/domains/{domain['id']}/intents/ghost/examples/generate").status_code
        == 404
    )


# --------------------------------------------------------------------- UI flows
def test_ui_domain_page_offers_both_modes(app_factory):
    client, domain = mock_client(app_factory)
    page = client.get(f"/ui/domains/{domain['id']}").text
    assert "Generate intents (mock)" in page
    assert "Create intent" in page  # the manual form is still there


def test_ui_generate_intents_saves_and_rerenders_the_list(app_factory):
    client, domain = mock_client(app_factory)

    response = client.post(
        f"/ui/domains/{domain['id']}/intents/generate",
        data={"count": "3", "examples_per_intent": "5", "brief": ""},
        headers={"HX-Request": "true"},
    )

    assert response.status_code == 200
    assert "Generated 3 intent(s) with 15 example(s)" in response.text
    assert "CONTRACT_SEARCH" in response.text
    assert len(client.get(f"/api/v1/domains/{domain['id']}/intents").json()) == 3


def test_ui_generate_examples_saves_and_rerenders(app_factory):
    client, domain = mock_client(app_factory)
    intent = client.post(
        f"/api/v1/domains/{domain['id']}/intents", json={"name": "CONTRACT_SEARCH"}
    ).json()

    response = client.post(
        f"/ui/domains/{domain['id']}/intents/{intent['id']}/examples/generate",
        data={"count": "4"},
        headers={"HX-Request": "true"},
    )

    assert "Added 4 example(s)" in response.text
    assert (
        len(client.get(f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples").json())
        == 4
    )


def test_ui_reports_a_missing_provider_rather_than_failing(app_factory):
    _, client = app_factory({"llm_provider": "none"})
    domain = client.post("/api/v1/domains", json={"name": "contract"}).json()

    response = client.post(
        f"/ui/domains/{domain['id']}/intents/generate",
        data={"count": "3"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "flash error" in response.text
    assert "No LLM provider configured" in response.text


def test_ui_intent_page_offers_example_generation(app_factory):
    client, domain = mock_client(app_factory)
    intent = client.post(
        f"/api/v1/domains/{domain['id']}/intents", json={"name": "CONTRACT_SEARCH"}
    ).json()
    page = client.get(f"/ui/domains/{domain['id']}/intents/{intent['id']}").text
    assert "Generate with LLM (mock)" in page
    assert "Add example" in page  # manual entry remains
