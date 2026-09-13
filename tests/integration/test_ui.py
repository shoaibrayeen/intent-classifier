"""The HTML UI. Mutations go through the same services as the JSON API."""


def test_dashboard_lists_domains(client, contract_domain):
    response = client.get("/")
    assert response.status_code == 200
    assert "contract" in response.text
    assert "Create domain" in response.text


def test_empty_dashboard_invites_the_first_domain(client):
    assert "No domains yet" in client.get("/").text


def test_domain_page_shows_intents_and_index_status(client, contract_domain):
    response = client.get(f"/ui/domains/{contract_domain['domain']['id']}")
    assert response.status_code == 200
    assert "CONTRACT_SEARCH" in response.text
    assert "index: ready" in response.text


def test_intent_page_shows_examples_tool_and_schema(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["search"]["id"]
    response = client.get(f"/ui/domains/{domain_id}/intents/{intent_id}")
    assert response.status_code == 200
    assert "Find all contracts with Microsoft" in response.text
    assert "search_contracts" in response.text
    assert "counterparty" in response.text


def test_missing_page_renders_an_error_page_not_a_fragment(client):
    response = client.get("/ui/domains/ghost")
    assert response.status_code == 404
    assert "<html" in response.text


def test_create_domain_form_returns_the_updated_list(client):
    response = client.post(
        "/ui/domains",
        data={"name": "finance", "description": "Invoices"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "finance" in response.text
    assert "hx-swap-oob" in response.text  # flash rides along out of band


def test_duplicate_domain_reports_an_error_flash_without_a_4xx(client):
    client.post("/ui/domains", data={"name": "finance"}, headers={"HX-Request": "true"})
    response = client.post("/ui/domains", data={"name": "finance"}, headers={"HX-Request": "true"})
    # htmx does not swap 4xx bodies, so failures come back as a 200 + flash
    assert response.status_code == 200
    assert "flash error" in response.text
    assert "already exists" in response.text


def test_invalid_entity_schema_json_is_reported_to_the_user(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    response = client.post(
        f"/ui/domains/{domain_id}/intents",
        data={"name": "BROKEN", "entity_schema": "{not json"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "flash error" in response.text
    assert "not valid JSON" in response.text
    assert client.get(f"/api/v1/domains/{domain_id}/intents").json().__len__() == 2


def test_adding_an_example_through_the_form_updates_the_list(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    intent_id = contract_domain["expiry"]["id"]
    response = client.post(
        f"/ui/domains/{domain_id}/intents/{intent_id}/examples",
        data={"text": "anything ending this quarter"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "anything ending this quarter" in response.text


def test_delete_domain_through_the_form(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    response = client.request("DELETE", f"/ui/domains/{domain_id}", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "No domains yet" in response.text
    assert client.get("/api/v1/domains").json() == []


def test_reindex_button_returns_a_fresh_status_badge(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    response = client.post(f"/ui/domains/{domain_id}/reindex", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "index: ready" in response.text
    assert "16 BM25 docs" in response.text


def test_playground_renders_the_full_retrieval_trace(client, contract_domain):
    response = client.post(
        "/ui/playground/classify",
        data={"domain": "contract", "text": "Find all contracts with Microsoft"},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    body = response.text
    # The reply lands in the chat thread...
    assert "CONTRACT_SEARCH" in body
    assert 'class="bubble user"' in body and 'class="bubble assistant' in body
    # ...and the reasoning is swapped into the detail pane beside it.
    assert 'id="detail-panel"' in body and 'hx-swap-oob="true"' in body
    assert "Confidence breakdown" in body
    assert "Retrieval" in body and ">Dense<" in body and ">BM25<" in body


def test_playground_shows_unknown_with_its_reason(client, contract_domain):
    response = client.post(
        "/ui/playground/classify",
        data={"domain": "contract", "text": "what is the weather today"},
        headers={"HX-Request": "true"},
    )
    assert "UNKNOWN" in response.text
    assert "reason:" in response.text


def test_static_assets_are_served_locally(client):
    assert client.get("/static/app.css").status_code == 200
    htmx = client.get("/static/vendor/htmx.min.js")
    assert htmx.status_code == 200
    assert "htmx" in htmx.text[:200]


def test_the_api_reference_is_served_in_the_ui(client):
    response = client.get("/ui/api")
    assert response.status_code == 200
    assert "Intent Classifier API" in response.text
    # Generated from the schema, so real endpoints must appear.
    assert "/api/v1/classify" in response.text
    assert "ClassifyRequest" in response.text


def test_the_interactive_docs_are_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200

    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Intent Classifier"
    assert "/api/v1/classify" in schema["paths"]
    assert {t["name"] for t in schema["tags"]} >= {"classify", "sessions", "domains"}


def test_every_page_links_to_the_documentation(client):
    for path in ["/", "/ui/playground", "/ui/operations"]:
        body = client.get(path).text
        assert 'href="/ui/api"' in body, path
        assert 'href="/docs"' in body, path
