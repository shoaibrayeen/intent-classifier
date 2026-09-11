"""Domain instruction generation, end to end through API and UI."""

BRIEF = "tracking vendor agreements, renewals and obligations"


def mock_client(app_factory):
    _, client = app_factory({"llm_provider": "mock"})
    return client


def test_create_with_generation_saves_drafted_instructions(app_factory):
    client = mock_client(app_factory)
    created = client.post(
        "/api/v1/domains",
        json={"name": "contract", "description": BRIEF, "generate_instructions": True},
    )
    assert created.status_code == 201
    body = created.json()
    assert BRIEF in body["system_instructions"]
    assert "never infer" in body["system_instructions"].lower()
    assert body["user_instructions"]

    # persisted, not just returned
    stored = client.get(f"/api/v1/domains/{body['id']}").json()
    assert stored["system_instructions"] == body["system_instructions"]


def test_explicit_instructions_win_over_generation(app_factory):
    client = mock_client(app_factory)
    body = client.post(
        "/api/v1/domains",
        json={
            "name": "contract",
            "description": BRIEF,
            "generate_instructions": True,
            "system_instructions": "Mine, hand-written.",
        },
    ).json()
    assert body["system_instructions"] == "Mine, hand-written."


def test_create_without_the_flag_generates_nothing(app_factory):
    client = mock_client(app_factory)
    body = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()
    assert body["system_instructions"] == ""
    assert body["user_instructions"] == ""


def test_create_with_generation_and_no_provider_fails_before_creating(app_factory):
    _, client = app_factory({"llm_provider": "none"})
    response = client.post(
        "/api/v1/domains",
        json={"name": "contract", "description": BRIEF, "generate_instructions": True},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "unavailable"
    # nothing half-created
    assert client.get("/api/v1/domains").json() == []


def test_regenerate_endpoint_saves_and_reports(app_factory):
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()

    result = client.post(f"/api/v1/domains/{domain['id']}/instructions/generate").json()
    assert result["status"] == "ok"
    assert result["saved"] is True
    assert result["provider"] == "mock"
    assert BRIEF in result["system_instructions"]

    stored = client.get(f"/api/v1/domains/{domain['id']}").json()
    assert stored["system_instructions"] == result["system_instructions"]


def test_a_brief_overrides_the_stored_description(app_factory):
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()

    result = client.post(
        f"/api/v1/domains/{domain['id']}/instructions/generate",
        json={"brief": "comparing procurement deals across regions"},
    ).json()
    assert "comparing procurement deals across regions" in result["system_instructions"]
    assert BRIEF not in result["system_instructions"]


def test_regenerate_needs_a_description_or_brief(app_factory):
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "bare"}).json()
    response = client.post(f"/api/v1/domains/{domain['id']}/instructions/generate")
    assert response.status_code == 400
    assert "brief" in response.json()["error"]["message"]


def test_regenerate_without_a_provider_is_503(app_factory):
    _, client = app_factory({"llm_provider": "none"})
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()
    response = client.post(f"/api/v1/domains/{domain['id']}/instructions/generate")
    assert response.status_code == 503


def test_regenerate_on_a_missing_domain_is_404(app_factory):
    client = mock_client(app_factory)
    assert client.post("/api/v1/domains/ghost/instructions/generate").status_code == 404


def test_generated_instructions_reach_the_extractor(app_factory):
    """The point of the feature: the draft must actually shape extraction."""
    from tests.conftest import seed_contract_domain

    app, client = app_factory({"llm_provider": "mock", "entity_extraction_enabled": True})
    fixture = seed_contract_domain(client)
    client.post(
        f"/api/v1/domains/{fixture['domain']['id']}/instructions/generate", json={"brief": BRIEF}
    )

    client.post("/api/v1/classify", json={"domain": "contract", "text": "Find Microsoft contracts"})
    mock = app.state.container.entity_extractor._client
    system, _ = mock.calls[-1]
    assert BRIEF in system  # the saved draft is now part of the extraction prompt


def test_manual_edit_after_generation_sticks(app_factory):
    """Generate, then refine by hand: the interface the user asked for."""
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()
    client.post(f"/api/v1/domains/{domain['id']}/instructions/generate")

    edited = client.put(
        f"/api/v1/domains/{domain['id']}",
        json={"system_instructions": "Refined by a human."},
    ).json()
    assert edited["system_instructions"] == "Refined by a human."
    # the untouched block survives the partial update
    assert edited["user_instructions"] != ""


def test_write_scope_is_required_to_generate(app_factory):
    _, client = app_factory(
        {
            "llm_provider": "mock",
            "auth_enabled": True,
            "api_keys": "admin-key:*:admin, reader:*:read",
        }
    )
    domain = client.post(
        "/api/v1/domains",
        json={"name": "contract", "description": BRIEF},
        headers={"X-API-Key": "admin-key"},
    ).json()
    denied = client.post(
        f"/api/v1/domains/{domain['id']}/instructions/generate",
        headers={"X-API-Key": "reader"},
    )
    assert denied.status_code == 403


# ------------------------------------------------------------------- UI flows
def test_ui_generate_button_saves_and_rerenders_the_editor(app_factory):
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()

    response = client.post(
        f"/ui/domains/{domain['id']}/instructions/generate",
        data={"brief": ""},
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    assert "Instructions drafted and saved" in response.text
    assert BRIEF in response.text  # the editor comes back filled

    stored = client.get(f"/api/v1/domains/{domain['id']}").json()
    assert BRIEF in stored["system_instructions"]


def test_ui_generate_with_no_brief_or_description_explains_itself(app_factory):
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "bare"}).json()
    response = client.post(
        f"/ui/domains/{domain['id']}/instructions/generate",
        data={"brief": "   "},
        headers={"HX-Request": "true"},
    )
    assert "flash error" in response.text
    assert "brief" in response.text


def test_ui_create_with_generation_checkbox(app_factory):
    client = mock_client(app_factory)
    response = client.post(
        "/ui/domains",
        data={
            "name": "finance",
            "description": "invoices and payments",
            "generate_instructions": "1",
        },
        headers={"HX-Request": "true"},
    )
    assert "drafted and saved" in response.text
    domain = client.get("/api/v1/domains").json()[0]
    assert "invoices and payments" in domain["system_instructions"]


def test_ui_domain_page_offers_the_generator(app_factory):
    client = mock_client(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()
    page = client.get(f"/ui/domains/{domain['id']}").text
    assert "Generate with LLM (mock)" in page
    assert "instructions-editor" in page


def test_ui_shows_generation_as_unavailable_without_a_provider(app_factory):
    _, client = app_factory({"llm_provider": "none"})
    domain = client.post("/api/v1/domains", json={"name": "contract", "description": BRIEF}).json()
    page = client.get(f"/ui/domains/{domain['id']}").text
    assert "Needs a provider" in page
