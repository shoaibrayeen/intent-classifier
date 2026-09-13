"""MCP end to end: registry, binding, and the reported call."""

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"counterparty": {"type": "string"}, "status": {"type": "string"}},
    "required": ["counterparty"],
}
TOOLS_LIST = {
    "tools": [
        {"name": "search_contracts", "description": "Search", "inputSchema": SEARCH_SCHEMA},
        {"name": "summarize_contract", "description": "Summarize", "inputSchema": {}},
    ]
}


def mock_app(app_factory, **overrides):
    app, client = app_factory(
        {"llm_provider": "mock", "entity_extraction_enabled": True, **overrides}
    )
    return app, client


def imported(client, server="contracts"):
    return client.post(
        "/api/v1/mcp/tools/import",
        json={
            "server": server,
            "transport": "stdio",
            "endpoint": "npx -y @acme/mcp",
            "tools": TOOLS_LIST,
        },
    ).json()


def test_a_server_catalogue_can_be_imported_and_listed(app_factory):
    _, client = mock_app(app_factory)
    result = imported(client)

    assert sorted(result["created"]) == ["search_contracts", "summarize_contract"]
    listed = client.get("/api/v1/mcp/tools").json()
    assert {t["qualified_name"] for t in listed} == {
        "mcp__contracts__search_contracts",
        "mcp__contracts__summarize_contract",
    }


def test_a_tool_can_be_registered_by_hand(app_factory):
    _, client = mock_app(app_factory)
    created = client.post(
        "/api/v1/mcp/tools",
        json={"server": "contracts", "name": "search_contracts", "input_schema": SEARCH_SCHEMA},
    )
    assert created.status_code == 201
    assert created.json()["qualified_name"] == "mcp__contracts__search_contracts"


def test_registering_the_same_tool_twice_conflicts(app_factory):
    _, client = mock_app(app_factory)
    body = {"server": "contracts", "name": "search_contracts"}
    client.post("/api/v1/mcp/tools", json=body)
    assert client.post("/api/v1/mcp/tools", json=body).status_code == 409


def test_classification_reports_the_bound_mcp_call(app_factory):
    from tests.conftest import seed_contract_domain

    _, client = mock_app(app_factory)
    imported(client)
    fixture = seed_contract_domain(client)
    tool_id = next(
        t["id"] for t in client.get("/api/v1/mcp/tools").json() if t["name"] == "search_contracts"
    )
    client.put(
        f"/api/v1/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}",
        json={"tool": {"name": "search_contracts", "mcp_tool_id": tool_id}},
    )

    body = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": "Find all active contracts with Microsoft"},
    ).json()

    mcp = body["tool"]["mcp"]
    assert mcp["qualified_name"] == "mcp__contracts__search_contracts"
    assert mcp["server"] == "contracts"
    assert mcp["transport"] == "stdio"
    assert mcp["endpoint"] == "npx -y @acme/mcp"
    assert mcp["arguments"]["counterparty"] == "Microsoft"
    assert mcp["ready"] is True


def test_an_unbound_intent_reports_no_mcp_call(app_factory):
    from tests.conftest import seed_contract_domain

    _, client = mock_app(app_factory)
    seed_contract_domain(client)
    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()
    assert body["tool"]["mcp"] is None


def test_deleting_a_tool_leaves_the_binding_visibly_unresolved(app_factory):
    from tests.conftest import seed_contract_domain

    _, client = mock_app(app_factory)
    imported(client)
    fixture = seed_contract_domain(client)
    tool_id = next(
        t["id"] for t in client.get("/api/v1/mcp/tools").json() if t["name"] == "search_contracts"
    )
    client.put(
        f"/api/v1/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}",
        json={"tool": {"name": "search_contracts", "mcp_tool_id": tool_id}},
    )
    client.delete(f"/api/v1/mcp/tools/{tool_id}")

    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()

    assert body["tool"]["mcp"]["unresolved"]
    assert body["tool"]["mcp"]["ready"] is False


def test_generated_intents_are_bound_to_registered_tools(app_factory):
    _, client = mock_app(app_factory)
    imported(client)
    domain = client.post(
        "/api/v1/domains", json={"name": "contract", "description": "vendor agreements"}
    ).json()

    body = client.post(f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 3}).json()

    bound = {i["name"]: i["mcp_tool"] for i in body["intents"]}
    assert bound["CONTRACT_SEARCH"] == "mcp__contracts__search_contracts"
    assert bound["CONTRACT_SUMMARY"] == "mcp__contracts__summarize_contract"
    # Nothing registered serves DETAILS, so it stays unbound rather than guessing.
    assert bound["CONTRACT_DETAILS"] is None


def test_generation_with_an_empty_registry_binds_nothing(app_factory):
    _, client = mock_app(app_factory)
    domain = client.post("/api/v1/domains", json={"name": "contract"}).json()
    body = client.post(f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 2}).json()
    assert all(i["mcp_tool"] is None for i in body["intents"])


def test_the_registry_counts_bound_intents(app_factory):
    _, client = mock_app(app_factory)
    imported(client)
    domain = client.post("/api/v1/domains", json={"name": "contract"}).json()
    client.post(f"/api/v1/domains/{domain['id']}/intents/generate", json={"count": 3})

    listed = client.get("/api/v1/mcp/tools").json()
    search = next(t for t in listed if t["name"] == "search_contracts")
    assert search["bound_intents"] == 1


def test_write_scope_is_required_to_register(app_factory):
    _, client = app_factory({"auth_enabled": True, "api_keys": "admin:*:admin, reader:*:read"})
    denied = client.post(
        "/api/v1/mcp/tools",
        json={"server": "s", "name": "t"},
        headers={"X-API-Key": "reader"},
    )
    assert denied.status_code == 403


def test_execution_is_refused_by_default(app_factory):
    """The seam exists, and the default implementation does not call anything."""
    import asyncio

    from app.models.mcp import McpCall

    app, _ = mock_app(app_factory)
    executor = app.state.container.mcp_executor
    assert executor.available is False
    result = asyncio.run(
        executor.call(McpCall(tool_id="x", server="s", tool="t", qualified_name="mcp__s__t"))
    )
    assert result.status == "refused"


# --------------------------------------------------------------------- UI
def test_the_registry_page_lists_tools(app_factory):
    _, client = mock_app(app_factory)
    imported(client)
    page = client.get("/ui/mcp").text
    assert "mcp__contracts__search_contracts" in page
    assert "Import from a server" in page


def test_the_tool_page_shows_its_arguments(app_factory):
    _, client = mock_app(app_factory)
    imported(client)
    tool_id = client.get("/api/v1/mcp/tools").json()[0]["id"]
    page = client.get(f"/ui/mcp/{tool_id}").text
    assert "counterparty" in page
    assert "Where it runs" in page


def test_importing_through_the_ui(app_factory):
    import json

    _, client = mock_app(app_factory)
    response = client.post(
        "/ui/mcp/import",
        data={"server": "contracts", "transport": "stdio", "tools": json.dumps(TOOLS_LIST)},
        headers={"HX-Request": "true"},
    )
    assert "Imported 2 new" in response.text
    assert len(client.get("/api/v1/mcp/tools").json()) == 2


def test_bad_json_in_the_import_form_is_explained(app_factory):
    _, client = mock_app(app_factory)
    response = client.post(
        "/ui/mcp/import",
        data={"server": "contracts", "tools": "{not json"},
        headers={"HX-Request": "true"},
    )
    assert "flash error" in response.text
    assert "valid JSON" in response.text


def test_binding_an_intent_through_the_ui(app_factory):
    from tests.conftest import seed_contract_domain

    _, client = mock_app(app_factory)
    imported(client)
    fixture = seed_contract_domain(client)
    tool_id = next(
        t["id"] for t in client.get("/api/v1/mcp/tools").json() if t["name"] == "search_contracts"
    )

    response = client.post(
        f"/ui/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}/mcp",
        data={"mcp_tool_id": tool_id},
        headers={"HX-Request": "true"},
    )
    assert "Bound CONTRACT_SEARCH to mcp__contracts__search_contracts" in response.text

    intent = client.get(
        f"/api/v1/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}"
    ).json()
    assert intent["tool"]["mcp_tool_id"] == tool_id


def test_a_binding_can_be_cleared(app_factory):
    from tests.conftest import seed_contract_domain

    _, client = mock_app(app_factory)
    imported(client)
    fixture = seed_contract_domain(client)
    url = f"/ui/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}/mcp"
    tool_id = client.get("/api/v1/mcp/tools").json()[0]["id"]
    client.post(url, data={"mcp_tool_id": tool_id}, headers={"HX-Request": "true"})

    response = client.post(url, data={"mcp_tool_id": ""}, headers={"HX-Request": "true"})

    assert "Cleared the MCP binding" in response.text
    intent = client.get(
        f"/api/v1/domains/{fixture['domain']['id']}/intents/{fixture['search']['id']}"
    ).json()
    assert intent["tool"]["mcp_tool_id"] is None
