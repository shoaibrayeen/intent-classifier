"""MCP registry: resolution, schema mapping, and import shapes."""

import pytest

from app.config import Settings
from app.db.chroma import ChromaStore
from app.errors import ConflictError, InvalidInputError
from app.models.intent import IntentConfig, ToolRef
from app.models.mcp import McpImportRequest, McpToolCreate, McpToolUpdate, qualified_name
from app.repositories.chroma_mcp import ChromaMcpToolRepository
from app.services.mcp_service import McpToolService, schema_summary

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"counterparty": {"type": "string"}, "status": {"type": "string"}},
    "required": ["counterparty"],
}


@pytest.fixture
def service():
    store = ChromaStore(Settings(chroma_mode="ephemeral"))
    store.reset_all()
    return McpToolService(store, ChromaMcpToolRepository(store))


def register(service, name="search_contracts", server="contracts", schema=None):
    return service.create(
        McpToolCreate(
            server=server,
            name=name,
            description="Search contracts",
            input_schema=schema if schema is not None else SEARCH_SCHEMA,
            transport="stdio",
            endpoint="npx -y @acme/contracts-mcp",
        )
    )


def test_qualified_name_uses_the_client_convention():
    assert qualified_name("contracts", "search") == "mcp__contracts__search"


def test_registering_twice_conflicts(service):
    register(service)
    with pytest.raises(ConflictError):
        register(service)


def test_the_same_tool_name_on_another_server_is_fine(service):
    register(service)
    assert register(service, server="legacy").server == "legacy"


@pytest.mark.parametrize(
    "reference",
    [
        "mcp__contracts__search_contracts",
        "contracts::search_contracts",
        "contracts/search_contracts",
    ],
)
def test_a_tool_resolves_by_any_reasonable_reference(service, reference):
    tool = register(service)
    assert service.resolve(reference).id == tool.id
    assert service.resolve(tool.id).id == tool.id


def test_an_unknown_reference_resolves_to_nothing(service):
    register(service)
    assert service.resolve("mcp__other__missing") is None
    assert service.resolve("") is None


def test_entities_are_mapped_onto_the_tools_arguments(service):
    tool = register(service)
    intent = IntentConfig(name="X", domain_id="d", tool=ToolRef(mcp_tool_id=tool.id))

    call = service.build_call(intent, {"counterparty": "Microsoft", "status": "ACTIVE"})

    assert call.qualified_name == "mcp__contracts__search_contracts"
    assert call.arguments == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert call.missing_required == []
    assert call.ready is True
    assert call.endpoint == "npx -y @acme/contracts-mcp"


def test_entities_the_tool_does_not_accept_are_reported_not_sent(service):
    tool = register(service)
    intent = IntentConfig(name="X", domain_id="d", tool=ToolRef(mcp_tool_id=tool.id))

    call = service.build_call(intent, {"counterparty": "Microsoft", "nonsense": "x"})

    assert call.arguments == {"counterparty": "Microsoft"}
    assert call.unmapped == ["nonsense"]


def test_a_missing_required_argument_is_not_ready(service):
    tool = register(service)
    intent = IntentConfig(name="X", domain_id="d", tool=ToolRef(mcp_tool_id=tool.id))

    call = service.build_call(intent, {"status": "ACTIVE"})

    assert call.missing_required == ["counterparty"]
    assert call.ready is False


def test_a_binding_to_a_deleted_tool_is_reported_as_unresolved(service):
    tool = register(service)
    service.delete(tool.id)
    intent = IntentConfig(name="X", domain_id="d", tool=ToolRef(mcp_tool_id=tool.id))

    call = service.build_call(intent, {})

    assert call.unresolved is not None
    assert call.ready is False


def test_a_disabled_tool_is_not_ready(service):
    tool = register(service)
    service.update(tool.id, McpToolUpdate(status="disabled"))
    intent = IntentConfig(name="X", domain_id="d", tool=ToolRef(mcp_tool_id=tool.id))

    call = service.build_call(intent, {"counterparty": "Microsoft"})

    assert call.ready is False
    assert "disabled" in call.unresolved


def test_an_unbound_intent_has_no_mcp_call(service):
    intent = IntentConfig(name="X", domain_id="d", tool=ToolRef(name="local_tool"))
    assert service.build_call(intent, {"a": 1}) is None


# ---------------------------------------------------------------------- import
TOOLS = [
    {"name": "search_contracts", "description": "Search", "inputSchema": SEARCH_SCHEMA},
    {"name": "summarize_contract", "description": "Summarize", "inputSchema": {}},
]


@pytest.mark.parametrize(
    "payload",
    [
        TOOLS,
        {"tools": TOOLS},
        {"result": {"tools": TOOLS}},
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": TOOLS}},
    ],
)
def test_import_accepts_every_shape_people_actually_have(service, payload):
    result = service.import_tools(McpImportRequest(server="contracts", tools=payload))
    assert sorted(result.created) == ["search_contracts", "summarize_contract"]


def test_reimport_updates_in_place(service):
    service.import_tools(McpImportRequest(server="contracts", tools=TOOLS))
    changed = [{"name": "search_contracts", "description": "New wording", "inputSchema": {}}]

    result = service.import_tools(McpImportRequest(server="contracts", tools=changed))

    assert result.updated == ["search_contracts"]
    assert result.created == []
    assert service.resolve("contracts::search_contracts").description == "New wording"


def test_replace_removes_tools_the_payload_omits(service):
    service.import_tools(McpImportRequest(server="contracts", tools=TOOLS))
    keep = [{"name": "search_contracts", "inputSchema": {}}]

    result = service.import_tools(McpImportRequest(server="contracts", tools=keep, replace=True))

    assert result.removed == ["summarize_contract"]
    assert [t.name for t in service.list("contracts")] == ["search_contracts"]


def test_replace_does_not_touch_another_server(service):
    service.import_tools(McpImportRequest(server="contracts", tools=TOOLS))
    service.import_tools(McpImportRequest(server="other", tools=TOOLS))

    service.import_tools(
        McpImportRequest(server="contracts", tools=[{"name": "search_contracts"}], replace=True)
    )

    assert len(service.list("other")) == 2


def test_a_payload_with_no_tools_is_rejected(service):
    with pytest.raises(InvalidInputError):
        service.import_tools(McpImportRequest(server="contracts", tools={"nothing": "here"}))


def test_nameless_tools_are_skipped_not_fatal(service):
    result = service.import_tools(
        McpImportRequest(server="contracts", tools=[{"description": "no name"}, TOOLS[0]])
    )
    assert result.created == ["search_contracts"]
    assert result.skipped


def test_schema_summary_puts_required_arguments_first(service):
    tool = register(service)
    rows = schema_summary(tool)
    assert [r["name"] for r in rows] == ["counterparty", "status"]
    assert rows[0]["required"] is True
