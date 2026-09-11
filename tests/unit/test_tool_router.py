from app.models.intent import IntentConfig, ToolRef
from app.services.tool_router import ToolRouter

ROUTER = ToolRouter()


def intent(**overrides) -> IntentConfig:
    base = {
        "name": "CONTRACT_SEARCH",
        "domain_id": "d1",
        "tool": ToolRef(name="search_contracts", version="v2"),
        "entity_schema": {
            "counterparty": {"type": "string", "required": True},
            "status": {"type": "string"},
        },
    }
    base.update(overrides)
    return IntentConfig(**base)


def test_entities_become_tool_arguments():
    call = ROUTER.route(intent(), {"counterparty": "Microsoft", "status": "ACTIVE"})
    assert call.name == "search_contracts"
    assert call.version == "v2"
    assert call.arguments == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert call.ready is True


def test_an_intent_with_no_tool_routes_nowhere():
    assert ROUTER.route(intent(tool=ToolRef()), {"counterparty": "Microsoft"}) is None


def test_arguments_outside_the_schema_never_reach_the_tool():
    call = ROUTER.route(intent(), {"counterparty": "Microsoft", "injected": "rm -rf /"})
    assert call.arguments == {"counterparty": "Microsoft"}


def test_missing_required_arguments_block_readiness():
    call = ROUTER.route(intent(), {"status": "ACTIVE"})
    assert call.missing_required == ["counterparty"]
    assert call.ready is False


def test_an_intent_without_a_schema_accepts_no_arguments():
    call = ROUTER.route(intent(entity_schema={}), {"counterparty": "Microsoft"})
    assert call.arguments == {}
    assert call.ready is True
