"""Generated intents are model output, which means untrusted input."""

import pytest

from app.models.domain import DomainConfig
from app.models.intent import IntentConfig
from app.services.llm.client import LLMUnavailable, ScriptedLLMClient, StubLLMClient
from app.services.llm.intent_generator import (
    MAX_INTENTS,
    IntentGenerator,
    _clean_entity_schema,
    _clean_examples,
    _clean_intent_name,
    _clean_tool_name,
)
from app.services.llm.mock import MockLLMClient

DOMAIN = DomainConfig(name="contract", description="vendor agreements")


def intent_payload(**overrides):
    base = {
        "name": "CONTRACT_SEARCH",
        "description": "Search contracts",
        "tool": "search_contracts",
        "entity_schema": {"counterparty": {"type": "string"}},
        "examples": ["find contracts", "show me contracts"],
    }
    base.update(overrides)
    return {"intents": [base]}


# ------------------------------------------------------------------- name rules
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("contract search", "CONTRACT_SEARCH"),
        ("  Contract-Search  ", "CONTRACT-SEARCH"),
        ("contract  search!!", "CONTRACT_SEARCH"),
        ("__CONTRACT__", "CONTRACT"),
        ("CONTRACT_SEARCH", "CONTRACT_SEARCH"),
    ],
)
def test_names_are_coerced_to_the_models_pattern(given, expected):
    assert _clean_intent_name(given) == expected


@pytest.mark.parametrize("given", ["", "   ", "!!!", None, 42, "___"])
def test_unusable_names_are_rejected(given):
    assert _clean_intent_name(given) == ""


def test_a_tool_name_is_derived_when_the_model_gives_none():
    assert _clean_tool_name(None, "CONTRACT_SEARCH") == "contract_search"
    assert _clean_tool_name("Search Contracts!", "X") == "search_contracts"


# ----------------------------------------------------------------- schema rules
def test_unknown_entity_types_are_dropped():
    schema, notes = _clean_entity_schema({"a": {"type": "string"}, "b": {"type": "wormhole"}})
    assert schema == {"a": {"type": "string"}}
    assert any("wormhole" in n for n in notes)


def test_a_bare_type_string_is_accepted():
    schema, _ = _clean_entity_schema({"a": "integer"})
    assert schema == {"a": {"type": "integer"}}


def test_an_enum_without_values_is_dropped():
    schema, notes = _clean_entity_schema({"status": {"type": "enum", "values": []}})
    assert schema == {}
    assert any("enum with no values" in n for n in notes)


def test_enum_values_and_required_survive():
    schema, _ = _clean_entity_schema(
        {"status": {"type": "enum", "values": ["ACTIVE", " EXPIRED "], "required": True}}
    )
    assert schema["status"] == {"type": "enum", "required": True, "values": ["ACTIVE", "EXPIRED"]}


def test_array_items_fall_back_to_string():
    schema, _ = _clean_entity_schema({"tags": {"type": "array", "items": {"type": "nonsense"}}})
    assert schema["tags"]["items"] == {"type": "string"}


def test_a_non_object_schema_is_discarded():
    schema, notes = _clean_entity_schema(["not", "an", "object"])
    assert schema == {} and notes


# --------------------------------------------------------------- example rules
def test_examples_are_deduplicated_by_index_normalisation():
    kept, notes = _clean_examples(["Find Contracts", "find   contracts", "show me"], 10)
    assert kept == ["Find Contracts", "show me"]
    assert any("duplicate" in n for n in notes)


def test_examples_already_on_the_intent_are_dropped():
    kept, _ = _clean_examples(["find contracts", "new one"], 10, existing=["Find contracts"])
    assert kept == ["new one"]


def test_non_text_examples_are_dropped():
    kept, notes = _clean_examples(["ok", 42, None, {"a": 1}], 10)
    assert kept == ["ok"]
    assert len([n for n in notes if "non-text" in n]) == 3


def test_the_requested_count_is_a_hard_cap():
    kept, _ = _clean_examples([f"example {i}" for i in range(50)], 5)
    assert len(kept) == 5


# -------------------------------------------------------------------- generator
@pytest.mark.asyncio
async def test_a_good_proposal_is_returned_validated():
    generator = IntentGenerator(ScriptedLLMClient(default=intent_payload()))
    result = await generator.generate_intents(DOMAIN, count=3)

    assert result.status == "ok"
    assert result.intents[0].name == "CONTRACT_SEARCH"
    assert result.intents[0].tool.name == "search_contracts"


@pytest.mark.asyncio
async def test_names_that_already_exist_are_skipped():
    generator = IntentGenerator(ScriptedLLMClient(default=intent_payload()))
    result = await generator.generate_intents(DOMAIN, existing_names=["contract_search"])

    assert result.status == "failed"
    assert any("already exists" in n for n in result.skipped)


@pytest.mark.asyncio
async def test_duplicate_names_within_one_response_are_skipped():
    payload = {"intents": [intent_payload()["intents"][0], intent_payload()["intents"][0]]}
    result = await IntentGenerator(ScriptedLLMClient(default=payload)).generate_intents(DOMAIN)

    assert len(result.intents) == 1
    assert any("already exists" in n for n in result.skipped)


@pytest.mark.asyncio
async def test_the_requested_intent_count_is_capped():
    payload = {"intents": [dict(intent_payload()["intents"][0], name=f"I_{i}") for i in range(40)]}
    result = await IntentGenerator(ScriptedLLMClient(default=payload)).generate_intents(
        DOMAIN, count=999
    )
    assert len(result.intents) == MAX_INTENTS


@pytest.mark.asyncio
async def test_a_response_with_no_intents_fails_cleanly():
    for payload in [{}, {"intents": []}, {"intents": "nope"}]:
        result = await IntentGenerator(ScriptedLLMClient(default=payload)).generate_intents(DOMAIN)
        assert result.status == "failed"


@pytest.mark.asyncio
async def test_provider_failure_is_reported_not_raised():
    generator = IntentGenerator(ScriptedLLMClient(raises=LLMUnavailable("timeout")))
    result = await generator.generate_intents(DOMAIN)
    assert result.status == "failed" and "timeout" in result.detail


@pytest.mark.asyncio
async def test_no_provider_is_a_distinct_status():
    result = await IntentGenerator(StubLLMClient()).generate_intents(DOMAIN)
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_the_prompt_carries_the_domain_and_existing_intents():
    client = ScriptedLLMClient(default=intent_payload())
    await IntentGenerator(client).generate_intents(DOMAIN, brief="renewals", existing_names=["X"])
    _, user = client.calls[0]
    assert "generate_domain_intents" in user
    assert "contract" in user and "renewals" in user and '"X"' in user


# ------------------------------------------------------------ example generation
@pytest.mark.asyncio
async def test_example_generation_excludes_what_the_intent_already_has():
    client = ScriptedLLMClient(default={"examples": ["find contracts", "brand new"]})
    intent = IntentConfig(name="CONTRACT_SEARCH", domain_id=DOMAIN.id)
    result = await IntentGenerator(client).generate_examples(
        DOMAIN, intent, count=5, existing=["Find contracts"]
    )
    assert result.examples == ["brand new"]


@pytest.mark.asyncio
async def test_example_generation_fails_when_nothing_is_usable():
    client = ScriptedLLMClient(default={"examples": [1, None]})
    intent = IntentConfig(name="CONTRACT_SEARCH", domain_id=DOMAIN.id)
    result = await IntentGenerator(client).generate_examples(DOMAIN, intent)
    assert result.status == "failed"


@pytest.mark.asyncio
async def test_the_mock_provider_drafts_deterministic_intents():
    generator = IntentGenerator(MockLLMClient())
    first = await generator.generate_intents(DOMAIN, count=4, examples_per_intent=5)
    second = await generator.generate_intents(DOMAIN, count=4, examples_per_intent=5)

    assert first.status == "ok"
    assert [i.name for i in first.intents] == [i.name for i in second.intents]
    assert all(i.name.startswith("CONTRACT_") for i in first.intents)
    assert all(len(i.examples) == 5 for i in first.intents)
