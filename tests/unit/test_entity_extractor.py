"""Extraction must degrade to 'no entities', never to a failed classification."""

import pytest

from app.config import Settings
from app.services.llm.client import (
    LLMUnavailable,
    ScriptedLLMClient,
    StubLLMClient,
    parse_json_object,
)
from app.services.llm.entity_extractor import EntityExtractor

SCHEMA = {
    "counterparty": {"type": "string"},
    "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
}


def settings(**overrides) -> Settings:
    base = {
        "chroma_mode": "ephemeral",
        "entity_extraction_enabled": True,
        "openai_api_key": "test-key",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.mark.asyncio
async def test_extracts_and_validates_what_the_model_returns():
    client = ScriptedLLMClient(default={"counterparty": "Microsoft", "status": "active"})
    extractor = EntityExtractor(settings(), client)

    entities, info = await extractor.extract("Find active Microsoft contracts", SCHEMA)

    assert entities == {"counterparty": "Microsoft", "status": "ACTIVE"}
    assert info.status == "ok"
    assert info.rejected == []


@pytest.mark.asyncio
async def test_hallucinated_keys_are_dropped_and_reported():
    client = ScriptedLLMClient(default={"counterparty": "Microsoft", "invented": "value"})
    entities, info = await EntityExtractor(settings(), client).extract("anything", SCHEMA)

    assert entities == {"counterparty": "Microsoft"}
    assert any("invented" in note for note in info.rejected)


@pytest.mark.asyncio
async def test_provider_failure_does_not_raise():
    client = ScriptedLLMClient(raises=LLMUnavailable("connection refused"))
    entities, info = await EntityExtractor(settings(), client).extract("anything", SCHEMA)

    assert entities == {}
    assert info.status == "failed"
    assert "connection refused" in info.detail


@pytest.mark.asyncio
async def test_an_unexpected_exception_is_contained():
    client = ScriptedLLMClient(raises=RuntimeError("boom"))
    entities, info = await EntityExtractor(settings(), client).extract("anything", SCHEMA)

    assert entities == {}
    assert info.status == "failed"


@pytest.mark.asyncio
async def test_no_provider_means_unavailable_not_an_error():
    extractor = EntityExtractor(settings(openai_api_key=""), StubLLMClient())
    entities, info = await extractor.extract("anything", SCHEMA)

    assert entities == {}
    assert info.status == "unavailable"


@pytest.mark.asyncio
async def test_whether_to_extract_is_the_callers_decision_not_the_extractors():
    """The global flag lives in the orchestrator, so a per-request override is
    not silently overruled here. Calling extract() means extraction was asked
    for; see tests/integration/test_entities_and_tools.py for who decides."""
    client = ScriptedLLMClient(default={"counterparty": "Microsoft"})
    extractor = EntityExtractor(settings(entity_extraction_enabled=False), client)

    entities, _ = await extractor.extract("anything", SCHEMA)

    assert entities == {"counterparty": "Microsoft"}


@pytest.mark.asyncio
async def test_an_intent_without_a_schema_skips_the_call():
    client = ScriptedLLMClient(default={"anything": "at all"})
    entities, info = await EntityExtractor(settings(), client).extract("anything", {})

    assert entities == {}
    assert info.status == "skipped"
    assert client.calls == []


@pytest.mark.asyncio
async def test_the_prompt_carries_the_schema_and_todays_date():
    client = ScriptedLLMClient(default={})
    await EntityExtractor(settings(), client).extract("q", SCHEMA, today="2026-09-12")

    _, user = client.calls[0]
    assert "counterparty" in user
    assert "2026-09-12" in user


def test_json_is_recovered_from_a_fenced_response():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 1}') == {"a": 1}


def test_unusable_responses_raise_rather_than_returning_junk():
    for bad in ["not json at all", "[1, 2, 3]", ""]:
        with pytest.raises(LLMUnavailable):
            parse_json_object(bad)
