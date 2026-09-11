"""The mock provider is rule-based and must be predictable."""

import pytest

from app.services.llm.mock import MockLLMClient, extract_by_rule

SCHEMA = {
    "counterparty": {"type": "string"},
    "contract_id": {"type": "string"},
    "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
    "signed_after": {"type": "date"},
    "expiration_year": {"type": "integer"},
    "is_renewal": {"type": "boolean"},
}


def test_finds_a_proper_noun_an_enum_and_a_date():
    out = extract_by_rule("Show active Microsoft contracts signed after March 1, 2025", SCHEMA)
    assert out["counterparty"] == "Microsoft"
    assert out["status"] == "ACTIVE"
    assert out["signed_after"] == "2025-03-01"


def test_identifier_fields_never_fall_back_to_a_name():
    out = extract_by_rule("Summarize the deal with Acme Corp", SCHEMA)
    assert out["counterparty"] == "Acme Corp"
    assert "contract_id" not in out


def test_identifier_pattern_is_recognised():
    assert extract_by_rule("Summarize contract C-1042", SCHEMA)["contract_id"] == "C-1042"


def test_relative_year_resolves_against_today():
    out = extract_by_rule("Which contracts expire this year?", SCHEMA, today="2026-09-12")
    assert out["expiration_year"] == 2026


def test_leading_capitalised_verbs_are_not_names():
    out = extract_by_rule("Find Oracle agreements", SCHEMA)
    assert out["counterparty"] == "Oracle"


def test_nothing_to_find_returns_empty():
    assert extract_by_rule("when does it expire", SCHEMA) == {}


@pytest.mark.asyncio
async def test_client_reads_the_same_prompt_shape_a_real_provider_gets():
    from app.services.llm.entity_extractor import build_user_prompt

    client = MockLLMClient()
    user = build_user_prompt("Find Microsoft contracts", {"counterparty": {"type": "string"}})
    out = await client.complete_json("system", user)
    assert out == {"counterparty": "Microsoft"}
    assert len(client.calls) == 1
