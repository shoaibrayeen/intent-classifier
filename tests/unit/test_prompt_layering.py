"""Domain and intent instructions must reach the extractor, in the right place."""

import json

from app.models.classification import SessionTurn
from app.models.domain import DomainConfig
from app.models.intent import IntentConfig
from app.services.llm.entity_extractor import (
    BASE_SYSTEM_PROMPT,
    build_system_prompt,
    build_user_prompt,
)


def test_base_rules_are_always_present():
    assert build_system_prompt(None, None) == BASE_SYSTEM_PROMPT


def test_domain_and_intent_instructions_are_appended_in_order():
    domain = DomainConfig(name="contract", system_instructions="Counterparty is the other party.")
    intent = IntentConfig(
        name="SEARCH", domain_id=domain.id, extraction_hints="'live' means ACTIVE."
    )
    prompt = build_system_prompt(domain, intent)
    assert prompt.startswith(BASE_SYSTEM_PROMPT)
    assert prompt.index("Counterparty is the other party.") < prompt.index("'live' means ACTIVE.")
    assert "Domain instructions (contract)" in prompt
    assert "Intent notes (SEARCH)" in prompt


def test_blank_instructions_add_nothing():
    domain = DomainConfig(name="contract", system_instructions="   ")
    assert build_system_prompt(domain, None) == BASE_SYSTEM_PROMPT


def test_user_prompt_carries_instructions_history_request_and_schema():
    domain = DomainConfig(name="contract", user_instructions="Users name vendors by brand.")
    history = [
        SessionTurn(
            turn=1,
            text="Show Microsoft contracts",
            intent="SEARCH",
            entities={"counterparty": "Microsoft"},
        )
    ]
    payload = json.loads(
        build_user_prompt(
            "when do they expire",
            {"counterparty": {"type": "string"}},
            today="2026-09-12",
            domain=domain,
            history=history,
        )
    )
    assert payload["instructions"] == "Users name vendors by brand."
    assert payload["conversation_history"][0]["entities"] == {"counterparty": "Microsoft"}
    assert payload["request"] == "when do they expire"
    assert payload["today"] == "2026-09-12"
    assert list(payload)[:2] == ["instructions", "conversation_history"]


def test_user_prompt_without_extras_is_minimal():
    payload = json.loads(build_user_prompt("q", {"a": {"type": "string"}}))
    assert set(payload) == {"request", "entity_schema"}
