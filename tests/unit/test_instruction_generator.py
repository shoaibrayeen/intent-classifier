"""Generated instructions are model output, which means untrusted input."""

import pytest

from app.services.llm.client import LLMUnavailable, ScriptedLLMClient, StubLLMClient
from app.services.llm.instruction_generator import (
    MAX_INSTRUCTION_CHARS,
    InstructionGenerator,
)
from app.services.llm.mock import MockLLMClient


@pytest.mark.asyncio
async def test_a_good_draft_comes_back_trimmed_and_typed():
    client = ScriptedLLMClient(
        default={
            "system_instructions": "  Contracts are  agreements.\n\nNever guess.  ",
            "user_instructions": "Users are legal staff.",
        }
    )
    result = await InstructionGenerator(client).generate("contract", "vendor agreements")

    assert result.status == "ok"
    assert result.system_instructions == "Contracts are agreements. Never guess."
    assert result.user_instructions == "Users are legal staff."


@pytest.mark.asyncio
async def test_overlong_output_is_capped_to_the_field_limit():
    client = ScriptedLLMClient(
        default={"system_instructions": "x" * 10000, "user_instructions": "ok"}
    )
    result = await InstructionGenerator(client).generate("contract", "brief")
    assert len(result.system_instructions) == MAX_INSTRUCTION_CHARS


@pytest.mark.asyncio
async def test_non_string_values_are_discarded():
    client = ScriptedLLMClient(
        default={"system_instructions": ["a", "list"], "user_instructions": {"not": "text"}}
    )
    result = await InstructionGenerator(client).generate("contract", "brief")
    assert result.status == "failed"
    assert "no usable instructions" in result.detail


@pytest.mark.asyncio
async def test_one_usable_block_is_enough():
    client = ScriptedLLMClient(default={"system_instructions": "Only this.", "extra": "ignored"})
    result = await InstructionGenerator(client).generate("contract", "brief")
    assert result.status == "ok"
    assert result.system_instructions == "Only this."
    assert result.user_instructions == ""


@pytest.mark.asyncio
async def test_provider_failure_is_reported_not_raised():
    client = ScriptedLLMClient(raises=LLMUnavailable("timeout"))
    result = await InstructionGenerator(client).generate("contract", "brief")
    assert result.status == "failed"
    assert "timeout" in result.detail


@pytest.mark.asyncio
async def test_no_provider_is_a_distinct_status():
    result = await InstructionGenerator(StubLLMClient()).generate("contract", "brief")
    assert result.status == "unavailable"


@pytest.mark.asyncio
async def test_the_prompt_carries_name_and_brief():
    client = ScriptedLLMClient(default={"system_instructions": "s", "user_instructions": "u"})
    await InstructionGenerator(client).generate("finance", "invoices and payments")
    system, user = client.calls[0]
    assert "generate_domain_instructions" in user
    assert "finance" in user and "invoices and payments" in user
    assert "system_instructions" in system  # the contract is stated to the model


@pytest.mark.asyncio
async def test_the_mock_provider_drafts_deterministically():
    generator = InstructionGenerator(MockLLMClient())
    first = await generator.generate("contract", "vendor agreements and renewals")
    second = await generator.generate("contract", "vendor agreements and renewals")
    assert first.status == "ok"
    assert first.system_instructions == second.system_instructions
    assert "contract" in first.system_instructions
    assert "vendor agreements and renewals" in first.system_instructions
    assert "never infer" in first.system_instructions.lower()
