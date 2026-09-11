"""Seed, export and reset the intent catalogue.

Chroma is this service's system of record, so ``--export`` / ``--import`` are
the backup story: a corrupted store can be wiped and restored from JSON.

    uv run python -m scripts.seed              # add the demo catalogue
    uv run python -m scripts.seed --reset      # wipe everything first
    uv run python -m scripts.seed --export catalogue.json
    uv run python -m scripts.seed --import catalogue.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.errors import ConflictError
from app.models.domain import DomainCreate
from app.models.example import ExampleCreate
from app.models.intent import IntentCreate, ToolRef
from app.services.container import Container, build_container

CATALOGUE: list[dict[str, Any]] = [
    {
        "name": "contract",
        "description": "Contract and agreement operations",
        "system_instructions": (
            "Contracts are also called agreements, MSAs, SOWs and deals. "
            "A counterparty is the other organisation on the contract, never our own company. "
            "Contract ids look like C-1042. Never infer a status that is not stated."
        ),
        "user_instructions": (
            "Users are legal and procurement staff. They name vendors by brand "
            "(Microsoft, Oracle) and refer to renewal and expiry interchangeably."
        ),
        "intents": [
            {
                "name": "CONTRACT_SEARCH",
                "description": "Search contracts based on user-provided criteria",
                "tool": "search_contracts",
                "entity_schema": {
                    "counterparty": {"type": "string"},
                    "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED", "DRAFT"]},
                    "signed_after": {"type": "date"},
                },
                "extraction_hints": "'live' and 'current' mean ACTIVE.",
                "examples": [
                    "Find all contracts with Microsoft",
                    "Show me Microsoft agreements",
                    "Search contracts for Acme Corp",
                    "Give me all active contracts",
                    "Which contracts do we have with Salesforce?",
                    "List agreements signed with Oracle",
                    "Look up vendor contracts in the EU region",
                    "Show master service agreements for Deloitte",
                ],
            },
            {
                "name": "CONTRACT_SUMMARY",
                "description": "Summarize the key terms of a single contract",
                "tool": "summarize_contract",
                "entity_schema": {
                    "contract_id": {"type": "string"},
                    "counterparty": {"type": "string"},
                },
                "examples": [
                    "Summarize the Microsoft MSA",
                    "Give me a summary of contract C-1042",
                    "What are the key terms of the Acme agreement?",
                    "Explain the main obligations in this contract",
                    "Brief me on the Oracle license agreement",
                    "TL;DR of the Salesforce contract",
                    "What does the Deloitte SOW cover?",
                ],
            },
            {
                "name": "CONTRACT_EXPIRY",
                "description": "Find contracts by expiration or renewal date",
                "tool": "get_expiring_contracts",
                "entity_schema": {
                    "period": {"type": "string"},
                    "counterparty": {"type": "string"},
                    "expiration_year": {"type": "integer"},
                },
                "extraction_hints": "'this year' resolves to the current year from 'today'.",
                "examples": [
                    "Which contracts expire this year?",
                    "Show agreements expiring next month",
                    "What contracts are up for renewal in Q3?",
                    "List contracts ending before December",
                    "When does the Microsoft contract expire?",
                    "Find agreements that terminate in 2026",
                    "Contracts expiring in the next 90 days",
                ],
            },
            {
                "name": "CONTRACT_COMPARE",
                "description": "Compare two or more contracts clause by clause",
                "tool": "compare_contracts",
                "entity_schema": {"contract_ids": {"type": "array"}},
                "examples": [
                    "Compare the Microsoft and Oracle contracts",
                    "What is different between these two agreements?",
                    "Show the differences in liability clauses across vendors",
                    "Diff contract C-1042 against C-1099",
                    "How do the payment terms differ between these contracts?",
                    "Compare this year's MSA with last year's version",
                ],
            },
        ],
    },
    {
        "name": "employee",
        "description": "People and workforce operations",
        "system_instructions": (
            "Employee ids look like E-2201. 'I', 'me' and 'my' refer to the person asking; "
            "do not turn them into an employee_name. Departments and offices are locations "
            "or teams, not people."
        ),
        "user_instructions": (
            "Users are HR staff and employees asking about themselves or colleagues."
        ),
        "intents": [
            {
                "name": "EMPLOYEE_SEARCH",
                "description": "Find employees matching criteria",
                "tool": "search_employees",
                "entity_schema": {
                    "department": {"type": "string"},
                    "location": {"type": "string"},
                },
                "examples": [
                    "Find all engineers in Bangalore",
                    "Show me employees in the finance department",
                    "Who works in the London office?",
                    "List people reporting to Priya",
                    "Search for staff hired this year",
                    "Which employees are on the platform team?",
                ],
            },
            {
                "name": "EMPLOYEE_DETAILS",
                "description": "Fetch the profile of one employee",
                "tool": "get_employee",
                "entity_schema": {"employee_name": {"type": "string"}},
                "examples": [
                    "Show me the profile for Rahul Sharma",
                    "What is Anita's job title?",
                    "Give me details about employee E-2201",
                    "Who is the manager of Sam Patel?",
                    "Pull up the record for John Doe",
                    "What team does Meera belong to?",
                ],
            },
            {
                "name": "EMPLOYEE_LEAVE",
                "description": "Leave balance, applications and history",
                "tool": "get_leave_balance",
                "entity_schema": {
                    "employee_name": {"type": "string"},
                    "leave_type": {"type": "string"},
                },
                "examples": [
                    "How many leave days do I have left?",
                    "Show my leave balance",
                    "Apply for vacation next Friday",
                    "What is Rahul's remaining sick leave?",
                    "List my approved time off this quarter",
                    "How much parental leave am I entitled to?",
                ],
            },
        ],
    },
]


def seed(container: Container) -> dict[str, int]:
    stats = {"domains": 0, "intents": 0, "examples": 0}
    for domain_spec in CATALOGUE:
        try:
            domain = container.domains.create(
                DomainCreate(
                    name=domain_spec["name"],
                    description=domain_spec["description"],
                    system_instructions=domain_spec.get("system_instructions", ""),
                    user_instructions=domain_spec.get("user_instructions", ""),
                )
            )
            stats["domains"] += 1
        except ConflictError:
            domain = container.domains.resolve(domain_spec["name"])
            print(f"  domain '{domain.name}' already exists, reusing")

        for intent_spec in domain_spec["intents"]:
            try:
                intent = container.intents.create(
                    domain.id,
                    IntentCreate(
                        name=intent_spec["name"],
                        description=intent_spec["description"],
                        tool=ToolRef(name=intent_spec["tool"], version="v1"),
                        entity_schema=intent_spec["entity_schema"],
                        extraction_hints=intent_spec.get("extraction_hints", ""),
                    ),
                )
                stats["intents"] += 1
            except ConflictError:
                intent = container.intents.find_by_name(domain.id, intent_spec["name"])

            payloads = []
            for text in intent_spec["examples"]:
                payloads.append(ExampleCreate(text=text))
            added = _add_new_examples(container, domain.id, intent.id, payloads)
            stats["examples"] += added
        container.index_manager.rebuild(domain.id)
    return stats


def _add_new_examples(container: Container, domain_id: str, intent_id: str, payloads) -> int:
    """Add examples one by one so an existing one does not abort the batch."""
    added = 0
    for payload in payloads:
        try:
            container.examples.add(domain_id, intent_id, payload)
            added += 1
        except ConflictError:
            continue
    return added


def export_catalogue(container: Container) -> dict[str, Any]:
    payload: dict[str, Any] = {"domains": []}
    for domain in container.domains.list():
        domain_entry = {
            "name": domain.name,
            "description": domain.description,
            "system_instructions": domain.system_instructions,
            "user_instructions": domain.user_instructions,
            "intents": [],
        }
        for intent in container.intents.list(domain.id):
            domain_entry["intents"].append(
                {
                    "name": intent.name,
                    "description": intent.description,
                    "tool": intent.tool.name,
                    "entity_schema": intent.entity_schema,
                    "extraction_hints": intent.extraction_hints,
                    "examples": [e.text for e in container.examples.list(domain.id, intent.id)],
                }
            )
        payload["domains"].append(domain_entry)
    return payload


def import_catalogue(container: Container, payload: dict[str, Any]) -> dict[str, int]:
    global CATALOGUE
    CATALOGUE = payload["domains"]
    return seed(container)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the intent catalogue")
    parser.add_argument("--reset", action="store_true", help="delete every record first")
    parser.add_argument("--export", metavar="PATH", help="write the current catalogue to JSON")
    parser.add_argument(
        "--import", dest="import_path", metavar="PATH", help="load a catalogue from JSON"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    print(f"chroma: {settings.chroma_mode} at {settings.chroma_path}")
    container = build_container(settings)

    if args.export:
        Path(args.export).write_text(
            json.dumps(export_catalogue(container), indent=2, ensure_ascii=False), "utf-8"
        )
        print(f"exported catalogue to {args.export}")
        return 0

    if args.reset:
        container.store.reset_all()
        print("store reset")

    if args.import_path:
        payload = json.loads(Path(args.import_path).read_text("utf-8"))
        stats = import_catalogue(container, payload)
    else:
        stats = seed(container)

    print(
        f"seeded {stats['domains']} domains, {stats['intents']} intents, "
        f"{stats['examples']} examples"
    )
    for domain in container.domains.list():
        status = container.index_manager.status(domain.id)
        print(
            f"  {domain.name}: {domain.intent_count} intents, "
            f"{domain.example_count} examples, index v{status.version} ({status.state})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
