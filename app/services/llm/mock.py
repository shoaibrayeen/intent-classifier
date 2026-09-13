"""A deterministic, offline stand-in for a language model.

It reads the same prompt a real provider would receive and fills the schema
from the request text with plain rules: enum values by name, dates and years by
pattern, numbers by digit, booleans by keyword, and strings by capitalised
words. It is not clever, and that is the point -- it makes the *whole* pipeline
(instructions, history, validation, carry-over, tool routing) exercisable
without a key, and its output is stable enough to test against.

Enable with ``LLM_PROVIDER=mock``.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LONG_DATE = re.compile(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_YEAR = re.compile(r"\b(20\d{2}|19\d{2})\b")
_NUMBER = re.compile(r"(?<![\w-])-?\d+(?:[.,]\d+)?(?![\w-])")
_CAPITALISED = re.compile(r"\b([A-Z][A-Za-z0-9&'’.-]+(?:\s+[A-Z][A-Za-z0-9&'’.-]+)*)\b")
_ID_LIKE = re.compile(r"\b([A-Z]{1,4}-\d{2,})\b")

# Words that are capitalised for reasons other than being a name.
_SKIP = {
    "now",
    "then",
    "also",
    "next",
    "ok",
    "okay",
    "but",
    "so",
    "well",
    "great",
    "thanks",
    "hi",
    "hello",
    "hey",
    "yes",
    "no",
    "again",
    "instead",
    "same",
    "only",
    "i",
    "show",
    "find",
    "list",
    "give",
    "search",
    "what",
    "which",
    "when",
    "who",
    "how",
    "the",
    "a",
    "an",
    "all",
    "my",
    "our",
    "me",
    "please",
    "compare",
    "summarize",
    "summarise",
    "tell",
    "pull",
    "get",
    "display",
    "can",
    "could",
    "do",
    "does",
    "is",
    "are",
    "and",
    "or",
    "contracts",
    "contract",
    "agreements",
    "agreement",
    "employees",
    "employee",
    "q1",
    "q2",
    "q3",
    "q4",
    "tl",
    "dr",
    "msa",
    "sow",
    *MONTHS.split("|"),
}
_TRUE_WORDS = {"yes", "true", "active", "enabled", "renew", "renewal", "renewed"}
_FALSE_WORDS = {"no", "false", "inactive", "disabled", "expired", "cancelled", "canceled"}


class MockLLMClient:
    """Fills an entity schema from the request text by rule."""

    name = "mock"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def configured(self) -> bool:
        return True

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            return {}
        task = payload.get("task")
        if task == "generate_domain_instructions":
            return generate_instructions_by_rule(
                str(payload.get("domain_name", "")), str(payload.get("description", ""))
            )
        if task == "generate_domain_intents":
            return generate_intents_by_rule(
                str(payload.get("domain_name", "")),
                int(payload.get("intent_count", 5) or 5),
                int(payload.get("examples_per_intent", 6) or 0),
                [str(n) for n in payload.get("existing_intents", [])],
                payload.get("available_mcp_tools") or [],
            )
        if task == "generate_intent_examples":
            return generate_examples_by_rule(
                str(payload.get("domain_name", "")),
                str(payload.get("intent_name", "")),
                int(payload.get("example_count", 6) or 6),
                [str(t) for t in payload.get("existing_examples", [])],
            )
        text = str(payload.get("request", ""))
        schema = payload.get("entity_schema") or {}
        today = payload.get("today")
        return extract_by_rule(text, schema, today)


def extract_by_rule(text: str, schema: dict[str, Any], today: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    lowered = text.casefold()
    used_spans: list[str] = []

    for name, spec in schema.items():
        spec = spec if isinstance(spec, dict) else {"type": str(spec)}
        declared = str(spec.get("type", "string")).lower()
        value: Any = None

        if declared == "enum":
            for option in spec.get("values", []):
                if re.search(rf"\b{re.escape(str(option).casefold())}\b", lowered):
                    value = option
                    break
        elif declared == "date":
            value = _find_date(text, lowered, today)
        elif declared == "integer":
            if "year" in name.lower():
                resolved = _find_date(text, lowered, today)
                value = int(resolved[:4]) if resolved and _YEAR.fullmatch(resolved[:4]) else None
            else:
                match = _NUMBER.search(text)
                if match:
                    value = int(match.group(0).replace(",", "").split(".")[0])
        elif declared == "number":
            match = _NUMBER.search(text)
            if match:
                value = float(match.group(0).replace(",", ""))
        elif declared == "boolean":
            words = set(re.findall(r"[a-z]+", lowered))
            if words & _TRUE_WORDS:
                value = True
            elif words & _FALSE_WORDS:
                value = False
        elif declared == "array":
            names = _proper_nouns(text, used_spans)
            value = names or None
        else:  # string
            if "id" in name.lower():
                # An identifier field never falls back to a name: "March" is
                # not a contract id, and a wrong id is worse than none.
                match = _ID_LIKE.search(text)
                value = match.group(1) if match else None
            else:
                names = _proper_nouns(text, used_spans)
                if names:
                    value = names[0]
                    used_spans.append(names[0])

        if value is not None:
            result[name] = value
    return result


def _find_date(text: str, lowered: str, today: str | None) -> str | None:
    if match := _ISO_DATE.search(text):
        return match.group(0)
    if match := _LONG_DATE.search(text):
        month = datetime.strptime(match.group(1)[:3], "%b").month
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    if today and ("this year" in lowered or "current year" in lowered):
        return today[:4]
    if today and "next year" in lowered:
        return str(int(today[:4]) + 1)
    if match := _YEAR.search(text):
        return match.group(0)
    return None


def _proper_nouns(text: str, exclude: list[str]) -> list[str]:
    found: list[str] = []
    for match in _CAPITALISED.finditer(text):
        candidate = match.group(1).strip(".,;:!?")
        tokens = candidate.split()
        # Drop leading capitalised function words ("Show Microsoft" -> "Microsoft").
        while tokens and tokens[0].casefold() in _SKIP:
            tokens = tokens[1:]
        if not tokens:
            continue
        candidate = " ".join(tokens)
        if candidate.casefold() in _SKIP or candidate in exclude or candidate in found:
            continue
        if _ID_LIKE.fullmatch(candidate) or _YEAR.fullmatch(candidate):
            continue
        found.append(candidate)
    return found


#: Keyword to (role, audience). The generated system instructions open with the
#: role, because how the assistant should act is the first thing the extractor
#: needs to know: "contractual" implies a legal analyst, "add to cart" implies a
#: shopping assistant, and the two read the same sentence very differently.
_ROLE_BY_KEYWORD: list[tuple[tuple[str, ...], str, str]] = [
    (
        ("contract", "agreement", "legal", "clause", "msa", "nda", "obligation", "counsel"),
        "a contracts and legal analyst",
        "legal, procurement and contract-management staff",
    ),
    (
        (
            "cart",
            "checkout",
            "order",
            "product",
            "catalog",
            "catalogue",
            "shop",
            "store",
            "ecommerce",
            "e-commerce",
            "sku",
            "basket",
        ),
        "an e-commerce shopping assistant",
        "shoppers and store operations staff",
    ),
    (
        ("invoice", "payment", "billing", "expense", "finance", "accounting", "ledger"),
        "a finance and accounts analyst",
        "finance and accounts-payable staff",
    ),
    (
        ("employee", "hr", "payroll", "leave", "attendance", "recruit", "candidate", "staff"),
        "an HR and people-operations assistant",
        "HR staff and employees asking about themselves or colleagues",
    ),
    (
        ("ticket", "incident", "support", "helpdesk", "complaint", "sla"),
        "a customer-support desk assistant",
        "support agents and customers reporting problems",
    ),
    (
        ("patient", "clinical", "medical", "diagnosis", "prescription", "health"),
        "a clinical records assistant",
        "clinical and administrative healthcare staff",
    ),
    (
        ("shipment", "logistics", "delivery", "warehouse", "inventory", "stock", "freight"),
        "a logistics and inventory assistant",
        "warehouse, logistics and fulfilment staff",
    ),
]


def infer_role(name: str, description: str) -> tuple[str, str]:
    """Pick the role a domain implies, from its name and description."""
    haystack = f"{name} {description}".casefold()
    for keywords, role, audience in _ROLE_BY_KEYWORD:
        if any(re.search(rf"\b{re.escape(word)}", haystack) for word in keywords):
            return role, audience
    subject = name.strip() or "this domain"
    return f"a {subject} specialist", f"people working with {subject}"


def generate_instructions_by_rule(name: str, description: str) -> dict[str, str]:
    """Deterministic instruction drafts, so the flow runs offline.

    Templated on purpose: the mock's job is to exercise the pipeline, and a
    predictable draft is exactly what the admin then edits on the domain page.
    """
    name = name.strip() or "this"
    about = description.strip().rstrip(".")
    focus = about if about else f"{name} operations"
    role, audience = infer_role(name, about)
    return {
        "system_instructions": (
            f"You are acting as {role}. This is the {name} domain, which covers "
            f"{focus}. Interpret every request in that role and use only the "
            f"terminology the request itself contains. Never infer a value that "
            f"is not stated; when an entity is not mentioned, leave it out "
            f"rather than guessing."
        ),
        "user_instructions": (
            f"Users are {audience}, working with {focus}. They phrase requests "
            f"informally, name things by their common business terms, and often "
            f"refer back to items from earlier in the conversation."
        ),
    }


#: (operation suffix, tool verb, example templates). ``{d}`` is the domain name.
#: Templated on purpose: the mock exists to exercise the pipeline end to end,
#: and the phrasings have to be distinct enough that retrieval can actually
#: tell the generated intents apart.
_OPERATION_TEMPLATES: list[tuple[str, str, list[str]]] = [
    (
        "SEARCH",
        "search",
        [
            "find all {d}s",
            "show me {d}s for Microsoft",
            "search {d}s",
            "list every {d}",
            "which {d}s do we have",
            "look up {d}s for Oracle",
            "get me the {d}s",
            "show {d}s with Acme Corp",
        ],
    ),
    (
        "DETAILS",
        "get",
        [
            "show me the details of {d} C-1042",
            "open {d} C-1042",
            "what are the details for this {d}",
            "give me {d} C-1099",
            "pull up {d} C-1042",
            "I need the full {d} record",
            "display {d} C-2210",
            "details for {d} C-1042 please",
        ],
    ),
    (
        "SUMMARY",
        "summarize",
        [
            "summarize {d} C-1042",
            "give me a summary of this {d}",
            "tl;dr of {d} C-1042",
            "brief me on {d} C-1099",
            "what does this {d} say",
            "short overview of {d} C-1042",
            "summarise the {d} with Acme Corp",
            "key points of {d} C-2210",
        ],
    ),
    (
        "STATUS",
        "get",
        [
            "what is the status of {d} C-1042",
            "is {d} C-1042 still active",
            "check the status of this {d}",
            "{d} C-1099 status",
            "show me the state of {d} C-1042",
            "has {d} C-2210 been approved",
            "current status for {d} C-1042",
            "tell me if {d} C-1099 is active",
        ],
    ),
    (
        "COMPARE",
        "compare",
        [
            "compare {d} C-1042 with C-1099",
            "what changed between these two {d}s",
            "diff {d} C-1042 and C-1099",
            "put {d} C-1042 side by side with C-2210",
            "show differences between {d} C-1042 and C-1099",
            "how do these {d}s differ",
            "compare the two {d}s",
            "contrast {d} C-1099 against C-2210",
        ],
    ),
    (
        "EXPIRY",
        "get_expiring",
        [
            "which {d}s expire this year",
            "show {d}s expiring next month",
            "when does {d} C-1042 expire",
            "{d}s coming up for renewal",
            "list {d}s ending soon",
            "what {d}s are due to end in 2026",
            "find {d}s past their end date",
            "upcoming {d} expirations",
        ],
    ),
    (
        "CREATE",
        "create",
        [
            "create a new {d}",
            "start a {d} with Acme Corp",
            "draft a new {d}",
            "set up a {d} for Oracle",
            "I want to raise a {d}",
            "open a new {d} record",
            "add a {d} for Microsoft",
            "begin a new {d}",
        ],
    ),
    (
        "DELETE",
        "delete",
        [
            "delete {d} C-1042",
            "remove this {d}",
            "cancel {d} C-1099",
            "get rid of {d} C-2210",
            "terminate {d} C-1042",
            "close out this {d}",
            "void {d} C-1099",
            "archive {d} C-1042",
        ],
    ),
]

_SCHEMA_BY_OPERATION: dict[str, dict[str, dict]] = {
    "SEARCH": {
        "counterparty": {"type": "string"},
        "status": {"type": "enum", "values": ["ACTIVE", "EXPIRED", "DRAFT"]},
    },
    "DETAILS": {"record_id": {"type": "string", "required": True}},
    "SUMMARY": {"record_id": {"type": "string", "required": True}},
    "STATUS": {"record_id": {"type": "string", "required": True}},
    "COMPARE": {"record_id": {"type": "string", "required": True}},
    "EXPIRY": {"counterparty": {"type": "string"}, "expiration_year": {"type": "integer"}},
    "CREATE": {"counterparty": {"type": "string"}},
    "DELETE": {"record_id": {"type": "string", "required": True}},
}


def _singular(name: str) -> str:
    name = name.strip().casefold() or "record"
    return name[:-1] if name.endswith("s") and not name.endswith("ss") else name


def _pick_mcp_tool(offered: list[dict], verb: str, noun: str, suffix: str) -> str | None:
    """Choose an offered tool whose name looks like this operation.

    Matched on the operation word rather than the whole tool name, so a
    registry using its own naming still binds: SEARCH finds "search_contracts",
    "find_contracts" or "contracts_query".
    """
    if not offered:
        return None
    synonyms = {
        "SEARCH": ("search", "find", "query", "list"),
        "DETAILS": ("get", "detail", "fetch", "read", "show"),
        "SUMMARY": ("summar", "digest", "brief"),
        "STATUS": ("status", "state", "check"),
        "COMPARE": ("compare", "diff"),
        "EXPIRY": ("expir", "renew", "due"),
        "CREATE": ("create", "add", "new", "open", "draft"),
        "DELETE": ("delete", "remove", "cancel", "close", "archive"),
    }.get(suffix, (verb,))
    for tool in offered:
        name = str(tool.get("name", "")).casefold()
        if any(word in name for word in synonyms):
            return str(tool.get("qualified_name") or "") or None
    return None


def generate_intents_by_rule(
    domain_name: str,
    count: int,
    examples_per_intent: int,
    existing: list[str],
    offered_mcp_tools: list[dict] | None = None,
) -> dict:
    """Deterministic intent drafts covering the common operations of a domain."""
    noun = _singular(domain_name)
    prefix = re.sub(r"[^A-Z0-9]+", "_", domain_name.strip().upper()).strip("_") or "DOMAIN"
    taken = {name.casefold() for name in existing}

    intents = []
    for suffix, verb, templates in _OPERATION_TEMPLATES:
        if len(intents) >= count:
            break
        name = f"{prefix}_{suffix}"
        if name.casefold() in taken:
            continue
        intents.append(
            {
                "name": name,
                "description": f"{suffix.casefold().capitalize()} operation for {noun} records.",
                "tool": f"{verb}_{noun}s"
                if verb in {"search", "get_expiring"}
                else f"{verb}_{noun}",
                "entity_schema": _SCHEMA_BY_OPERATION.get(suffix, {}),
                "extraction_hints": "",
                "mcp_tool": _pick_mcp_tool(offered_mcp_tools or [], verb, noun, suffix) or "",
                "examples": [t.format(d=noun) for t in templates[:examples_per_intent]],
            }
        )
    return {"intents": intents}


def generate_examples_by_rule(
    domain_name: str, intent_name: str, count: int, existing: list[str]
) -> dict:
    """More phrasings for one intent, matched to its operation suffix."""
    noun = _singular(domain_name)
    suffix = intent_name.rsplit("_", 1)[-1].upper() if "_" in intent_name else intent_name.upper()
    templates = next(
        (t for s, _, t in _OPERATION_TEMPLATES if s == suffix), _OPERATION_TEMPLATES[0][2]
    )
    seen = {" ".join(e.split()).casefold() for e in existing}
    produced = [t.format(d=noun) for t in templates]
    fresh = [t for t in produced if t.casefold() not in seen]
    # Fall back to qualified variants when the obvious phrasings are used up.
    if len(fresh) < count:
        fresh += [
            f"{t.format(d=noun)} please"
            for t in templates
            if f"{t.format(d=noun)} please".casefold() not in seen
        ]
    return {"examples": fresh[:count]}
