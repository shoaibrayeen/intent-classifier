"""Build the hiring-platform demo end to end against a running service.

Creates the domain, drafts its instructions, registers an ATS server's MCP
tools, creates every intent with its examples, binds each intent to the tool it
would call, then records a conversation. Everything is written through the REST
API, so the data lands in Chroma and can be inspected in the UI afterwards.

    uv run --with pillow python -m scripts.demos.build_hiring_demo

Use LLM_PROVIDER=mock so the run costs nothing and is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from scripts.demos.hiring_catalogue import CONVERSATION, DOMAIN, INTENTS, MCP_SERVER

DOCS = Path(__file__).resolve().parents[2] / "docs"
SESSION = "hiring-demo"


def call(base: str, method: str, path: str, body: Any = None) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        return {"_error": exc.code, "_detail": exc.read().decode()[:300]}


def build(base: str, reset: bool) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    def record(stage: str, headline: str, detail: str, payload: Any = None) -> None:
        steps.append({"stage": stage, "headline": headline, "detail": detail, "payload": payload})
        print(f"  {stage:22s} {headline}")

    # --- a clean slate for this demo only ---------------------------------
    existing = call(base, "GET", "/api/v1/domains")
    prior = next((d for d in existing if d["name"] == DOMAIN["name"]), None)
    if prior and reset:
        call(base, "DELETE", f"/api/v1/domains/{prior['id']}")
        prior = None
    call(base, "DELETE", f"/api/v1/sessions/{SESSION}")

    # --- 1. the domain, with its instructions drafted ----------------------
    if prior:
        domain = call(base, "GET", f"/api/v1/domains/{prior['id']}")
    else:
        domain = call(
            base,
            "POST",
            "/api/v1/domains",
            {**DOMAIN, "generate_instructions": True},
        )
    record(
        "domain",
        f"created '{domain['name']}'",
        "The name and a one-line brief are all an administrator supplies. The role and "
        "vocabulary below were drafted from them and saved with the domain.",
        {
            "id": domain["id"],
            "description": domain["description"],
            "system_instructions": domain["system_instructions"],
            "user_instructions": domain["user_instructions"],
        },
    )

    # --- 2. the ATS server's tools ----------------------------------------
    imported = call(base, "POST", "/api/v1/mcp/tools/import", MCP_SERVER)
    tools = {t["name"]: t for t in call(base, "GET", "/api/v1/mcp/tools")}
    record(
        "mcp",
        f"registered {len(imported.get('created', [])) + len(imported.get('updated', []))} ATS tools",
        "Imported from the server's tools/list response. Each tool's own JSON Schema is what "
        "extracted entities are later matched against.",
        {
            "server": MCP_SERVER["server"],
            "endpoint": MCP_SERVER["endpoint"],
            "tools": [
                {
                    "qualified_name": tools[t["name"]]["qualified_name"],
                    "description": t["description"],
                    "arguments": list((t.get("inputSchema") or {}).get("properties", {})),
                    "required": (t.get("inputSchema") or {}).get("required", []),
                }
                for t in MCP_SERVER["tools"]["tools"]
                if t["name"] in tools
            ],
        },
    )

    # --- 3. intents, examples, bindings ------------------------------------
    created = []
    for spec in INTENTS:
        tool = tools.get(spec["tool"])
        intent = call(
            base,
            "POST",
            f"/api/v1/domains/{domain['id']}/intents",
            {
                "name": spec["name"],
                "description": spec["description"],
                "tool": {
                    "name": spec["tool"],
                    "mcp_tool_id": tool["id"] if tool else None,
                },
                "entity_schema": spec["entity_schema"],
                "extraction_hints": spec.get("extraction_hints", ""),
            },
        )
        if "_error" in intent:
            print(f"    ! {spec['name']}: {intent['_detail'][:90]}")
            continue
        call(
            base,
            "POST",
            f"/api/v1/domains/{domain['id']}/intents/{intent['id']}/examples/bulk",
            [{"text": text} for text in spec["examples"]],
        )
        created.append(
            {
                "name": spec["name"],
                "description": spec["description"],
                "tool": spec["tool"],
                "mcp": tool["qualified_name"] if tool else None,
                "entity_schema": spec["entity_schema"],
                "examples": spec["examples"],
            }
        )
    listed = call(base, "GET", f"/api/v1/domains/{domain['id']}/intents")
    record(
        "intents",
        f"{len(created)} intents, {sum(len(i['examples']) for i in created)} examples",
        "Each intent declares what it accepts and which ATS tool serves it. The examples are "
        "the whole training signal: no model was fine-tuned.",
        {"intents": created, "example_counts": {i["name"]: i["example_count"] for i in listed}},
    )

    index = call(base, "GET", "/api/v1/health/index")
    row = next((d for d in index["domains"] if d["domain_id"] == domain["id"]), {})
    record(
        "index",
        f"indexed {row.get('bm25_doc_count', 0)} examples",
        "Both indexes are built as the examples land: dense vectors in Chroma and a BM25 index "
        "over the same text. They must agree, and the operations page says when they do not.",
        row,
    )

    # --- 4. the conversation ------------------------------------------------
    turns = []
    for text, note in CONVERSATION:
        result = call(
            base,
            "POST",
            "/api/v1/classify/debug",
            {"domain": domain["name"], "text": text, "session_id": SESSION},
        )
        debug = result.get("debug") or {}
        mcp = (result.get("tool") or {}).get("mcp") or {}
        turns.append(
            {
                "query": text,
                "note": note,
                "intent": result["intent"],
                "confidence": result["confidence"],
                "reason": result.get("reason"),
                "strategy": result.get("strategy"),
                "latency_ms": result.get("latency_ms"),
                "entities": result.get("entities", {}),
                "tool": result.get("tool"),
                "mcp": mcp,
                "context": result.get("context"),
                "breakdown": debug.get("confidence_breakdown"),
                "top_intents": result.get("top_intents", [])[:4],
                "dense_hits": [
                    {"text": h["text"], "intent": h["intent_name"], "score": h["similarity"]}
                    for h in debug.get("dense_hits", [])[:4]
                ],
                "bm25_hits": [
                    {"text": h["text"], "intent": h["intent_name"], "score": h["score"]}
                    for h in debug.get("bm25_hits", [])[:4]
                ],
            }
        )
        marker = "ctx" if (result.get("context") or {}).get("used_for_retrieval") else "   "
        print(f"    {result['intent']:22s} {result['confidence']:.2f} {marker}  {text}")

    record(
        "conversation",
        f"{len(turns)} turns in session '{SESSION}'",
        "Turns share a session id, so a follow-up is read against what came before and entities "
        "carry forward. Every turn stores its own decision, which is what makes it inspectable "
        "later.",
        {"session_id": SESSION, "turns": turns},
    )

    session = call(base, "GET", f"/api/v1/sessions/{SESSION}")
    record(
        "session",
        f"{len(session.get('turns', []))} turns recorded",
        "Reopened from Chroma rather than recomputed. Opening this session in the playground "
        "shows exactly what happened, not a fresh classification.",
        {
            "last_intent": session.get("last_intent"),
            "entities_in_play": session.get("entities_in_play", {}),
            "turns": [
                {
                    "turn": t["turn"],
                    "text": t["text"],
                    "intent": t["intent"],
                    "confidence": t["confidence"],
                    "entities": t["entities"],
                }
                for t in session.get("turns", [])
            ],
        },
    )

    return {
        "domain": domain,
        "steps": steps,
        "turns": turns,
        "health": call(base, "GET", "/api/v1/health"),
        "index": index,
        "intent_count": len(created),
        "example_count": sum(len(i["examples"]) for i in created),
        "mcp_tools": call(base, "GET", "/api/v1/mcp/tools"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the hiring demo")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--keep", action="store_true", help="reuse an existing hiring domain")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    try:
        call(base, "GET", "/api/v1/health")
    except urllib.error.URLError as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        return 1

    print(f"building the hiring demo against {base}")
    run = build(base, reset=not args.keep)
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "demo-hiring-run.json").write_text(json.dumps(run, indent=1), "utf-8")
    print(f"\nwrote {DOCS / 'demo-hiring-run.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
