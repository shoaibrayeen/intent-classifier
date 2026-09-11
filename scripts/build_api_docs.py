"""Generate api-documentation.html from the live OpenAPI schema.

The reference is generated rather than written by hand so it cannot drift from
the code: every endpoint, parameter and field comes from the app itself. The
narrative around it is written once, here.

    uv run python -m scripts.build_api_docs
    uv run python -m scripts.build_api_docs --check   # fail if out of date
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from app.config import Settings
from app.main import create_app

OUTPUT = Path(__file__).resolve().parents[1] / "api-documentation.html"

METHOD_ORDER = ["get", "post", "put", "patch", "delete"]


def esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def build_schema() -> dict[str, Any]:
    settings = Settings(chroma_mode="ephemeral", auth_enabled=False, audit_log_enabled=False)
    return create_app(settings).openapi()


# --------------------------------------------------------------------- schema
def resolve(ref: str, schema: dict[str, Any]) -> dict[str, Any]:
    node: Any = schema
    for part in ref.removeprefix("#/").split("/"):
        node = node.get(part, {})
    return node if isinstance(node, dict) else {}


def type_label(node: dict[str, Any], schema: dict[str, Any], depth: int = 0) -> str:
    if depth > 4 or not node:
        return "object"
    if "$ref" in node:
        return node["$ref"].rsplit("/", 1)[-1]
    if "anyOf" in node or "oneOf" in node:
        options = node.get("anyOf") or node.get("oneOf")
        parts = [type_label(o, schema, depth + 1) for o in options]
        parts = [p for p in parts if p != "null"]
        return " or ".join(dict.fromkeys(parts)) or "null"
    if node.get("enum"):
        return " | ".join(json.dumps(v) for v in node["enum"])
    declared = node.get("type")
    if declared == "array":
        return f"{type_label(node.get('items', {}), schema, depth + 1)}[]"
    if declared == "object" and node.get("additionalProperties"):
        return "object"
    return str(declared or "object")


def referenced_models(name: str, schema: dict[str, Any], seen: set[str] | None = None) -> set[str]:
    """Every model reachable from this one.

    A response model is useless without the models nested inside it, and a link
    to an undocumented one is a broken anchor.
    """
    seen = seen if seen is not None else set()
    if name in seen:
        return seen
    seen.add(name)
    node = schema.get("components", {}).get("schemas", {}).get(name, {})

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
                referenced_models(ref.rsplit("/", 1)[-1], schema, seen)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(node)
    return seen


def model_rows(name: str, schema: dict[str, Any]) -> list[tuple[str, str, bool, str]]:
    node = schema.get("components", {}).get("schemas", {}).get(name, {})
    required = set(node.get("required", []))
    rows = []
    for field, spec in (node.get("properties") or {}).items():
        rows.append(
            (
                field,
                type_label(spec, schema),
                field in required,
                spec.get("description", ""),
            )
        )
    return rows


def body_model(operation: dict[str, Any]) -> str | None:
    content = (operation.get("requestBody") or {}).get("content", {})
    node = (content.get("application/json") or {}).get("schema", {})
    if "$ref" in node:
        return node["$ref"].rsplit("/", 1)[-1]
    if node.get("type") == "array" and "$ref" in node.get("items", {}):
        return node["items"]["$ref"].rsplit("/", 1)[-1] + "[]"
    return None


def response_model(operation: dict[str, Any]) -> str | None:
    for code in ("200", "201"):
        content = (operation.get("responses", {}).get(code) or {}).get("content", {})
        node = (content.get("application/json") or {}).get("schema", {})
        if "$ref" in node:
            return node["$ref"].rsplit("/", 1)[-1]
        if node.get("type") == "array" and "$ref" in node.get("items", {}):
            return node["items"]["$ref"].rsplit("/", 1)[-1] + "[]"
    return None


# ----------------------------------------------------------------- rendering
def render_endpoints(schema: dict[str, Any]) -> tuple[str, list[str]]:
    grouped: dict[str, list[tuple[str, str, dict]]] = {}
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            if method.lower() not in METHOD_ORDER:
                continue
            tag = (operation.get("tags") or ["other"])[0]
            grouped.setdefault(tag, []).append((path, method.lower(), operation))

    order = [t["name"] for t in schema.get("tags", [])]
    order += [t for t in grouped if t not in order]

    parts: list[str] = []
    models: set[str] = set()
    for tag in order:
        entries = grouped.get(tag)
        if not entries:
            continue
        description = next(
            (t.get("description", "") for t in schema.get("tags", []) if t["name"] == tag), ""
        )
        parts.append(f'<h3 id="tag-{esc(tag)}">{esc(tag)}</h3>')
        if description:
            parts.append(f"<p class='muted'>{esc(description)}</p>")
        entries.sort(key=lambda e: (e[0], METHOD_ORDER.index(e[1])))
        for path, method, operation in entries:
            body = body_model(operation)
            response = response_model(operation)
            for candidate in (body, response):
                if candidate:
                    models |= referenced_models(candidate.removesuffix("[]"), schema)
            parts.append(
                f'<div class="endpoint"><p class="sig">'
                f'<span class="method {method}">{method.upper()}</span> '
                f'<span class="mono">{esc(path)}</span></p>'
            )
            summary = operation.get("summary") or ""
            doc = (operation.get("description") or "").strip()
            if summary:
                parts.append(f"<p><strong>{esc(summary)}</strong></p>")
            if doc and doc != summary:
                parts.append(f"<p class='muted'>{esc(doc)}</p>")
            params = [p for p in operation.get("parameters", []) if p.get("in") == "query"]
            if params:
                rows = "".join(
                    f"<tr><td class='mono'>{esc(p['name'])}</td>"
                    f"<td class='mono small'>{esc(type_label(p.get('schema', {}), schema))}</td>"
                    f"<td class='muted small'>{esc(p.get('description', ''))}</td></tr>"
                    for p in params
                )
                parts.append(
                    "<p class='muted small'>Query parameters</p>"
                    f"<table><tbody>{rows}</tbody></table>"
                )
            meta = []
            if body:
                meta.append(
                    f"body <a class='mono' href='#model-{esc(body.removesuffix('[]'))}'>"
                    f"{esc(body)}</a>"
                )
            if response:
                meta.append(
                    f"returns <a class='mono' href='#model-"
                    f"{esc(response.removesuffix('[]'))}'>{esc(response)}</a>"
                )
            if meta:
                parts.append(f"<p class='small'>{' · '.join(meta)}</p>")
            parts.append("</div>")
    return "\n".join(parts), sorted(models)


def render_models(schema: dict[str, Any], names: list[str]) -> str:
    parts: list[str] = []
    for name in names:
        rows = model_rows(name, schema)
        if not rows:
            continue
        body = "".join(
            f"<tr><td class='mono'>{esc(field)}</td>"
            f"<td class='mono small'>{esc(kind)}</td>"
            f"<td class='small'>{'yes' if required else ''}</td>"
            f"<td class='muted small'>{esc(note)}</td></tr>"
            for field, kind, required, note in rows
        )
        parts.append(
            f'<div class="model" id="model-{esc(name)}"><h3>{esc(name)}</h3>'
            "<table><thead><tr><th>Field</th><th>Type</th><th>Required</th>"
            f"<th>Notes</th></tr></thead><tbody>{body}</tbody></table></div>"
        )
    return "\n".join(parts)


NARRATIVE = """
<h2 id="start">Start here</h2>
<p>One endpoint does the work. Everything else configures it or explains it.</p>

<pre><code>curl -X POST http://localhost:8000/api/v1/classify \\
  -H 'content-type: application/json' \\
  -d '{"domain": "contract", "text": "Show me all active contracts with Microsoft"}'</code></pre>

<pre><code>{
  "domain": "contract",
  "intent": "CONTRACT_SEARCH",
  "confidence": 0.88,
  "entities": { "counterparty": "Microsoft", "status": "ACTIVE" },
  "tool": {
    "name": "search_contracts",
    "version": "v1",
    "arguments": { "counterparty": "Microsoft", "status": "ACTIVE" },
    "missing_required": [],
    "ready": true
  },
  "reason": null,
  "top_intents": [ ... ],
  "latency_ms": 11.0,
  "request_id": "5f3c..."
}</code></pre>

<div class="note">
  <strong>Read <code>intent</code> together with <code>reason</code>.</strong>
  When nothing matches well enough the intent is <code>UNKNOWN</code> and
  <code>reason</code> says which rule fired: <code>low_confidence</code>,
  <code>low_similarity</code> or <code>no_examples</code>. That is a real answer,
  not an error. For an agent, a confident wrong intent invokes the wrong tool.
</div>

<h2 id="conversations">Conversations</h2>
<p>Pass a <code>session_id</code> and turns inform each other. The same id across calls is one
conversation; a different id is a different one. Sessions are also scoped per domain.</p>

<pre><code>POST /api/v1/classify  {"domain":"contract","text":"Show me all active contracts with Microsoft","session_id":"s-1"}
  -> CONTRACT_SEARCH   entities {counterparty: Microsoft, status: ACTIVE}

POST /api/v1/classify  {"domain":"contract","text":"and for Oracle","session_id":"s-1"}
  -> CONTRACT_SEARCH   entities {counterparty: Oracle, status: ACTIVE}
     context.used_for_retrieval = true
     context.retrieval_text = "Show me all active contracts with Microsoft and for Oracle"
     context.carried_entities = ["status"]</code></pre>

<p>Two mechanisms, both reported in <code>context</code> so nothing is hidden:</p>
<ul>
  <li><strong>Contextual retrieval.</strong> A turn that comes back <code>UNKNOWN</code> is
  retried as the previous question plus the new one, and the contextual result is kept only if
  it resolves. Context can rescue a follow-up; it can never overrule a confident answer.</li>
  <li><strong>Entity carry-over.</strong> A value named earlier fills an entity the new turn did
  not mention, but only where the new intent declares that entity.</li>
</ul>
<p>Send <code>"use_context": false</code> to treat one turn as standalone while still recording
it. <code>GET /api/v1/sessions/{id}</code> shows a conversation, <code>DELETE</code> forgets it.</p>

<h2 id="entities">Entities and tools</h2>
<p>Extraction runs only after a confident match, never for <code>UNKNOWN</code>, and is the only
place this service calls a language model. Whatever the model returns is treated as untrusted
input: values are coerced to the type the intent declared, enums are normalised to the declared
spelling, dates become ISO 8601, and <strong>any key the intent did not declare is dropped</strong>.
Rejections are listed in <code>entity_extraction.rejected</code>.</p>

<p>A required entity that was not found leaves the tool call <code>ready: false</code> with the
missing names listed, rather than calling a tool with a hole in its arguments. The service
reports the call and never performs it.</p>

<p>Turn extraction on globally with <code>ENTITY_EXTRACTION_ENABLED=true</code>, or per request
with <code>"extract_entities": true</code>. Set <code>LLM_PROVIDER=mock</code> to run the whole
flow offline with a rule-based extractor and no API key.</p>

<h2 id="errors">Errors</h2>
<p>Failures return a single shape, carrying the request id so a report can be traced:</p>
<pre><code>{"error": {"code": "not_found", "message": "domain 'ghost' not found", "request_id": "5f3c..."}}</code></pre>
<table>
  <thead><tr><th>Status</th><th>Code</th><th>When</th></tr></thead>
  <tbody>
    <tr><td>400</td><td class="mono">invalid_input</td><td>malformed input the schema cannot reject, such as an entity schema that is not JSON</td></tr>
    <tr><td>401</td><td class="mono">unauthenticated</td><td>authentication is on and the key is missing or wrong</td></tr>
    <tr><td>403</td><td class="mono">forbidden</td><td>the key lacks the scope, or the domain is outside its allowlist</td></tr>
    <tr><td>404</td><td class="mono">not_found</td><td>no such domain, intent, example or session</td></tr>
    <tr><td>409</td><td class="mono">conflict</td><td>duplicate domain name, intent name within a domain, or example text within an intent</td></tr>
    <tr><td>422</td><td class="mono">-</td><td>request body failed validation (FastAPI's own shape)</td></tr>
  </tbody>
</table>

<h2 id="auth">Authentication</h2>
<p>Off by default. Enable with <code>AUTH_ENABLED=true</code> and define keys as
<code>secret:domains:scopes</code>, comma separated:</p>
<pre><code>API_KEYS=ops-key:*:admin, contract-svc:contract|legal:classify, dashboard:*:read</code></pre>
<p>Send the key as <code>X-API-Key</code> or <code>Authorization: Bearer</code>. Scopes nest:
<code>admin</code> &gt; <code>write</code> &gt; <code>read</code> &gt; <code>classify</code>. The
domain list is a hard boundary, not a filter: a key scoped to one domain cannot classify against,
read, or even enumerate another. <code>/health</code> needs no credential, because a probe that
requires a secret stops working when the secret rotates.</p>
"""


def render(schema: dict[str, Any]) -> str:
    endpoints, model_names = render_endpoints(schema)
    models = render_models(schema, model_names)
    version = schema["info"]["version"]
    style = STYLE
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intent Classifier — API reference</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Intent Classifier API</h1>
  <p class="lede">Map free-form language to configured business intents, extract the entities
  that intent declares, and report the tool it maps to.</p>
  <p class="small muted">Version {esc(version)} · generated from the live OpenAPI schema by
  <span class="mono">scripts/build_api_docs.py</span></p>
  <nav class="links">
    <a href="/docs">Swagger UI</a>
    <a href="/redoc">ReDoc</a>
    <a href="/openapi.json">openapi.json</a>
    <a href="/ui/playground">Playground</a>
  </nav>
</header>

<nav class="toc">
  <a href="#start">Start here</a>
  <a href="#conversations">Conversations</a>
  <a href="#entities">Entities and tools</a>
  <a href="#errors">Errors</a>
  <a href="#auth">Authentication</a>
  <a href="#endpoints">Endpoints</a>
  <a href="#models">Models</a>
</nav>

{NARRATIVE}

<h2 id="endpoints">Endpoints</h2>
{endpoints}

<h2 id="models">Models</h2>
<p class="muted">Field names, types and requiredness come straight from the schema.</p>
{models}

</div>
</body>
</html>
"""


STYLE = """
:root {
  color-scheme: light dark;
  --bg:#faf9f7; --panel:#fff; --ink:#1a1917; --muted:#6b6a66; --line:#e3e1dc;
  --accent:#2f6f4f; --accent-soft:#e8f1eb; --warn:#8a6d1f;
  --get:#2f6f4f; --post:#2f5f8a; --put:#8a6d1f; --delete:#9c2f2f;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#16171a; --panel:#1e2023; --ink:#eceae6; --muted:#9b9a95; --line:#32343a;
    --accent:#6fbf94; --accent-soft:#223029; --warn:#d6b45c;
    --get:#6fbf94; --post:#7fb0dd; --put:#d6b45c; --delete:#e0806f; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.wrap { max-width:920px; margin:0 auto; padding:3rem 1.5rem 5rem; }
header { border-bottom:1px solid var(--line); padding-bottom:1.5rem; margin-bottom:1.5rem; }
h1 { font-size:2rem; margin:0 0 .4rem; letter-spacing:-.025em; }
h2 { font-size:1.3rem; margin:3rem 0 .75rem; padding-top:1.5rem;
  border-top:1px solid var(--line); letter-spacing:-.015em; }
h3 { font-size:1.05rem; margin:2rem 0 .5rem; }
p { margin:.85rem 0; }
.lede { font-size:1.1rem; color:var(--muted); margin:0; }
.muted { color:var(--muted); } .small { font-size:.85rem; }
a { color:var(--accent); }
code, .mono { font-family:var(--mono); font-size:.88em; }
code { background:var(--accent-soft); padding:.1em .35em; border-radius:4px; }
pre { background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:1rem 1.1rem; overflow-x:auto; font-family:var(--mono); font-size:13px;
  line-height:1.5; margin:1rem 0; }
pre code { background:none; padding:0; }
table { width:100%; border-collapse:collapse; font-size:14px; margin:1rem 0; }
th,td { text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line);
  vertical-align:top; }
th { font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
tbody tr:last-child td { border-bottom:none; }
.links { margin-top:1rem; display:flex; gap:1rem; flex-wrap:wrap; }
.links a { font-size:14px; padding:.25rem .7rem; border:1px solid var(--line);
  border-radius:999px; text-decoration:none; }
.toc { display:flex; gap:1rem; flex-wrap:wrap; font-size:14px;
  padding:.75rem 0; border-bottom:1px solid var(--line); }
.toc a { text-decoration:none; }
.endpoint, .model { background:var(--panel); border:1px solid var(--line);
  border-radius:10px; padding:.9rem 1.1rem; margin:.85rem 0; }
.endpoint .sig { margin:0 0 .4rem; }
.method { font-family:var(--mono); font-size:12px; font-weight:700;
  padding:.1rem .45rem; border-radius:5px; border:1px solid currentColor; }
.method.get{color:var(--get);} .method.post{color:var(--post);}
.method.put{color:var(--put);} .method.delete{color:var(--delete);}
.note { border-left:3px solid var(--accent); background:var(--accent-soft);
  padding:.85rem 1.1rem; border-radius:0 8px 8px 0; margin:1.5rem 0; }
ul { padding-left:1.25rem; } li { margin:.35rem 0; }
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the API reference page")
    parser.add_argument("--check", action="store_true", help="fail if the file is out of date")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    rendered = render(build_schema())

    if args.check:
        if not args.output.exists() or args.output.read_text("utf-8") != rendered:
            print(f"{args.output.name} is out of date: run scripts/build_api_docs.py")
            return 1
        print(f"{args.output.name} is up to date")
        return 0

    args.output.write_text(rendered, "utf-8")
    print(f"wrote {args.output} ({len(rendered):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
