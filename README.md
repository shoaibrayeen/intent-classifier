# intent-classifier

A domain-aware intent classification engine. It maps free-form natural language
to configured business intents using hybrid retrieval, reports a calibrated
confidence, and says `UNKNOWN` rather than guessing when nothing fits.

```
POST /api/v1/classify
{ "domain": "contract", "text": "Show me all active contracts with Microsoft" }

{ "intent": "CONTRACT_SEARCH",
  "confidence": 0.80,
  "entities": { "counterparty": "Microsoft", "status": "ACTIVE" },
  "tool": { "name": "search_contracts", "version": "v1",
            "arguments": { "counterparty": "Microsoft", "status": "ACTIVE" },
            "ready": true },
  "latency_ms": 11.0 }
```

Adding an intent means adding example phrasings. There is no model training
step, and no fine-tuning when the catalogue changes.

Open [docs/architecture.html](docs/architecture.html) for the design,
[docs/api-documentation.html](docs/api-documentation.html) for the API
reference, and [docs/changelog.html](docs/changelog.html) for what changed and
why. All open
straight from the repository; the running app also serves Swagger UI at `/docs`.

## The problem

Users express one intent many different ways:

> "Find contracts with Microsoft" · "Show me Microsoft agreements" ·
> "Which contracts do we have with Microsoft?" · "Search for Microsoft's contracts"

Keyword rules become unmaintainable as intents multiply, and an LLM asked to
pick an intent is hard to debug, hard to evaluate, and hard to keep stable. This
service treats the task as *retrieval*: every intent owns a set of example
utterances, a query is matched against that corpus by two independent
retrievers, and the decision is traceable at every stage.

## How a query is classified

```
query
  │
  ├─ normalize (NFKC, casefold, collapse whitespace)
  │
  ├──────────────────────────────┬──────────────────────────────┐
  ▼                              ▼                              │
dense retrieval                sparse retrieval                 │
FastEmbed → ChromaDB           BM25 (rank-bm25)                 │
top-k examples by cosine       top-k examples by term overlap   │
  │                              │                              │
  └──────────────┬───────────────┘                              │
                 ▼                                              │
       Reciprocal Rank Fusion  (1/(k+rank), k=60)               │
                 ▼                                              │
       group examples by intent, sum the best 3 per intent      │
                 ▼                                              │
       confidence scoring ──────────────────────────────────────┘
                 ▼
       intent  or  UNKNOWN + reason
```

Dense retrieval catches paraphrase ("agreements" ≈ "contracts"). BM25 catches
exact terminology a vector model blurs (identifiers, names, rare terms). Fusing
by *rank* avoids combining two incomparable score scales.

### Confidence is computed, not borrowed

The fused score is a ranking quantity, not a probability, so it is never
returned as confidence. Four bounded signals are combined instead:

| Signal | What it measures |
|---|---|
| `s_dense` | cosine similarity of the best matching example, rescaled between a floor and ceiling |
| `s_rrf` | aggregated fusion score against its theoretical maximum |
| `s_margin` | how far the winning intent leads the runner-up |
| `s_support` | how many retrieved examples back the winner |

`confidence = 0.35·s_dense + 0.25·s_rrf + 0.25·s_margin + 0.15·s_support`

BM25's raw score is deliberately excluded: its scale depends on corpus size and
document length, so it is not comparable across domains. Its evidence already
enters through the fusion ranking.

### When the answer is UNKNOWN

A result is `UNKNOWN` when confidence falls below `CONFIDENCE_THRESHOLD`, or
when the closest example is below `MIN_DENSE_SIMILARITY` regardless of how the
lexical retriever scored. The response says which:

```json
{ "intent": "UNKNOWN", "confidence": 0.40, "reason": "low_confidence",
  "top_intents": [ ... ] }
```

This matters most for agentic callers, where a confident wrong answer triggers
the wrong tool.

## Entities and tools

Classification alone is not enough to call anything. Once an intent resolves,
the engine extracts the values that intent declares and reports the call it maps
to.

```
query -> intent -> entity schema -> extraction -> validated arguments -> tool call
```

**Extraction runs only after a confident match**, never for `UNKNOWN`. Pulling
arguments out of a request the system did not understand produces confident
nonsense, which is worse than no answer. It is also never on the critical path:
if the provider times out or returns junk, the classification still returns, with
`entities` empty and `entity_extraction.status` saying why.

Whatever the model returns is treated as untrusted input. Each value is coerced
to the type the intent declared, enum values are normalized to the declared
spelling, dates become ISO 8601, and **any key the intent did not declare is
dropped**. The schema is an allowlist, not a suggestion.

```json
{ "counterparty": {"type": "string", "required": true},
  "status":       {"type": "enum", "values": ["ACTIVE", "EXPIRED"]},
  "signed_after": {"type": "date"} }
```

Supported types: `string`, `integer`, `number`, `boolean`, `date`, `enum`,
`array`. A required entity that was not found leaves the tool call marked
`ready: false` with the missing names listed, rather than silently calling a
tool with a hole in its arguments.

Extraction is off by default. Set `OPENAI_API_KEY` (or `LLM_PROVIDER=mock` to
run offline), then either turn it on globally with
`ENTITY_EXTRACTION_ENABLED=true` or per request with `"extract_entities": true`.

### Instructions can be drafted for you

An administrator knows their domain, not prompt engineering. Give a domain a
name and a one-line description, and the service drafts both instruction blocks
with the configured LLM provider and saves them:

```bash
curl -X POST localhost:8000/api/v1/domains \
  -H 'content-type: application/json' \
  -d '{"name": "contract",
       "description": "tracking vendor agreements, renewals and obligations",
       "generate_instructions": true}'
```

Or regenerate later, optionally steering with a different brief:

```bash
curl -X POST localhost:8000/api/v1/domains/$DOMAIN/instructions/generate \
  -d '{"brief": "comparing procurement deals across regions"}' \
  -H 'content-type: application/json'
```

The draft is saved to the domain and returned for review. The domain page has
the same flow: a generate button fills the editor, and the admin refines and
saves by hand. Explicit instructions always win over generation, a missing
provider fails before anything is created, and a provider outage never destroys
a created domain. `LLM_PROVIDER=mock` drafts deterministic templates offline.

### Intents: auto mode or by hand

Intents work the same two ways as instructions. Write them yourself, or let the
LLM draft the catalogue:

```bash
curl -X POST localhost:8000/api/v1/domains/$DOMAIN/intents/generate \
  -H 'content-type: application/json' \
  -d '{"count": 5, "examples_per_intent": 8}'
```

Each proposed intent arrives complete: name, description, tool mapping, entity
schema, and **training examples**. The examples matter, because an intent with
none can never be retrieved, so generating intents without them would produce a
catalogue that classifies nothing. With them, the domain is usable immediately.

Auto mode is a drafting aid, not a separate kind of object. Generated intents
land in the same collections as hand-written ones and are edited, deleted and
re-tooled through the same endpoints and screens.

- `"dry_run": true` returns the proposal without writing anything.
- Names that already exist in the domain are **skipped, never overwritten**, so
  running it twice extends the catalogue rather than duplicating it.
- Model output is untrusted: names are coerced to the pattern the intent model
  enforces, entity schemas are filtered to declared types, examples are
  deduplicated against what the intent already has, and anything unusable is
  reported in `skipped` rather than saved.

The companion, for an intent you wrote by hand:

```bash
curl -X POST localhost:8000/api/v1/domains/$DOMAIN/intents/$INTENT/examples/generate \
  -H 'content-type: application/json' -d '{"count": 8}'
```

The domain page carries both forms side by side, and the intent page has a
generate button next to manual example entry. `LLM_PROVIDER=mock` drafts
deterministic intents offline; it names records after the domain, so a real
provider produces better domain vocabulary.

### Instructions are per domain

The extraction prompt is assembled in layers, so each domain speaks its own
language without the code changing:

| Layer | Set where | Goes into |
|---|---|---|
| base rules (never invent keys, ISO dates, resolve "it" from history) | code | system prompt |
| `system_instructions` | the domain | system prompt |
| `extraction_hints` | the intent | system prompt |
| `user_instructions` | the domain | user turn, before the request |
| conversation history | the session | user turn |

**System instructions open with the role the domain implies**, because how the
assistant should act is the first thing the extractor needs to know. A
contractual domain gets "You are acting as a contracts and legal analyst"; one
about carts and checkout gets "You are acting as an e-commerce shopping
assistant". The same sentence reads differently in each. After the role come the
domain's vocabulary and what must never be inferred: "a counterparty is the
other party, never our own company", or "'me' and 'my' are the person asking,
not an employee name". Both blocks are edited from the domain page or the API.

### Running without a key

`LLM_PROVIDER=mock` swaps in an offline, rule-based extractor that reads the
same prompt a real model would receive and fills the schema by pattern: enum
values by name, dates and years, identifiers like `C-1042`, and proper nouns
for strings. It is deliberately simple and deterministic. Its job is to make the
*whole* flow runnable and testable end to end with no key: instructions,
history, validation, carry-over and tool routing all execute for real. Health
reports which provider is live.

## MCP tools

An intent can be wired to a Model Context Protocol tool. Register a server's
catalogue by pasting its `tools/list` response, and every classification of a
bound intent reports the exact call that would satisfy it:

```bash
curl -X POST localhost:8000/api/v1/mcp/tools/import \
  -H 'content-type: application/json' \
  -d '{"server": "contracts", "transport": "stdio",
       "endpoint": "npx -y @acme/contracts-mcp",
       "tools": {"tools": [{"name": "search_contracts",
                            "inputSchema": {"type": "object",
                                            "properties": {"counterparty": {"type": "string"}},
                                            "required": ["counterparty"]}}]}}'
```

A classification then carries the resolved call:

```json
"mcp": {
  "qualified_name": "mcp__contracts__search_contracts",
  "server": "contracts", "tool": "search_contracts",
  "transport": "stdio", "endpoint": "npx -y @acme/contracts-mcp",
  "arguments": {"counterparty": "Microsoft"},
  "missing_required": [], "unmapped": [], "ready": true
}
```

The extracted entities are matched against the tool's own JSON Schema, not the
intent's. Anything the tool does not declare is reported as `unmapped` rather
than sent; a missing required argument leaves the call `ready: false`. Delete a
bound tool and classifications say `unresolved` instead of silently losing the
call.

Auto mode binds as it generates: the registry is offered to the model, and a
proposed tool is accepted only if it is actually registered. The import accepts
the `tools/list` result, its bare `tools` array, or a JSON-RPC envelope, and
re-importing updates in place.

### Nothing is executed, and that is a seam not a wall

This service resolves and reports the call. It does not dial the MCP server.
Wiring up real invocation means implementing one protocol in
`app/services/mcp_executor.py` and passing it to `create_app`; the resolved
call already carries everything an invocation needs. The default implementation
refuses every call, so a caller adding one cannot be surprised by this service
having already run something.

### Tool routing stops before execution

The response says what should be called and with which arguments. It never calls
it. Execution belongs to the caller, who owns the credentials, the retry policy
and the blast radius. A classifier that also invokes tools cannot be safely
shared by a UI, an API and an agent at once.

## Multi-turn sessions

People do not ask one question. Pass a `session_id` on classify and turns in the
same session inform each other in two deliberate, inspectable ways:

```
turn 1  "Show me all contracts with Microsoft"   -> CONTRACT_SEARCH  {counterparty: Microsoft}
turn 2  "and for Oracle"                         -> alone: UNKNOWN
                                                    with context: CONTRACT_SEARCH {counterparty: Oracle}
turn 3  "which of them expire next month"        -> CONTRACT_EXPIRY {counterparty: Microsoft ← carried}
```

**Contextual retrieval.** A follow-up like "and for Oracle" has nothing to
retrieve on. If a turn comes back `UNKNOWN` and the session has a previous turn,
it is retried as the previous question plus the new one. The contextual result
is kept only if it actually resolves, so context can rescue a follow-up but can
never overrule a confident answer. The response says when this happened and what
text was used.

**Entity carry-over.** Values named earlier flow into a later intent that
declares the same entity, unless the new turn names its own. Only entities the
*new* intent accepts are eligible, so nothing leaks into a tool that has no use
for it. The response lists what was carried.

Two guards stop a rescue from absorbing a change of subject. The retry only runs
for text that reads as referential, a pronoun or a fragment too short to stand
alone, and its result must clear the threshold by `CONTEXT_RESCUE_MARGIN`.
Without them, "what is the weather today" asked after "summarize contract
C-1042" came back as `CONTRACT_SUMMARY`, because the previous question alone
matched the concatenation.

The extractor also sees the last few turns as `conversation_history`, so a real
model can resolve "it" and "they" itself.

Every turn records a compact snapshot of its own decision, so reopening a
conversation shows what actually happened then rather than a re-classification
against a catalogue that has since moved on.

Sessions live in the same Chroma store as everything else, in a `session_turns`
collection: one deployment is still one directory. Per-session size and a TTL
are enforced on write. `GET /api/v1/sessions/{id}` shows a conversation;
`DELETE` forgets it. `"use_context": false` treats a single turn as standalone
while still recording it. Sessions are isolated by id and by domain.

## Security

Authentication is off by default so the service runs out of the box. Turn it on
with `AUTH_ENABLED=true` and supply keys as `secret:domains:scopes`:

```
API_KEYS=ops-key:*:admin, contract-svc:contract|legal:classify, dashboard:*:read
```

Present a key as `X-API-Key` or `Authorization: Bearer`. Scopes nest:
`admin` > `write` > `read` > `classify`. The domain list is a hard boundary: a
key scoped to `contract` cannot classify against, read, or even see another
domain. `/health` needs no credential, because a probe that requires a secret is
a probe that stops working when the secret rotates.

Keys are compared in constant time and are never written to logs; the audit log
records a short fingerprint instead.

## Operating it

| Endpoint | What it gives you |
|---|---|
| `/api/v1/health` | store, model, and which features are actually on |
| `/api/v1/health/index` | per-domain index state, and whether the two indexes agree |
| `/api/v1/metrics` | Prometheus metrics |
| `/api/v1/audit` | recent audit entries (admin scope) |
| `/api/v1/strategies` | available retrieval variants |

`/ui/operations` shows all of it on one page.

**Request ids.** Every response carries `X-Request-ID`, echoing one you supply.
The same id appears in the classification body, the audit record and the trace,
so "this query returned the wrong intent" is traceable from one value.

**Metrics** are labelled by route *template*, never by concrete path. A label
containing a domain id would create a new time series per domain, which is how a
metrics backend falls over.

**The audit log** is append-only JSONL at `AUDIT_LOG_PATH`: one object per line
with the request id, principal, action, outcome, intent, confidence and latency.
Query text is **not** recorded unless you set `AUDIT_LOG_QUERY_TEXT=true`, since
user queries can carry personal data.

**Tracing** is opt-in. With no collector configured, an exporter would retry in
the background and add latency to every request. Set `TRACING_ENABLED=true` and
`OTLP_ENDPOINT` when you have one.

**Indexes are warmed at startup**, so a restart does not make the first
classification pay the build cost, and index health tells the truth immediately.

## A/B testing retrieval

Changing retrieval is the riskiest change this service can make, so variants are
named, explicit, and assigned deterministically by hashing the request id. The
same id always lands on the same pipeline, which makes a reported result
reproducible.

| Variant | What it does |
|---|---|
| `hybrid_rrf` | dense and BM25 fused. The default. |
| `dense_only` | semantic retrieval alone |
| `bm25_only` | lexical retrieval alone |
| `hybrid_rrf_k20` | smaller fusion constant, sharper top ranks |
| `hybrid_top1` | score each intent by its single best example |
| `hybrid_wide` | retrieve twice as many candidates |

Pin one per request with `"variant": "dense_only"`, or enable assignment with
`AB_TESTING_ENABLED=true`. Measured on the evaluation set:

| Strategy | Top-1 | UNKNOWN detection | p50 |
|---|---|---|---|
| hybrid_rrf | 90.6% | 90% | 3.4 ms |
| dense_only | 87.5% | 70% | 3.4 ms |
| bm25_only | 65.6% | 40% | 0.6 ms |
| hybrid_wide | 87.5% | 100% | 3.7 ms |

That is the argument for the hybrid default stated as a measurement: either
retriever alone is worse, and lexical-only is much worse at knowing when to
decline.

Single-retriever variants are scored against their own ceiling rather than
penalised for evidence they were never configured to collect.

## Documentation

Everything a reader opens lives in `docs/`, and the app serves it too.

| File | Route | What it is |
|---|---|---|
| `docs/architecture.html` | `/ui/architecture` | the design, the pipeline, and the decisions behind it |
| `docs/api-documentation.html` | `/ui/api` | the API reference, generated from the live schema |
| `docs/properties.html` | `/ui/properties` | every setting, its default, and what it does |
| `/ui/architecture` | the design note |
| `docs/changelog.html` | `/ui/changelog`, `/changelog` | what changed and why |
| `docs/demo.html` | `/ui/demo` | a recorded conversation, interactive |
| `docs/demo.gif` | `/ui/docs/demo.gif` | the same run as an animation |

The whole directory is served at `/ui/docs/`, because the documents link to each
other by filename. Serving them under one prefix keeps those links working in
the browser exactly as they do on disk. The short paths above redirect there.
| `README.md` | — | setup, configuration, and how to run it |

`README.md` stays at the repository root rather than moving into `docs/`:
`pyproject.toml` declares it as the project readme and the build backend
validates that it is there, so the package will not build without it.

The API reference is generated, never hand-edited:

```bash
uv run python -m scripts.build_api_docs          # regenerate
uv run python -m scripts.build_api_docs --check  # fail if stale
```

The configuration reference is generated the same way, from the `Settings`
model:

```bash
uv run python -m scripts.build_properties
uv run python -m scripts.build_properties --check
```

Tests run both checks, so neither page can drift from the code.

The demo is recorded the same way, against a running service:

```bash
LLM_PROVIDER=mock docker compose up -d
uv run --with pillow python -m scripts.build_demo
```

It clears its own sessions first, so a re-recording cannot inherit entities from
a previous run.

## Quickstart (Docker)

```bash
docker compose up --build
```

Then open <http://localhost:8000/>. That is the whole quickstart: compose sets
`SEED_ON_STARTUP=true`, so the demo catalogue (two domains, seven intents, 46
examples) loads on the first run, and only while the store is empty. Set it to
`false` in `docker-compose.yml` for a real deployment.

`.env` is optional and gitignored. Without one the defaults apply, and they run
everything except entity extraction. For that:

```bash
cp .env.example .env     # then set OPENAI_API_KEY
```

`.env.example` carries one line, because one line is all most deployments need.
Every other setting has a working default: see
[docs/properties.html](docs/properties.html) for the full list, or `/ui/properties`
on a running service.

**Do not run the seed script against a store a running server already holds:**

```bash
# wrong: this writes from a second process
docker compose exec intent-classifier python -m scripts.seed
```

Embedded Chroma is single-writer. The write succeeds, but the running server
keeps its own view of the store and fails to read the new vectors with
`Error creating hnsw segment reader: Nothing found on disk` until it is
restarted. Use `SEED_ON_STARTUP`, the REST API, or the UI to add intents to a
live service. If you do seed this way, run `docker compose restart` afterwards.

The embedding model is baked into the image, so the first classification does
not wait on a download and the container runs without network access. The
intent catalogue lives in `./data`, mounted as a volume, and survives rebuilds.

`OPENAI_API_KEY` is **not** required to classify. It is used only by entity
extraction, which is off by default.

## Local development

```bash
uv sync
uv run python -m scripts.seed
uv run uvicorn app.main:app --reload
```

Requires Python 3.12 (`uv python install 3.12`). The first run downloads the
embedding model (~67 MB) into `./data/models`.

## The UI

| Page | What it does |
|---|---|
| `/` | domains with intent and example counts, create and delete |
| `/ui/domains/{id}` | intents in a domain, tool mapping, index status, rebuild |
| `/ui/domains/{id}/intents/{id}` | entity schema, examples, MCP binding, generate more |
| `/ui/playground` | a chat: ask on the right, see the reasoning on the left |
| `/ui/intents` | every intent across every domain, and what it is wired to |
| `/ui/mcp` | the MCP tool registry; `/ui/mcp/{id}` for one tool |
| `/ui/sessions` | recent conversations; open one to reread and inspect it |
| `/ui/changelog` | what changed and why, also at `/changelog` |

Confidence is banded the same way everywhere it is shown: green above 80%,
amber 50 to 80%, red below. The playground puts a dot on every confidence
signal and every ranked intent, always beside the number rather than instead of
it. Static assets are served with a version stamp, so a stylesheet change
reaches a browser that already cached the old one.
| `/ui/evaluation` | run the held-out evaluation set against the live catalogue |
| `/ui/operations` | index health, configuration in force, recent activity |
| `/ui/api` | the generated API reference |
| `/ui/properties` | every setting, its default, and what it does |
| `/ui/architecture` | the design note |

The playground is the debugging tool: it shows the normalized query, the dense
hits with cosine similarities, the BM25 hits with tokens and scores, the fused
intent ranking, and the four confidence signals that produced the decision.

Server-rendered with Jinja2 and htmx. No build step, no Node, no CDN at runtime.

## API documentation

Four surfaces, all served by the running app:

| Where | What it is |
|---|---|
| `/ui/api` | single-page reference: narrative, every endpoint, every model |
| `/docs` | Swagger UI, interactive, try requests in the browser |
| `/redoc` | ReDoc rendering of the same schema |
| `/openapi.json` | the OpenAPI 3.1 schema, for client generation |

[docs/api-documentation.html](docs/api-documentation.html) is the same page as `/ui/api`
and opens straight from the repository. It is **generated from the live schema**,
so it cannot drift:

```bash
uv run python -m scripts.build_api_docs
```

A test fails if the checked-in file falls out of step with the code, so
regenerate it whenever an endpoint or model changes. `--check` does the same in
CI without writing.

## API

Base path `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/classify` | classify a query |
| `POST` | `/classify/debug` | same, with the full retrieval trace |
| `GET` | `/health` | store, model and configuration status |
| `GET` | `/health/index` | per-domain index health |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/audit` | recent audit entries |
| `GET` | `/strategies` | available retrieval variants |
| `POST` `GET` | `/domains` | create, list |
| `GET` `PUT` `DELETE` | `/domains/{domainId}` | read, update, delete (cascades) |
| `POST` `GET` | `/domains/{domainId}/intents` | create, list |
| `GET` `PUT` `DELETE` | `/domains/{domainId}/intents/{intentId}` | read, update, delete |
| `POST` `GET` | `/domains/{domainId}/intents/{intentId}/examples` | add, list |
| `POST` | `.../examples/bulk` | add many in one embedding pass |
| `DELETE` | `.../examples/{exampleId}` | remove one |
| `POST` `GET` | `/mcp/tools` | register or list MCP tools |
| `GET` `PUT` `DELETE` | `/mcp/tools/{toolId}` | one MCP tool |
| `POST` | `/mcp/tools/import` | load a server's `tools/list` catalogue |
| `POST` | `/domains/{domainId}/intents/generate` | draft intents with examples (auto mode) |
| `POST` | `/domains/{domainId}/intents/{intentId}/examples/generate` | draft more examples for one intent |
| `GET` `DELETE` | `/sessions/{sessionId}` | inspect or forget a conversation |
| `POST` | `/domains/{domainId}/reindex` | force a BM25 rebuild |
| `GET` | `/domains/{domainId}/index/status` | index state, version, document counts |

Interactive docs at `/docs`, the narrative reference at `/ui/api`, the raw
schema at `/openapi.json`.

`domain` in a classify request accepts either the domain id or its name.
Errors return `{"error": {"code": ..., "message": ...}}` with 404, 409 or 400.

### Creating a catalogue

```bash
curl -X POST localhost:8000/api/v1/domains \
  -H 'content-type: application/json' \
  -d '{"name": "contract", "description": "Contract operations"}'

curl -X POST localhost:8000/api/v1/domains/$DOMAIN/intents \
  -H 'content-type: application/json' \
  -d '{"name": "CONTRACT_SEARCH",
       "description": "Search contracts by criteria",
       "tool": {"name": "search_contracts", "version": "v1"},
       "entity_schema": {"counterparty": {"type": "string"}}}'

curl -X POST localhost:8000/api/v1/domains/$DOMAIN/intents/$INTENT/examples/bulk \
  -H 'content-type: application/json' \
  -d '[{"text": "Find all contracts with Microsoft"},
       {"text": "Show me Microsoft agreements"}]'
```

Both indexes are updated synchronously, so the new phrasing is classifiable on
the very next request.

## Domains

A domain is an isolated intent namespace. Retrieval never crosses a domain
boundary, so "show me active agreements" can mean different things in different
domains, and a large catalogue does not dilute a small one.

## Storage

ChromaDB is the system of record. Three collections:

| Collection | Holds |
|---|---|
| `domains` | domain configuration as JSON documents |
| `intents` | intent configuration, tool mapping, entity schema |
| `intent_examples` | example text plus its 384-dimensional embedding |
| `session_turns` | conversation turns: text, intent, entities, per session and domain |
| `mcp_tools` | the MCP registry: server, tool, input schema, transport, endpoint |

Uniqueness, cascading deletes and index invalidation are enforced in the service
layer, since Chroma has no constraints. All access runs through one client under
a single lock: embedded Chroma is a single-writer store, which is also why the
container runs one worker.

The BM25 index is derived state, held in memory per domain and rebuilt eagerly
whenever examples change. It is never the source of truth.

Repositories sit behind protocols, so a relational backend can replace Chroma
for configuration without touching the engine.

### Backup

```bash
uv run python -m scripts.seed --export catalogue.json
uv run python -m scripts.seed --reset --import catalogue.json
```

## Configuration

Every setting has a working default. The service starts, classifies and serves
its UI with **no `.env` at all**, so `.env` holds only what you want to differ
from the default.

Two values become mandatory once you enable the feature that needs them:

| Variable | Required when |
|---|---|
| `OPENAI_API_KEY` | `LLM_PROVIDER` resolves to `openai` |
| `API_KEYS` | `AUTH_ENABLED` is `true` |

[docs/properties.html](docs/properties.html), served at `/ui/properties`, lists
all 48 settings with their defaults, types and constraints. It is generated from
`app/config.py`, so it cannot go stale.

```bash
cp .env.example .env    # then set OPENAI_API_KEY if you want extraction
```

## Tests and evaluation

```bash
uv run pytest                              # unit, integration, evaluation
uv run python -m tests.evaluation.run_eval # accuracy report
uv run python -m tests.evaluation.run_eval --sweep    # calibrate thresholds
uv run python -m tests.evaluation.run_eval --compare  # compare retrieval strategies
uv run python -m tests.evaluation.run_eval --verbose  # every prediction
```

The evaluation set holds 42 queries that are **not** training examples,
including 10 out-of-domain queries that must return `UNKNOWN`. Current results
against the seeded catalogue:

| Metric | Result |
|---|---|
| Top-1 accuracy | 90.6% |
| Top-3 accuracy | 100% |
| UNKNOWN detection | 90% |
| Valid queries wrongly refused | 3.1% |

`--sweep` grid-searches both thresholds and prints the trade-off. Raising
`MIN_DENSE_SIMILARITY` to `0.70` reaches 100% UNKNOWN detection but wrongly
refuses about 9% of valid queries. Choose based on whether a wrong tool call
costs more than a refusal.

`tests/evaluation/test_evaluation.py` turns these numbers into a regression
floor, so a change that degrades retrieval fails the suite.

## Deliberate boundaries

- **Tool invocation stays out of scope.** A result reports the mapped tool; the
  caller decides whether to run it.
- **The LLM is never in the classification path.** Retrieval is deterministic,
  inspectable and evaluable. Generation is reserved for entity extraction, where
  it is genuinely the right tool.
- **The BM25 index is never persisted as truth.** Rebuilding from Chroma is
  fast, and a second on-disk copy drifts the moment the tokenizer changes.
- **One worker only.** Embedded Chroma is a single-writer store.

Chroma remains the configuration store. The repository layer sits behind
protocols, so moving configuration to PostgreSQL is a contained change if
multi-tenant authorization or heavy audit querying later justifies it.

## Troubleshooting

**First run is slow.** The embedding model is downloading into
`MODEL_CACHE_DIR`. The Docker image has it pre-baked.

**A catalogue change is not reflected.** Check `GET /domains/{id}/index/status`.
Mutations rebuild eagerly, so a `dirty` state means a mutation failed. `POST
/domains/{id}/reindex` forces a rebuild.

**Starting over.** Stop the app, delete `./data/chroma`, re-run the seed script.
Export first if the catalogue matters.

**Never run more than one worker.** Embedded Chroma is single-writer, and
concurrent writers can corrupt the on-disk store.

**Why chromadb is pinned to 1.5.2.** Versions 1.5.4 through at least 1.5.9
intermittently abort with `recursive_mutex lock failed` at process exit on
macOS ARM64 (upstream chroma-core/chroma#6852). 1.5.2 ran a six-for-six clean
stress of the full suite here and reads stores written by 1.5.9. Re-pin upward
once the upstream fix lands, and rerun the suite several times before trusting
it.

## Stack

FastAPI · Pydantic v2 · ChromaDB · FastEmbed (`bge-small-en-v1.5`) · rank-bm25 ·
OpenAI (entity extraction only) · Prometheus · OpenTelemetry · Jinja2 + htmx ·
uv · pytest
