# intent-classifier

A domain-aware intent classification engine. It maps free-form natural language
to configured business intents using hybrid retrieval, reports a calibrated
confidence, and says `UNKNOWN` rather than guessing when nothing fits.

```
POST /api/v1/classify
{ "domain": "contract", "text": "Show me all contracts with Microsoft expiring this year" }

{ "intent": "CONTRACT_SEARCH", "confidence": 0.71,
  "tool": { "name": "search_contracts", "version": "v1" }, "entities": {} }
```

Adding an intent means adding example phrasings. There is no model training
step, and no fine-tuning when the catalogue changes.

Open [architecture.html](architecture.html) in a browser for the full design.

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

## Quickstart (Docker)

```bash
cp .env.example .env     # then set OPENAI_API_KEY if you want entity extraction
docker compose up --build
```

Seed the demo catalogue (two domains, seven intents, 46 examples):

```bash
docker compose exec intent-classifier python -m scripts.seed
```

Then open <http://localhost:8000/>.

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
| `/ui/domains/{id}/intents/{id}` | entity schema, training examples, add and delete |
| `/ui/playground` | classify a query and see every retrieval stage |

The playground is the debugging tool: it shows the normalized query, the dense
hits with cosine similarities, the BM25 hits with tokens and scores, the fused
intent ranking, and the four confidence signals that produced the decision.

Server-rendered with Jinja2 and htmx. No build step, no Node, no CDN at runtime.

## API

Base path `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/classify` | classify a query |
| `POST` | `/classify/debug` | same, with the full retrieval trace |
| `GET` | `/health` | store, model and configuration status |
| `POST` `GET` | `/domains` | create, list |
| `GET` `PUT` `DELETE` | `/domains/{domainId}` | read, update, delete (cascades) |
| `POST` `GET` | `/domains/{domainId}/intents` | create, list |
| `GET` `PUT` `DELETE` | `/domains/{domainId}/intents/{intentId}` | read, update, delete |
| `POST` `GET` | `/domains/{domainId}/intents/{intentId}/examples` | add, list |
| `POST` | `.../examples/bulk` | add many in one embedding pass |
| `DELETE` | `.../examples/{exampleId}` | remove one |
| `POST` | `/domains/{domainId}/reindex` | force a BM25 rebuild |
| `GET` | `/domains/{domainId}/index/status` | index state, version, document counts |

Interactive docs at `/docs`.

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

Every value has a working default in `.env`.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | only needed for entity extraction |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | override for a compatible endpoint |
| `OPENAI_MODEL` | `gpt-4o-mini` | model used for extraction |
| `ENTITY_EXTRACTION_ENABLED` | `false` | turn on the LLM extraction step |
| `CHROMA_MODE` | `persistent` | `ephemeral` for tests |
| `CHROMA_PATH` | `./data/chroma` | store location |
| `MODEL_NAME` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `MODEL_CACHE_DIR` | `./data/models` | where the model is cached |
| `RETRIEVAL_TOP_K` | `10` | examples fetched per retriever |
| `RRF_K` | `60` | fusion constant |
| `AGG_TOP_N` | `3` | examples summed per intent |
| `CONFIDENCE_THRESHOLD` | `0.55` | below this, `UNKNOWN` |
| `MIN_DENSE_SIMILARITY` | `0.60` | hard similarity floor |
| `DENSE_SIM_FLOOR` / `DENSE_SIM_CEIL` | `0.50` / `0.90` | rescaling range for `s_dense` |
| `LOG_LEVEL` | `INFO` | |

## Tests and evaluation

```bash
uv run pytest                              # unit, integration, evaluation
uv run python -m tests.evaluation.run_eval # accuracy report
uv run python -m tests.evaluation.run_eval --sweep    # calibrate thresholds
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

## What this phase does not do

- **Entity extraction** is wired but disabled. Intents already carry an
  `entity_schema`, and `entities` is always `{}` for now.
- **Tool invocation** is out of scope by design. A result reports the mapped
  tool; the caller decides whether to run it. Keeping the classifier free of
  side effects is what makes it reusable by an API, a UI and an agent alike.

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

## Stack

FastAPI · Pydantic v2 · ChromaDB · FastEmbed (`bge-small-en-v1.5`) · rank-bm25 ·
Jinja2 + htmx · uv · pytest
