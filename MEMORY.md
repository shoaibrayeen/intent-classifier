# Memory

Durable knowledge about this service: the things that are expensive to
rediscover, and the reasons behind decisions that look arbitrary from the code
alone. The [README](README.md) says what the system does;
[docs/architecture.html](docs/architecture.html) says how. This file says what
we learned the hard way.

Keep it short. A fact belongs here only if it would otherwise be relearned by
breaking something.

---

## What this is, in one sentence

A domain-aware intent classifier: hybrid dense and lexical retrieval over
*example* utterances, fused with Reciprocal Rank Fusion, aggregated into
intents, scored for confidence, and mapped to a tool. Adding an intent means
adding examples, never retraining.

## Invariants

Break these and the design stops working.

- **One writer.** Embedded Chroma is single-writer. Run one worker
  (`--workers 1`). Never write to the store from a second process while the
  server holds it: the write succeeds and the server then fails to read it with
  `Error creating hnsw segment reader: Nothing found on disk` until restarted.
  This is why `SEED_ON_STARTUP` seeds in-process.
- **Nothing is executed.** The service resolves a tool call and reports it. The
  caller invokes it, because the caller owns the credentials and the blast
  radius. `app/services/mcp_executor.py` is the seam; its default refuses.
- **UNKNOWN is a feature, not a failure.** A wrong intent means calling the
  wrong tool. Any change that raises accuracy while lowering UNKNOWN detection
  is a regression, whatever the headline number says.
- **Classification never calls a language model.** Retrieval is deterministic,
  inspectable and evaluable. Generation is confined to entity extraction and to
  drafting configuration.
- **Model output is untrusted input.** Everything an LLM returns is coerced,
  filtered against a declared schema, and reported when dropped.

## Scars

Each of these is a bug that reached a running system. The guard exists because
of it; removing the guard brings the bug back.

| What happened | Why | The guard |
|---|---|---|
| "what is the weather today" classified as `CONTRACT_SUMMARY` at 0.66 | The contextual retry classifies the *previous question plus the new one*, so the previous question alone carried the match | A retry only runs for referential text, and must clear the threshold by `CONTEXT_RESCUE_MARGIN` |
| `auto` strategy raised accuracy but dropped UNKNOWN detection from 90% to 70% | Falling back accepted any alternate that merely cleared the threshold | `AUTO_RESCUE_MARGIN`, and fallbacks must use dense retrieval so the similarity floor still applies |
| A lexical-only pass matched off-domain text at 0.86 | BM25 cannot apply the minimum-similarity floor, and is normalised against its own lower ceiling | `bm25_only` is excluded from `auto` fallbacks, structurally, not just by list order |
| Every read of an intent failed after editing its tool | `model_copy` does not re-validate, so a nested `ToolRef` supplied as a dict was stored raw | `IntentService.update` re-validates through `model_validate` |
| The evaluation page silently did nothing in Docker | The dataset lived under `tests/`, which is not copied into the image | It ships at `app/evaluation/dataset.json`; an empty run now says so instead of rendering zeroes |
| Documentation links 404ed when served | The pages link to each other by filename, which only resolves as siblings | The whole `docs/` directory is served at `/ui/docs/`; short paths redirect there |
| A CSS change was invisible after deploy | The browser cached `app.css` | Static assets carry a version stamp from their modification time |
| The image would not build | `pyproject.toml` declares `README.md` as the project readme and the backend validates it | The Dockerfile copies it; `README.md` must stay at the repo root |

## Things that look wrong but are deliberate

- **`chromadb` is pinned to 1.5.2, not the latest.** 1.5.4 through at least
  1.5.9 intermittently abort with `recursive_mutex lock failed` at interpreter
  exit on macOS ARM64 (chroma-core/chroma#6852). Re-pin upward only after
  running the full suite several times.
- **`tests/conftest.py` calls `os._exit`.** Same crash, in the test process.
  It runs in `pytest_unconfigure`, after the summary is written, and is skipped
  under coverage.
- **The BM25 index is never persisted.** Rebuilding from Chroma is faster than
  unpickling, and a second on-disk copy drifts the moment the tokenizer changes.
- **The offline `mock` provider is crude on purpose.** It exists so the whole
  flow, including generation, runs and is tested with no API key. It names
  generated records after the domain, so a real provider produces better
  vocabulary.
- **Confidence is not the RRF score.** RRF magnitude is not comparable across
  queries. Confidence is computed separately from four normalised signals.

## Where the non-obvious things live

- `app/config.py` — every setting, self-documenting. `docs/properties.html` is
  generated from it, so add fields there and regenerate; never hand-edit.
- `app/evaluation/dataset.json` — the held-out set, inside the package because
  the running service reads it.
- `scripts/build_api_docs.py`, `scripts/build_properties.py`,
  `scripts/build_demo.py` — the three generated documents. Tests fail if the
  first two are stale.
- `app/services/turn_details.py` — the snapshot stored with each conversation
  turn, so reopening a session shows what happened *then* rather than a
  re-classification against a catalogue that has since moved on.

## Verifying a change

```bash
uv run pytest                                        # ~410 tests
uv run ruff check .
uv run python -m scripts.build_api_docs --check
uv run python -m scripts.build_properties --check
uv run python -m tests.evaluation.run_eval           # accuracy and UNKNOWN
uv run python -m tests.evaluation.run_eval --compare # before changing retrieval
docker compose up --build                            # the primary way to run it
```

Current baseline: top-1 90.6%, top-3 100%, UNKNOWN detection 90%, valid queries
refused 3.1%, on 42 held-out cases. Compare against this, not against feel.

**Never trust a passing container without recreating it.** `docker compose up`
reuses a container of the same name from an earlier project, so a rebuilt image
may not be what is running. `docker rm -f intent-classifier` first.

## Code map

Structural hubs, from `graphify-out/graph.json` (regenerate with
`graphify update .`):

`Settings` · `IntentConfig` · `DomainService` · `IndexManager` ·
`DomainConfig` · `Container` · `Principal` · `ChromaStore` · `IntentService`

`Container` is the only place collaborators are wired; repositories sit behind
`typing.Protocol`s in `app/repositories/base.py`, which is what would make a
move off Chroma contained.
