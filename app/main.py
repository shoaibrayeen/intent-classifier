"""FastAPI application factory."""

from __future__ import annotations

import html
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import Settings, get_settings
from app.errors import AppError
from app.observability import metrics, tracing
from app.observability.context import (
    REQUEST_ID_HEADER,
    get_request_id,
    new_request_id,
    set_principal,
    set_request_id,
)
from app.services.container import build_container
from app.services.llm.client import LLMClient
from app.services.mcp_executor import McpExecutor
from app.ui.routes import router as ui_router
from app.ui.templating import templates

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

API_DESCRIPTION = """
Domain-aware intent classification using hybrid retrieval: dense vectors from
FastEmbed over ChromaDB, BM25 for exact terminology, fused with Reciprocal Rank
Fusion. Adding an intent means adding example phrasings, not retraining a model.

### Where to start

* `POST /api/v1/classify` is the endpoint almost every caller needs.
* `POST /api/v1/classify/debug` returns the same decision with the full
  retrieval trace: every dense hit and its similarity, every BM25 hit and its
  score, the fused ranking, the confidence signals and per-stage timings.

### Things worth knowing before you integrate

* **`UNKNOWN` is a real answer.** When nothing matches well enough the response
  is `UNKNOWN` with a `reason`, rather than the closest guess. For an agent,
  a confident wrong intent invokes the wrong tool.
* **The tool is reported, never called.** A result names the tool and its
  validated arguments; running it is the caller's decision.
* **Entity extraction is the only LLM call,** it runs only after a confident
  match, and it never fails a classification: on a provider outage `entities`
  is empty and `entity_extraction.status` says why.
* **Pass a `session_id`** to make turns in a conversation inform each other.
  A follow-up that cannot stand alone is retried against the previous question,
  and entities named earlier carry into an intent that accepts them.
* **Authentication is off by default.** When enabled, send `X-API-Key` or
  `Authorization: Bearer`. Scopes nest: `admin` > `write` > `read` > `classify`.
* Every response carries `X-Request-ID`, echoing one you supply.

### Other formats

This schema is served at `/openapi.json`. There is a ReDoc rendering at
`/redoc`, and a single-page narrative reference at `/ui/api`.
"""

OPENAPI_TAGS = [
    {
        "name": "classify",
        "description": (
            "Classify free text against one domain. `/classify/debug` adds the "
            "retrieval trace that the playground renders."
        ),
    },
    {
        "name": "sessions",
        "description": (
            "Conversations. A session is created implicitly by the first "
            "classify call that names it; these endpoints inspect or forget one."
        ),
    },
    {
        "name": "domains",
        "description": (
            "A domain is an isolated intent namespace. Retrieval never crosses "
            "a domain boundary, so the same phrase can mean different things in "
            "different domains. Domains also carry the extraction instructions "
            "applied to every intent inside them."
        ),
    },
    {
        "name": "intents",
        "description": (
            "Intents within a domain: description, entity schema and tool "
            "mapping. Renaming one updates its examples."
        ),
    },
    {
        "name": "examples",
        "description": (
            "The retrieval corpus. Adding an example re-embeds it and rebuilds "
            "the domain's BM25 index before the request returns, so a new "
            "phrasing is classifiable immediately."
        ),
    },
    {
        "name": "index",
        "description": "Force a rebuild, or read index state and version.",
    },
    {
        "name": "operations",
        "description": (
            "Health, index health, Prometheus metrics, the audit trail and the "
            "available retrieval strategies."
        ),
    },
]


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _seed_if_empty(container) -> None:
    """Load the demo catalogue, but only into an empty store.

    Runs in the server's own process on purpose. Embedded Chroma is
    single-writer: seeding from a second process against a store the server
    already holds open leaves the server unable to read the new vectors until
    it is restarted.
    """
    # Everything here is best-effort, including the emptiness check: loading a
    # demo catalogue is a convenience and must never stop the service starting.
    try:
        if container.domains.list():
            logger.info("SEED_ON_STARTUP is on but the catalogue is not empty; leaving it alone")
            return

        from scripts.seed import seed

        stats = seed(container)
        logger.info(
            "seeded %d domain(s), %d intent(s), %d example(s)",
            stats["domains"],
            stats["intents"],
            stats["examples"],
        )
    except Exception:
        logger.exception("startup seeding failed; continuing with an empty catalogue")


def _route_template(request: Request) -> str:
    """Label metrics by route template, never by the concrete path.

    A label containing a domain id would create a new time series per domain,
    which is how a metrics backend falls over. The template is rebuilt by
    substituting each matched path parameter back into the URL, which does not
    depend on how the routers happen to be nested.
    """
    if request.scope.get("route") is None:
        return "unmatched"
    path = request.url.path
    for name, value in (request.scope.get("path_params") or {}).items():
        if value:
            path = path.replace(str(value), "{" + name + "}")
    return path


def create_app(
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    mcp_executor: McpExecutor | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        tracing.setup(settings)
        app.state.container = build_container(
            settings, llm_client=llm_client, mcp_executor=mcp_executor
        )
        if settings.auth_enabled:
            logger.info(
                "authentication is on with %d key(s)", app.state.container.authenticator.key_count
            )
        else:
            logger.warning("authentication is OFF; every caller has full access")
        extractor = app.state.container.entity_extractor
        if settings.entity_extraction_enabled and not extractor.available:
            logger.warning(
                "ENTITY_EXTRACTION_ENABLED is true but no LLM provider is configured; "
                "classifications will return no entities"
            )
        container = app.state.container
        if settings.seed_on_startup:
            _seed_if_empty(container)
        warmed = container.index_manager.warm([d.id for d in container.domains.list()])
        logger.info("warmed %d domain index(es)", warmed)
        logger.info("%s %s ready", settings.app_name, settings.app_version)
        yield
        tracing.shutdown()

    app = FastAPI(
        title="Intent Classifier",
        version=settings.app_version,
        summary=(
            "Map free-form language to configured business intents, extract the "
            "entities that intent declares, and report the tool it maps to."
        ),
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    # The documents link to each other by filename, which is how they read from
    # disk. Serving the directory under one prefix keeps those links working in
    # the browser too: architecture.html from /ui/docs/changelog.html resolves
    # to /ui/docs/architecture.html rather than a 404 beside the UI routes.
    docs_dir = BASE_DIR.parent / "docs"
    if docs_dir.is_dir():
        app.mount("/ui/docs", StaticFiles(directory=docs_dir), name="docs")
    else:
        logger.warning("docs/ is missing; the reference pages will not be served")
    app.include_router(api_router)
    app.include_router(ui_router)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_request_id()
        set_request_id(request_id)
        set_principal("anonymous")
        started = time.perf_counter()

        with tracing.span("http.request", method=request.method, path=request.url.path):
            response = await call_next(request)

        elapsed = time.perf_counter() - started
        if settings.metrics_enabled:
            template = _route_template(request)
            metrics.http_requests.labels(
                method=request.method, path=template, status=str(response.status_code)
            ).inc()
            metrics.http_latency.labels(method=request.method, path=template).observe(elapsed)

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["Server-Timing"] = f"app;dur={elapsed * 1000:.1f}"
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        if request.url.path.startswith("/api"):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "code": exc.code,
                        "message": exc.message,
                        "request_id": get_request_id(),
                    }
                },
            )
        if request.headers.get("hx-request"):
            # htmx does not swap 4xx bodies by default, so an inline failure is
            # reported as an out-of-band flash on an otherwise unchanged page.
            message = html.escape(exc.message)
            return HTMLResponse(
                f'<div id="flash" class="flash error" hx-swap-oob="true">{message}</div>',
                status_code=200,
            )
        return templates.TemplateResponse(
            request=request,
            name="pages/error.html",
            context={"message": exc.message, "status_code": exc.status_code},
            status_code=exc.status_code,
        )

    return app


app = create_app()
