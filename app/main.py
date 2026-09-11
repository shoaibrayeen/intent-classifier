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
from app.ui.routes import router as ui_router
from app.ui.templating import templates

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


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


def create_app(settings: Settings | None = None, llm_client: LLMClient | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        tracing.setup(settings)
        app.state.container = build_container(settings, llm_client=llm_client)
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
        logger.info("%s %s ready", settings.app_name, settings.app_version)
        yield
        tracing.shutdown()

    app = FastAPI(
        title="Intent Classifier",
        version=settings.app_version,
        description=(
            "Domain-aware intent classification using hybrid dense + sparse "
            "retrieval fused with Reciprocal Rank Fusion, with entity "
            "extraction and tool routing."
        ),
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
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
