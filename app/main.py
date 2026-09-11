"""FastAPI application factory."""

from __future__ import annotations

import html
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import Settings, get_settings
from app.errors import AppError
from app.services.container import build_container
from app.ui.routes import router as ui_router
from app.ui.templating import templates

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.container = build_container(settings)
        if not settings.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. Classification does not need it; "
                "entity extraction stays disabled until it is provided."
            )
        logger.info("%s %s ready", settings.app_name, settings.app_version)
        yield

    app = FastAPI(
        title="Intent Classifier",
        version=settings.app_version,
        description=(
            "Domain-aware intent classification using hybrid dense + sparse "
            "retrieval fused with Reciprocal Rank Fusion."
        ),
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
    app.include_router(api_router)
    app.include_router(ui_router)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        if request.url.path.startswith("/api"):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"code": exc.code, "message": exc.message}},
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
