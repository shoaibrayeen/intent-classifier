# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first so edits to the source do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Bake the embedding model into the image (~67 MB) so the first classification
# is not a cold download and the container can run without network access.
# It lives outside /app/data on purpose: that path is a bind mount at runtime
# and would shadow anything baked underneath it.
#
# This sits above the source copy so that editing code or docs does not force
# the download to run again on every rebuild.
ENV MODEL_CACHE_DIR=/opt/models
RUN python -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"

# README.md is required here, not optional: pyproject declares it as the
# project readme, and the build backend validates that the file exists when
# the project itself is installed by the sync below.
COPY README.md ./README.md
COPY app ./app
COPY scripts ./scripts
# Documentation: the generated API reference is served at /ui/api and the
# changelog at /ui/changelog, so both ship inside the image.
COPY docs ./docs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status==200 else 1)"

# One worker only: embedded Chroma is a single-writer store.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
