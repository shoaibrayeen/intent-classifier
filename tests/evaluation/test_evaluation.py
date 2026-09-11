"""Regression guard on retrieval quality.

The thresholds here are floors, not targets. If a change drops accuracy below
them, the evaluation report explains which queries moved:

    uv run python -m tests.evaluation.run_eval --verbose
"""

from __future__ import annotations

import asyncio

import pytest

from app.config import Settings
from app.services.container import build_container
from app.services.evaluation import load_cases
from scripts.seed import seed
from tests.evaluation.run_eval import predict, score

MIN_TOP1 = 0.85
MIN_TOP3 = 0.95
MIN_UNKNOWN_DETECTION = 0.80
MAX_FALSE_UNKNOWN = 0.10


@pytest.fixture(scope="module")
def metrics():
    settings = Settings(
        chroma_mode="ephemeral",
        auth_enabled=False,
        audit_log_enabled=False,
        entity_extraction_enabled=False,
    )
    container = build_container(settings)
    container.store.reset_all()
    container.index_manager.clear()
    seed(container)
    predictions = asyncio.run(predict(container, load_cases(), None))
    return score(predictions, settings)


def test_top1_accuracy(metrics):
    assert metrics["top1"] >= MIN_TOP1


def test_top3_accuracy(metrics):
    assert metrics["top3"] >= MIN_TOP3


def test_unknown_detection(metrics):
    assert metrics["unknown_recall"] >= MIN_UNKNOWN_DETECTION


def test_valid_queries_are_not_rejected(metrics):
    assert metrics["false_unknown"] <= MAX_FALSE_UNKNOWN
