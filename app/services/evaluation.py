"""Evaluation against a held-out dataset.

The same scoring the command-line harness uses, callable from the UI so a
threshold or catalogue change can be judged immediately instead of on trust.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models.classification import UNKNOWN_INTENT, ClassifyRequest

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from app.services.container import Container

logger = logging.getLogger(__name__)

#: Ships inside the package. It lived under tests/ once, which meant the
#: evaluation page ran zero cases in Docker: tests/ is not copied into the
#: image, so the feature silently had nothing to evaluate.
DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"


@dataclass
class CaseResult:
    domain: str
    query: str
    expected: str
    predicted: str
    confidence: float
    best_similarity: float
    ranked: list[str]
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected

    @property
    def in_top3(self) -> bool:
        return self.expected in self.ranked[:3]


@dataclass
class EvaluationReport:
    strategy: str
    cases: list[CaseResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Set when the run could not happen at all, rather than happening and
    #: matching nothing. An empty report that looks like a pass is worse than
    #: no report.
    problem: str | None = None
    dataset: str = ""
    total_cases: int = 0

    @property
    def in_domain(self) -> list[CaseResult]:
        return [c for c in self.cases if c.expected != UNKNOWN_INTENT]

    @property
    def out_domain(self) -> list[CaseResult]:
        return [c for c in self.cases if c.expected == UNKNOWN_INTENT]

    @property
    def incorrect(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.correct]

    def _ratio(self, count: int, total: int) -> float:
        return count / total if total else 0.0

    @property
    def top1(self) -> float:
        return self._ratio(sum(1 for c in self.in_domain if c.correct), len(self.in_domain))

    @property
    def top3(self) -> float:
        return self._ratio(sum(1 for c in self.in_domain if c.in_top3), len(self.in_domain))

    @property
    def unknown_detection(self) -> float:
        hits = sum(1 for c in self.out_domain if c.predicted == UNKNOWN_INTENT)
        return self._ratio(hits, len(self.out_domain))

    @property
    def false_unknown(self) -> float:
        misses = sum(1 for c in self.in_domain if c.predicted == UNKNOWN_INTENT)
        return self._ratio(misses, len(self.in_domain))

    @property
    def per_intent(self) -> list[dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for case in self.in_domain:
            row = rows.setdefault(
                case.expected, {"intent": case.expected, "total": 0, "correct": 0}
            )
            row["total"] += 1
            row["correct"] += int(case.correct)
        for row in rows.values():
            row["accuracy"] = row["correct"] / row["total"] if row["total"] else 0.0
        return sorted(rows.values(), key=lambda r: (r["accuracy"], r["intent"]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "cases": len(self.cases),
            "in_domain": len(self.in_domain),
            "out_domain": len(self.out_domain),
            "top1": self.top1,
            "top3": self.top3,
            "unknown_detection": self.unknown_detection,
            "false_unknown": self.false_unknown,
            "skipped": self.skipped,
            "problem": self.problem,
            "dataset": self.dataset,
        }


def dataset_path(settings=None) -> Path:
    """Where the evaluation cases come from: the setting, else the packaged set."""
    configured = getattr(settings, "evaluation_dataset_path", "") if settings else ""
    return Path(configured) if configured else DATASET_PATH


def load_cases(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or DATASET_PATH
    if not target.exists():
        logger.warning("no evaluation dataset at %s", target)
        return []
    try:
        return json.loads(target.read_text("utf-8")).get("cases", [])
    except (OSError, json.JSONDecodeError):
        logger.exception("could not read the evaluation dataset at %s", target)
        return []


async def run_evaluation(
    container: Container, variant: str | None = None, path: Path | None = None
) -> EvaluationReport:
    """Classify every case against the live catalogue."""
    strategy_name = variant or container.settings.default_strategy
    target = path or dataset_path(container.settings)
    cases = load_cases(target)
    report = EvaluationReport(strategy=strategy_name, dataset=str(target), total_cases=len(cases))
    if not cases:
        report.problem = (
            f"No evaluation cases found at {target}. Point "
            "EVALUATION_DATASET_PATH at a JSON file with a 'cases' array."
        )
        return report

    known = {domain.name.casefold() for domain in container.domains.list()}
    if not known:
        report.problem = "No domains are configured, so there is nothing to evaluate against."
        return report

    for case in cases:
        if case["domain"].casefold() not in known:
            note = f"domain '{case['domain']}' is not configured"
            if note not in report.skipped:
                report.skipped.append(note)
            continue
        payload = ClassifyRequest(
            domain=case["domain"],
            text=case["query"],
            extract_entities=False,  # evaluation measures retrieval, not the LLM
            variant=variant,
        )
        try:
            result = await container.classification.classify(payload, debug=True)
        except Exception as exc:
            logger.exception("evaluation case failed")
            report.cases.append(
                CaseResult(
                    domain=case["domain"],
                    query=case["query"],
                    expected=case["expected"],
                    predicted="ERROR",
                    confidence=0.0,
                    best_similarity=0.0,
                    ranked=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        breakdown = result.debug.confidence_breakdown if result.debug else None
        report.cases.append(
            CaseResult(
                domain=case["domain"],
                query=case["query"],
                expected=case["expected"],
                predicted=result.intent,
                confidence=result.confidence,
                best_similarity=breakdown.best_similarity if breakdown else 0.0,
                ranked=[item.intent for item in result.top_intents],
            )
        )

    if not report.cases:
        report.problem = (
            "Every case was skipped: none of the dataset's domains are configured here. "
            f"It expects {', '.join(sorted({c['domain'] for c in cases}))}."
        )
    return report
