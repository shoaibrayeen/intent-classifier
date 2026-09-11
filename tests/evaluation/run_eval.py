"""Offline evaluation of the classification engine.

Runs the held-out dataset against a freshly seeded in-memory catalogue and
reports accuracy, UNKNOWN handling, and (with --sweep) how the two decision
thresholds trade off against each other.

    uv run python -m tests.evaluation.run_eval
    uv run python -m tests.evaluation.run_eval --sweep
    uv run python -m tests.evaluation.run_eval --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.models.classification import UNKNOWN_INTENT
from app.services.container import Container, build_container
from scripts.seed import seed

DATASET = Path(__file__).parent / "dataset.json"


@dataclass
class Prediction:
    domain: str
    query: str
    expected: str
    predicted: str
    confidence: float
    best_similarity: float
    ranked: list[str]


def load_cases() -> list[dict]:
    return json.loads(DATASET.read_text("utf-8"))["cases"]


def predict(container: Container, cases: list[dict]) -> list[Prediction]:
    predictions: list[Prediction] = []
    for case in cases:
        domain = container.domains.resolve(case["domain"])
        result = container.classifier.classify(domain, case["query"], debug=True)
        predictions.append(
            Prediction(
                domain=case["domain"],
                query=case["query"],
                expected=case["expected"],
                predicted=result.intent,
                confidence=result.confidence,
                best_similarity=result.debug.confidence_breakdown.best_similarity,
                ranked=[item.intent for item in result.debug.rrf_intents],
            )
        )
    return predictions


def score(predictions: list[Prediction], settings: Settings) -> dict:
    """Re-apply the decision rule at the given thresholds, without re-retrieving."""
    in_domain = [p for p in predictions if p.expected != UNKNOWN_INTENT]
    out_domain = [p for p in predictions if p.expected == UNKNOWN_INTENT]

    def decide(p: Prediction) -> str:
        if not p.ranked:
            return UNKNOWN_INTENT
        if p.best_similarity < settings.min_dense_similarity:
            return UNKNOWN_INTENT
        if p.confidence < settings.confidence_threshold:
            return UNKNOWN_INTENT
        return p.ranked[0]

    top1 = sum(1 for p in in_domain if decide(p) == p.expected)
    top3 = sum(1 for p in in_domain if p.expected in p.ranked[:3])
    retrieval_top1 = sum(1 for p in in_domain if p.ranked and p.ranked[0] == p.expected)
    unknown_correct = sum(1 for p in out_domain if decide(p) == UNKNOWN_INTENT)
    false_unknown = sum(1 for p in in_domain if decide(p) == UNKNOWN_INTENT)

    return {
        "in_domain": len(in_domain),
        "out_domain": len(out_domain),
        "top1": top1 / len(in_domain) if in_domain else 0.0,
        "top3": top3 / len(in_domain) if in_domain else 0.0,
        "retrieval_top1": retrieval_top1 / len(in_domain) if in_domain else 0.0,
        "unknown_recall": unknown_correct / len(out_domain) if out_domain else 0.0,
        "false_unknown": false_unknown / len(in_domain) if in_domain else 0.0,
        "decide": decide,
    }


def print_report(predictions: list[Prediction], settings: Settings, verbose: bool) -> dict:
    metrics = score(predictions, settings)
    decide = metrics["decide"]

    print("\n=== Intent classification evaluation ===")
    print(f"confidence_threshold  {settings.confidence_threshold}")
    print(f"min_dense_similarity  {settings.min_dense_similarity}")
    print(
        f"cases                 {len(predictions)} "
        f"({metrics['in_domain']} in-domain, {metrics['out_domain']} out-of-domain)"
    )
    print()
    print(f"top-1 accuracy          {metrics['top1']:.1%}")
    print(f"top-3 accuracy          {metrics['top3']:.1%}")
    print(f"retrieval top-1         {metrics['retrieval_top1']:.1%}  (before thresholds)")
    print(f"UNKNOWN detection       {metrics['unknown_recall']:.1%}")
    print(f"false UNKNOWN rate      {metrics['false_unknown']:.1%}")

    wrong = [p for p in predictions if decide(p) != p.expected]
    if wrong:
        print(f"\n--- {len(wrong)} incorrect ---")
        for p in wrong:
            got = decide(p)
            print(f"  [{p.domain}] {p.query}")
            print(
                f"      expected {p.expected}, got {got} "
                f"(conf {p.confidence:.2f}, sim {p.best_similarity:.3f})"
            )

    if verbose:
        print("\n--- all predictions ---")
        for p in predictions:
            mark = "ok " if decide(p) == p.expected else "MISS"
            print(
                f"  {mark} {decide(p):18s} conf={p.confidence:.2f} sim={p.best_similarity:.3f}"
                f"  {p.query}"
            )
    return metrics


def sweep(predictions: list[Prediction], base: Settings) -> None:
    print("\n=== Threshold sweep ===")
    print("Picks the pair with the best balanced accuracy across both groups.\n")
    header = f"{'conf':>6} {'sim':>6} {'top1':>8} {'unknown':>9} {'false-unk':>10} {'balanced':>9}"
    print(header)
    print("-" * len(header))

    best = None
    for conf_threshold in [round(0.35 + 0.05 * i, 2) for i in range(9)]:
        for sim_floor in [round(0.40 + 0.05 * i, 2) for i in range(9)]:
            candidate = base.model_copy(
                update={
                    "confidence_threshold": conf_threshold,
                    "min_dense_similarity": sim_floor,
                }
            )
            metrics = score(predictions, candidate)
            balanced = (metrics["top1"] + metrics["unknown_recall"]) / 2
            row = (conf_threshold, sim_floor, metrics, balanced)
            if best is None or balanced > best[3]:
                best = row
            if balanced >= 0.90:
                print(
                    f"{conf_threshold:6.2f} {sim_floor:6.2f} {metrics['top1']:7.1%} "
                    f"{metrics['unknown_recall']:8.1%} {metrics['false_unknown']:9.1%} "
                    f"{balanced:8.1%}"
                )

    if best:
        conf_threshold, sim_floor, metrics, balanced = best
        print(
            f"\nbest: CONFIDENCE_THRESHOLD={conf_threshold} MIN_DENSE_SIMILARITY={sim_floor} "
            f"(balanced {balanced:.1%}, top-1 {metrics['top1']:.1%}, "
            f"UNKNOWN {metrics['unknown_recall']:.1%})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the intent classifier")
    parser.add_argument("--sweep", action="store_true", help="grid-search the two thresholds")
    parser.add_argument("--verbose", action="store_true", help="print every prediction")
    parser.add_argument("--min-top1", type=float, default=0.0, help="fail below this top-1")
    parser.add_argument(
        "--min-unknown", type=float, default=0.0, help="fail below this UNKNOWN detection rate"
    )
    args = parser.parse_args(argv)

    settings = Settings(chroma_mode="ephemeral")
    container = build_container(settings)
    print("seeding an in-memory catalogue ...")
    stats = seed(container)
    print(f"  {stats['domains']} domains, {stats['intents']} intents, {stats['examples']} examples")

    predictions = predict(container, load_cases())
    metrics = print_report(predictions, settings, args.verbose)

    if args.sweep:
        sweep(predictions, settings)

    if metrics["top1"] < args.min_top1 or metrics["unknown_recall"] < args.min_unknown:
        print("\nFAILED: below the requested thresholds")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
