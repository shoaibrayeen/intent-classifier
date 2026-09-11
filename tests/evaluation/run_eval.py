"""Offline evaluation of the classification engine.

Seeds an in-memory catalogue, runs the held-out dataset through the real
service, and reports accuracy, UNKNOWN handling and latency. ``--sweep``
re-applies the decision rule at different thresholds against cached retrieval
results, so a grid search costs one pass rather than one pass per cell.

    uv run python -m tests.evaluation.run_eval
    uv run python -m tests.evaluation.run_eval --sweep
    uv run python -m tests.evaluation.run_eval --compare
    uv run python -m tests.evaluation.run_eval --strategy dense_only --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
from dataclasses import dataclass

from app.config import Settings
from app.models.classification import UNKNOWN_INTENT, ClassifyRequest
from app.services.container import Container, build_container
from app.services.evaluation import load_cases
from app.services.strategies import BUILTIN_STRATEGIES
from scripts.seed import seed


@dataclass
class Prediction:
    domain: str
    query: str
    expected: str
    predicted: str
    confidence: float
    best_similarity: float
    ranked: list[str]
    latency_ms: float


async def predict(
    container: Container, cases: list[dict], strategy: str | None
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for case in cases:
        result = await container.classification.classify(
            ClassifyRequest(
                domain=case["domain"],
                text=case["query"],
                extract_entities=False,
                variant=strategy,
            ),
            debug=True,
        )
        breakdown = result.debug.confidence_breakdown if result.debug else None
        predictions.append(
            Prediction(
                domain=case["domain"],
                query=case["query"],
                expected=case["expected"],
                predicted=result.intent,
                confidence=result.confidence,
                best_similarity=breakdown.best_similarity if breakdown else 0.0,
                ranked=[item.intent for item in result.top_intents],
                latency_ms=result.latency_ms,
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
    latencies = [p.latency_ms for p in predictions] or [0.0]

    return {
        "in_domain": len(in_domain),
        "out_domain": len(out_domain),
        "top1": top1 / len(in_domain) if in_domain else 0.0,
        "top3": top3 / len(in_domain) if in_domain else 0.0,
        "retrieval_top1": retrieval_top1 / len(in_domain) if in_domain else 0.0,
        "unknown_recall": unknown_correct / len(out_domain) if out_domain else 0.0,
        "false_unknown": false_unknown / len(in_domain) if in_domain else 0.0,
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "decide": decide,
    }


def print_report(
    predictions: list[Prediction], settings: Settings, strategy: str, verbose: bool
) -> dict:
    metrics = score(predictions, settings)
    decide = metrics["decide"]

    print("\n=== Intent classification evaluation ===")
    print(f"strategy              {strategy}")
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
    print(f"latency p50 / p95       {metrics['p50_ms']:.1f} ms / {metrics['p95_ms']:.1f} ms")

    wrong = [p for p in predictions if decide(p) != p.expected]
    if wrong:
        print(f"\n--- {len(wrong)} incorrect ---")
        for p in wrong:
            print(f"  [{p.domain}] {p.query}")
            print(
                f"      expected {p.expected}, got {decide(p)} "
                f"(conf {p.confidence:.2f}, sim {p.best_similarity:.3f})"
            )

    if verbose:
        print("\n--- all predictions ---")
        for p in predictions:
            mark = "ok " if decide(p) == p.expected else "MISS"
            print(
                f"  {mark} {decide(p):18s} conf={p.confidence:.2f} "
                f"sim={p.best_similarity:.3f}  {p.query}"
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
            if best is None or balanced > best[3]:
                best = (conf_threshold, sim_floor, metrics, balanced)
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


async def compare(container: Container, cases: list[dict], settings: Settings) -> None:
    print("\n=== Strategy comparison ===\n")
    header = (
        f"{'strategy':>16} {'top1':>8} {'top3':>8} {'unknown':>9} {'false-unk':>10} {'p50 ms':>8}"
    )
    print(header)
    print("-" * len(header))
    for name in BUILTIN_STRATEGIES:
        predictions = await predict(container, cases, name)
        metrics = score(predictions, settings)
        print(
            f"{name:>16} {metrics['top1']:7.1%} {metrics['top3']:7.1%} "
            f"{metrics['unknown_recall']:8.1%} {metrics['false_unknown']:9.1%} "
            f"{metrics['p50_ms']:7.1f}"
        )
    print("\nA variant is only better if it wins on accuracy without losing UNKNOWN detection.")


async def run(args: argparse.Namespace) -> int:
    settings = Settings(
        chroma_mode="ephemeral",
        auth_enabled=False,
        audit_log_enabled=False,
        entity_extraction_enabled=False,
    )
    container = build_container(settings)
    print("seeding an in-memory catalogue ...")
    stats = seed(container)
    print(f"  {stats['domains']} domains, {stats['intents']} intents, {stats['examples']} examples")

    cases = load_cases()
    strategy = args.strategy or settings.default_strategy
    predictions = await predict(container, cases, args.strategy)
    metrics = print_report(predictions, settings, strategy, args.verbose)

    if args.sweep:
        sweep(predictions, settings)
    if args.compare:
        await compare(container, cases, settings)

    if metrics["top1"] < args.min_top1 or metrics["unknown_recall"] < args.min_unknown:
        print("\nFAILED: below the requested thresholds")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the intent classifier")
    parser.add_argument("--sweep", action="store_true", help="grid-search the two thresholds")
    parser.add_argument("--compare", action="store_true", help="compare retrieval strategies")
    parser.add_argument("--strategy", help="run one named retrieval strategy")
    parser.add_argument("--verbose", action="store_true", help="print every prediction")
    parser.add_argument("--min-top1", type=float, default=0.0, help="fail below this top-1")
    parser.add_argument(
        "--min-unknown", type=float, default=0.0, help="fail below this UNKNOWN detection rate"
    )
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
