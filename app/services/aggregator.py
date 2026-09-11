"""Turn example-level fusion scores into intent-level scores.

Retrieval returns examples, but the product classifies intents. Summing the
top-N example scores per intent rewards agreement between several strong
examples while capping the advantage an intent gets purely from having a large
number of examples in the corpus.
"""

from __future__ import annotations

from app.models.classification import IntentScore


def aggregate(
    rrf_scores: dict[str, float],
    example_intents: dict[str, str],
    intent_names: dict[str, str],
    dense_similarity: dict[str, float],
    top_n: int = 3,
) -> list[IntentScore]:
    grouped: dict[str, list[tuple[str, float]]] = {}
    for example_id, score in rrf_scores.items():
        intent_id = example_intents.get(example_id)
        if intent_id is None:
            continue
        grouped.setdefault(intent_id, []).append((example_id, score))

    results: list[IntentScore] = []
    for intent_id, entries in grouped.items():
        entries.sort(key=lambda pair: pair[1], reverse=True)
        score = sum(value for _, value in entries[:top_n])
        similarities = [
            dense_similarity[example_id]
            for example_id, _ in entries
            if example_id in dense_similarity
        ]
        results.append(
            IntentScore(
                intent=intent_names.get(intent_id, intent_id),
                intent_id=intent_id,
                score=round(score, 6),
                supporting_examples=len(entries),
                best_similarity=round(max(similarities), 6) if similarities else 0.0,
            )
        )

    results.sort(key=lambda item: (item.score, item.best_similarity), reverse=True)
    return results
