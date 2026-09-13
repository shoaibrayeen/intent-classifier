"""A compact snapshot of one classification, stored with its session turn.

The playground shows the same detail panel whether you have just asked a
question or clicked one from last week. That only works if the details are
recorded at the time: re-classifying an old turn later would be slower, and
would quietly give different answers once the catalogue changed, which is
exactly the wrong thing to show next to a historical conversation.

So the panel is rendered from this dictionary in both cases, and what you see
for an old turn is what actually happened then.
"""

from __future__ import annotations

from typing import Any

from app.models.classification import ClassifyDebug, ClassifyResponse

#: Retrieval hit lists are the bulky part; the top few carry the explanation.
MAX_HITS = 5
MAX_HIT_TEXT = 160


def _hit(text: str, intent: str, score: float, key: str) -> dict[str, Any]:
    return {"text": text[:MAX_HIT_TEXT], "intent": intent, key: round(score, 6)}


def snapshot_of(result: ClassifyResponse, debug: ClassifyDebug | None = None) -> dict[str, Any]:
    """Everything the detail panel needs, small enough to keep per turn.

    ``debug`` is passed explicitly rather than read from the response, because
    whether the *caller* asked to see the retrieval trace has nothing to do with
    whether the turn should remember it. The trace is assembled either way, so a
    turn recorded through the plain API is as inspectable later as one from the
    playground.
    """
    debug = debug or result.debug
    snapshot: dict[str, Any] = {
        "text": "",  # filled in by the caller, which owns the raw query
        "intent": result.intent,
        "intent_id": result.intent_id,
        "confidence": round(result.confidence, 6),
        "reason": str(result.reason) if result.reason else None,
        "strategy": result.strategy,
        "strategies_tried": list(result.strategies_tried),
        "latency_ms": result.latency_ms,
        "entities": dict(result.entities),
        "top_intents": [
            {
                "intent": item.intent,
                "score": round(item.score, 6),
                "supporting_examples": item.supporting_examples,
                "best_similarity": round(item.best_similarity, 6),
            }
            for item in result.top_intents
        ],
    }

    if result.entity_extraction is not None:
        snapshot["extraction"] = result.entity_extraction.model_dump()
    if result.tool is not None:
        tool = result.tool.model_dump()
        snapshot["mcp"] = tool.pop("mcp", None)
        snapshot["tool"] = tool
    if result.context is not None:
        snapshot["context"] = result.context.model_dump()

    if debug is not None:
        snapshot["breakdown"] = debug.confidence_breakdown.model_dump()
        snapshot["timings"] = debug.timings.model_dump()
        snapshot["index_version"] = debug.index_version
        snapshot["normalized_text"] = debug.normalized_text
        snapshot["tokens"] = list(debug.tokens)
        snapshot["dense_hits"] = [
            _hit(hit.text, hit.intent_name, hit.similarity, "similarity")
            for hit in debug.dense_hits[:MAX_HITS]
        ]
        snapshot["bm25_hits"] = [
            _hit(hit.text, hit.intent_name, hit.score, "score")
            for hit in debug.bm25_hits[:MAX_HITS]
        ]
    return snapshot


def snapshot_of_turn(turn) -> dict[str, Any]:
    """Fall back to the turn's own fields when no snapshot was stored.

    Turns recorded before snapshots existed still render, with the panel saying
    what it does not have rather than showing an empty breakdown as if every
    signal were zero.
    """
    details = dict(turn.details or {})
    details.setdefault("intent", turn.intent)
    details.setdefault("intent_id", turn.intent_id)
    details.setdefault("confidence", turn.confidence)
    details.setdefault("entities", turn.entities)
    details["text"] = turn.text
    details["turn"] = turn.turn
    details["created_at"] = turn.created_at
    return details
