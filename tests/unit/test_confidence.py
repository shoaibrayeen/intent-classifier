from app.config import Settings
from app.models.classification import IntentScore
from app.services.confidence import compute
from app.services.rrf import max_possible_score

SETTINGS = Settings(chroma_mode="ephemeral")


def score(intent: str, value: float, support: int, similarity: float) -> IntentScore:
    return IntentScore(
        intent=intent,
        intent_id=intent.lower(),
        score=value,
        supporting_examples=support,
        best_similarity=similarity,
    )


def test_no_candidates_gives_zero_confidence():
    breakdown = compute([], SETTINGS)
    assert breakdown.confidence == 0.0


def test_perfect_match_is_confidently_above_threshold():
    ceiling = max_possible_score(SETTINGS.rrf_k, SETTINGS.agg_top_n)
    breakdown = compute([score("SEARCH", ceiling, 5, 0.95)], SETTINGS)
    assert breakdown.s_dense == 1.0
    assert breakdown.s_rrf == 1.0
    assert breakdown.s_margin == 1.0  # no runner-up
    assert breakdown.s_support == 1.0
    assert breakdown.confidence == 1.0


def test_weak_similarity_drags_confidence_below_threshold():
    breakdown = compute([score("SEARCH", 0.05, 2, 0.40)], SETTINGS)
    assert breakdown.s_dense == 0.0
    assert breakdown.confidence < SETTINGS.confidence_threshold


def test_close_runner_up_reduces_the_margin_signal():
    clear = compute([score("A", 0.09, 3, 0.9), score("B", 0.01, 1, 0.6)], SETTINGS)
    contested = compute([score("A", 0.09, 3, 0.9), score("B", 0.088, 3, 0.89)], SETTINGS)
    assert contested.s_margin < clear.s_margin
    assert contested.confidence < clear.confidence


def test_single_intent_domain_has_full_margin_but_still_needs_similarity():
    breakdown = compute([score("ONLY", 0.03, 1, 0.52)], SETTINGS)
    assert breakdown.s_margin == 1.0
    assert breakdown.s_dense < 0.1
    assert breakdown.confidence < SETTINGS.confidence_threshold


def test_confidence_is_bounded():
    breakdown = compute([score("A", 99.0, 99, 9.0)], SETTINGS)
    assert 0.0 <= breakdown.confidence <= 1.0
