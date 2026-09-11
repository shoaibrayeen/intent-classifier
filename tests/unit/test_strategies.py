from app.config import Settings
from app.services.strategies import BM25_ONLY, DENSE_ONLY, HYBRID_RRF, StrategySelector


def settings(**overrides) -> Settings:
    base = {"chroma_mode": "ephemeral"}
    base.update(overrides)
    return Settings(**base)


def test_ab_testing_is_off_by_default():
    selector = StrategySelector(settings())
    assert selector.enabled is False
    assert selector.select("any-request-id").name == HYBRID_RRF


def test_assignment_is_stable_for_the_same_request_id():
    """A reported result has to be reproducible, so the same id must always
    land on the same pipeline."""
    selector = StrategySelector(settings(ab_testing_enabled=True))
    first = selector.select("request-42").name
    for _ in range(20):
        assert selector.select("request-42").name == first


def test_assignment_spreads_across_variants():
    selector = StrategySelector(
        settings(ab_testing_enabled=True, ab_variants=f"{HYBRID_RRF},{DENSE_ONLY}")
    )
    seen = {selector.select(f"request-{i}").name for i in range(60)}
    assert seen == {HYBRID_RRF, DENSE_ONLY}


def test_an_explicit_variant_always_wins():
    selector = StrategySelector(settings(ab_testing_enabled=True))
    assert selector.select("request-1", override=BM25_ONLY).name == BM25_ONLY


def test_an_unknown_variant_falls_back_to_the_default():
    selector = StrategySelector(settings())
    assert selector.select("r", override="does_not_exist").name == HYBRID_RRF


def test_unknown_names_in_the_variant_list_are_ignored():
    selector = StrategySelector(
        settings(ab_testing_enabled=True, ab_variants="hybrid_rrf,typo_variant")
    )
    assert selector.variants == [HYBRID_RRF]
    assert selector.enabled is False  # one usable variant is not an experiment


def test_variants_inherit_the_service_defaults():
    selector = StrategySelector(settings(rrf_k=30, agg_top_n=2, retrieval_top_k=15))
    strategy = selector.get(HYBRID_RRF)
    assert (strategy.rrf_k, strategy.agg_top_n, strategy.retrieval_top_k) == (30, 2, 15)


def test_single_retriever_variants_declare_what_they_drop():
    selector = StrategySelector(settings())
    assert selector.get(DENSE_ONLY).use_bm25 is False
    assert selector.get(BM25_ONLY).use_dense is False
