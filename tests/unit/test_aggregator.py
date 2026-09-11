from app.services.aggregator import aggregate


def test_aggregate_sums_top_n_examples_per_intent():
    rrf = {"e1": 0.03, "e2": 0.02, "e3": 0.01, "e4": 0.005, "e5": 0.04}
    example_intents = {"e1": "i1", "e2": "i1", "e3": "i1", "e4": "i1", "e5": "i2"}
    names = {"i1": "SEARCH", "i2": "EXPIRY"}
    similarity = {"e1": 0.9, "e5": 0.7}

    ranked = aggregate(rrf, example_intents, names, similarity, top_n=3)

    assert [item.intent for item in ranked] == ["SEARCH", "EXPIRY"]
    # only the best three of i1's four examples count: 0.03 + 0.02 + 0.01
    assert ranked[0].score == 0.06
    assert ranked[0].supporting_examples == 4
    assert ranked[0].best_similarity == 0.9


def test_aggregate_ignores_examples_with_no_known_intent():
    ranked = aggregate({"ghost": 0.5}, {}, {}, {})
    assert ranked == []


def test_aggregate_reports_zero_similarity_for_bm25_only_intents():
    ranked = aggregate({"e1": 0.01}, {"e1": "i1"}, {"i1": "ONLY"}, {})
    assert ranked[0].best_similarity == 0.0
