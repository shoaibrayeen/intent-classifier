from app.services.rrf import max_possible_score, reciprocal_rank_fusion


def test_rrf_rewards_appearing_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    # b is retrieved by both retrievers; a and c by only one each.
    assert fused["b"] > fused["a"] > fused["c"]


def test_rrf_is_convex_so_extremes_beat_a_steady_middle():
    """Ranks 1 and 3 outscore 2 and 2. This is a property of 1/(k+rank),
    and it is why the aggregated score is not read as a probability."""
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k=60)
    assert fused["a"] == fused["c"] > fused["b"]


def test_rrf_single_list_matches_formula():
    fused = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert fused["x"] == 1 / 61
    assert fused["y"] == 1 / 62


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_max_possible_score_is_the_two_list_top_n_ceiling():
    expected = 2 * (1 / 61 + 1 / 62 + 1 / 63)
    assert max_possible_score(60, 3, 2) == expected
