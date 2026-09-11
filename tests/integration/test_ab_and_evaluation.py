"""A/B testing of retrieval strategies, and the evaluation dashboard."""

from tests.conftest import seed_contract_domain

QUERY = "Find all contracts with Microsoft"


def test_a_request_can_pin_a_retrieval_strategy(client, contract_domain):
    body = client.post(
        "/api/v1/classify/debug",
        json={"domain": "contract", "text": QUERY, "variant": "bm25_only"},
    ).json()

    assert body["debug"]["strategy"] == "bm25_only"
    assert body["debug"]["dense_hits"] == []
    assert body["debug"]["bm25_hits"]


def test_dense_only_skips_the_lexical_retriever(client, contract_domain):
    body = client.post(
        "/api/v1/classify/debug",
        json={"domain": "contract", "text": QUERY, "variant": "dense_only"},
    ).json()

    assert body["debug"]["bm25_hits"] == []
    assert body["debug"]["dense_hits"]
    assert body["debug"]["timings"]["bm25_ms"] == 0


def test_every_variant_still_finds_an_obvious_intent(client, contract_domain):
    for variant in ["hybrid_rrf", "dense_only", "bm25_only", "hybrid_rrf_k20", "hybrid_wide"]:
        body = client.post(
            "/api/v1/classify", json={"domain": "contract", "text": QUERY, "variant": variant}
        ).json()
        assert body["intent"] == "CONTRACT_SEARCH", variant


def test_assignment_is_stable_across_requests(app_factory):
    app, client = app_factory({"ab_testing_enabled": True})
    seed_contract_domain(client)

    picks = set()
    for _ in range(5):
        body = client.post(
            "/api/v1/classify/debug",
            json={"domain": "contract", "text": QUERY},
            headers={"X-Request-ID": "same-id"},
        ).json()
        picks.add(body["debug"]["strategy"])
    assert len(picks) == 1


def test_the_strategies_endpoint_describes_what_is_available(client):
    body = client.get("/api/v1/strategies").json()

    names = {entry["name"] for entry in body["available"]}
    assert {"hybrid_rrf", "dense_only", "bm25_only"} <= names
    assert body["ab_testing_enabled"] is False
    assert body["default"] == "hybrid_rrf"


def test_metrics_break_classifications_down_by_strategy(client, contract_domain):
    client.post(
        "/api/v1/classify", json={"domain": "contract", "text": QUERY, "variant": "dense_only"}
    )
    body = client.get("/api/v1/metrics").text
    assert 'strategy="dense_only"' in body


def test_the_evaluation_page_renders(client):
    response = client.get("/ui/evaluation")
    assert response.status_code == 200
    assert "Run evaluation" in response.text


def test_the_evaluation_run_reports_metrics(client, contract_domain):
    response = client.post("/ui/evaluation/run", data={}, headers={"HX-Request": "true"})

    assert response.status_code == 200
    assert "Top-1 accuracy" in response.text
    assert "UNKNOWN detection" in response.text
    # The seeded fixture has no employee domain, which must be reported as skipped.
    assert "Skipped" in response.text


def test_the_evaluation_run_accepts_a_variant(client, contract_domain):
    response = client.post(
        "/ui/evaluation/run", data={"variant": "dense_only"}, headers={"HX-Request": "true"}
    )
    assert "dense_only" in response.text


def test_the_operations_page_shows_index_health(client, contract_domain):
    response = client.get("/ui/operations")

    assert response.status_code == 200
    assert "Index health" in response.text
    assert "contract" in response.text
    assert "Pipeline configuration" in response.text
