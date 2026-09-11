"""Metrics, audit logging, request ids and index health."""

import json

from tests.conftest import seed_contract_domain

QUERY = "Find all contracts with Microsoft"


def test_every_response_carries_a_request_id(client):
    response = client.get("/api/v1/health")
    assert response.headers["X-Request-ID"]
    assert response.headers["Server-Timing"].startswith("app;dur=")


def test_a_supplied_request_id_is_echoed_back(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "trace-me"})
    assert response.headers["X-Request-ID"] == "trace-me"


def test_the_request_id_appears_in_the_classification(client, contract_domain):
    body = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": QUERY},
        headers={"X-Request-ID": "trace-me"},
    ).json()
    assert body["request_id"] == "trace-me"


def test_errors_carry_the_request_id_too(client):
    body = client.post(
        "/api/v1/classify",
        json={"domain": "ghost", "text": "hello"},
        headers={"X-Request-ID": "trace-me"},
    ).json()
    assert body["error"]["request_id"] == "trace-me"


def test_latency_is_measured_and_reported(client, contract_domain):
    body = client.post("/api/v1/classify", json={"domain": "contract", "text": QUERY}).json()
    assert body["latency_ms"] > 0


def test_metrics_expose_classification_outcomes(client, contract_domain):
    client.post("/api/v1/classify", json={"domain": "contract", "text": QUERY})
    client.post("/api/v1/classify", json={"domain": "contract", "text": "what is the weather"})

    body = client.get("/api/v1/metrics").text

    assert 'intent_classifications_total{outcome="matched"' in body
    assert 'intent_classifications_total{outcome="unknown"' in body
    assert "intent_classification_duration_seconds" in body
    assert "intent_stage_duration_seconds" in body


def test_metrics_label_routes_by_template_not_by_id(client, contract_domain):
    """Otherwise every domain id would create its own time series."""
    domain_id = contract_domain["domain"]["id"]
    client.get(f"/api/v1/domains/{domain_id}")

    body = client.get("/api/v1/metrics").text

    assert "/api/v1/domains/{domain_id}" in body
    assert domain_id not in body


def test_metrics_can_be_turned_off(app_factory):
    _, client = app_factory({"metrics_enabled": False})
    assert client.get("/api/v1/metrics").status_code == 404


def test_index_health_reports_both_indexes_in_step(client, contract_domain):
    body = client.get("/api/v1/health/index").json()

    assert body["status"] == "ok"
    domain = body["domains"][0]
    assert domain["in_sync"] is True
    assert domain["bm25_doc_count"] == domain["dense_count"] == 16


def test_health_reports_the_feature_configuration(client):
    body = client.get("/api/v1/health").json()

    assert body["entity_extraction"]["enabled"] is False
    assert body["entity_extraction"]["provider_configured"] is False
    assert body["auth_enabled"] is False
    assert "strategy" in body


def test_the_audit_log_records_classifications(app_factory, tmp_path):
    log = tmp_path / "audit.jsonl"
    app, client = app_factory({"audit_log_enabled": True, "audit_log_path": str(log)})
    seed_contract_domain(client)

    client.post("/api/v1/classify", json={"domain": "contract", "text": QUERY})

    entries = [json.loads(line) for line in log.read_text().splitlines()]
    actions = {entry["action"] for entry in entries}
    assert "domain.create" in actions
    assert "classify" in actions

    classification = next(e for e in entries if e["action"] == "classify")
    assert classification["intent"] == "CONTRACT_SEARCH"
    assert classification["outcome"] == "matched"
    assert classification["request_id"]
    # Query text is personal data, so it is not recorded unless asked for.
    assert classification["query"] is None


def test_query_text_is_recorded_only_when_enabled(app_factory, tmp_path):
    log = tmp_path / "audit.jsonl"
    app, client = app_factory(
        {
            "audit_log_enabled": True,
            "audit_log_path": str(log),
            "audit_log_query_text": True,
        }
    )
    seed_contract_domain(client)

    client.post("/api/v1/classify", json={"domain": "contract", "text": QUERY})

    entries = [json.loads(line) for line in log.read_text().splitlines()]
    classification = next(e for e in entries if e["action"] == "classify")
    assert classification["query"] == QUERY


def test_denied_requests_are_audited(app_factory, tmp_path):
    log = tmp_path / "audit.jsonl"
    app, client = app_factory(
        {
            "auth_enabled": True,
            "api_keys": "k:*:admin",
            "audit_log_enabled": True,
            "audit_log_path": str(log),
        }
    )

    client.post("/api/v1/classify", json={"domain": "contract", "text": "hi"})

    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert any(e["action"] == "auth" and e["outcome"] == "denied" for e in entries)


def test_the_audit_endpoint_requires_admin(app_factory):
    app, client = app_factory({"auth_enabled": True, "api_keys": "reader:*:read"})
    assert client.get("/api/v1/audit", headers={"X-API-Key": "reader"}).status_code == 403
