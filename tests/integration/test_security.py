"""Authentication and per-domain authorization."""

from tests.conftest import seed_contract_domain

OPS = "ops-secret"
CONTRACT_ONLY = "contract-secret"
READ_ONLY = "readonly-secret"
CONTRACT_READER = "contract-reader-secret"

KEYS = (
    f"{OPS}:*:admin, "
    f"{CONTRACT_ONLY}:contract:classify, "
    f"{READ_ONLY}:*:read, "
    f"{CONTRACT_READER}:contract:read"
)


def secured(app_factory):
    app, client = app_factory({"auth_enabled": True, "api_keys": KEYS})
    client.headers.update({"X-API-Key": OPS})
    fixture = seed_contract_domain(client)
    client.headers.pop("X-API-Key")
    return client, fixture


def test_a_request_without_a_key_is_rejected(app_factory):
    client, _ = secured(app_factory)
    response = client.post("/api/v1/classify", json={"domain": "contract", "text": "hello"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


def test_an_invalid_key_is_rejected(app_factory):
    client, _ = secured(app_factory)
    response = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": "hello"},
        headers={"X-API-Key": "guessed"},
    )
    assert response.status_code == 401


def test_a_valid_key_is_accepted(app_factory):
    client, _ = secured(app_factory)
    response = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": "Find all contracts with Microsoft"},
        headers={"X-API-Key": CONTRACT_ONLY},
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "CONTRACT_SEARCH"


def test_a_key_also_works_as_a_bearer_token(app_factory):
    client, _ = secured(app_factory)
    response = client.get("/api/v1/domains", headers={"Authorization": f"Bearer {READ_ONLY}"})
    assert response.status_code == 200


def test_a_classify_only_key_cannot_write(app_factory):
    client, _ = secured(app_factory)
    response = client.post(
        "/api/v1/domains", json={"name": "sneaky"}, headers={"X-API-Key": CONTRACT_ONLY}
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_a_read_key_cannot_write(app_factory):
    client, _ = secured(app_factory)
    response = client.post(
        "/api/v1/domains", json={"name": "sneaky"}, headers={"X-API-Key": READ_ONLY}
    )
    assert response.status_code == 403


def test_admin_implies_every_lesser_scope(app_factory):
    client, _ = secured(app_factory)
    response = client.post("/api/v1/domains", json={"name": "finance"}, headers={"X-API-Key": OPS})
    assert response.status_code == 201


def test_a_domain_scoped_key_cannot_classify_another_domain(app_factory):
    client, _ = secured(app_factory)
    client.post("/api/v1/domains", json={"name": "finance"}, headers={"X-API-Key": OPS})

    response = client.post(
        "/api/v1/classify",
        json={"domain": "finance", "text": "anything"},
        headers={"X-API-Key": CONTRACT_ONLY},
    )
    assert response.status_code == 403


def test_listing_hides_domains_a_key_may_not_see(app_factory):
    client, _ = secured(app_factory)
    client.post("/api/v1/domains", json={"name": "finance"}, headers={"X-API-Key": OPS})

    visible = client.get("/api/v1/domains", headers={"X-API-Key": CONTRACT_READER}).json()
    assert [d["name"] for d in visible] == ["contract"]

    everything = client.get("/api/v1/domains", headers={"X-API-Key": READ_ONLY}).json()
    assert {d["name"] for d in everything} == {"contract", "finance"}


def test_a_classify_only_key_cannot_list_domains(app_factory):
    """read implies classify, not the reverse: a key that may ask questions
    about one domain should not be able to enumerate the catalogue."""
    client, _ = secured(app_factory)
    assert client.get("/api/v1/domains", headers={"X-API-Key": CONTRACT_ONLY}).status_code == 403


def test_health_needs_no_credential(app_factory):
    client, _ = secured(app_factory)
    assert client.get("/api/v1/health").status_code == 200


def test_auth_off_means_open_access(app_factory):
    app, client = app_factory({"auth_enabled": False})
    seed_contract_domain(client)
    assert client.get("/api/v1/domains").status_code == 200
