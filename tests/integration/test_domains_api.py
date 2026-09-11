def test_create_and_read_domain(client):
    created = client.post("/api/v1/domains", json={"name": "contract"})
    assert created.status_code == 201
    domain_id = created.json()["id"]

    fetched = client.get(f"/api/v1/domains/{domain_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "contract"
    assert fetched.json()["intent_count"] == 0


def test_duplicate_name_is_rejected_case_insensitively(client):
    client.post("/api/v1/domains", json={"name": "contract"})
    duplicate = client.post("/api/v1/domains", json={"name": "CONTRACT"})
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "conflict"


def test_missing_domain_returns_404(client):
    response = client.get("/api/v1/domains/does-not-exist")
    assert response.status_code == 404


def test_list_domains_reports_intent_and_example_counts(client, contract_domain):
    domains = client.get("/api/v1/domains").json()
    assert len(domains) == 1
    assert domains[0]["intent_count"] == 2
    assert domains[0]["example_count"] == 16


def test_update_renames_a_domain(client):
    domain_id = client.post("/api/v1/domains", json={"name": "contract"}).json()["id"]
    updated = client.put(f"/api/v1/domains/{domain_id}", json={"name": "agreements"})
    assert updated.status_code == 200
    assert updated.json()["name"] == "agreements"


def test_rename_onto_an_existing_name_is_rejected(client):
    first = client.post("/api/v1/domains", json={"name": "contract"}).json()["id"]
    client.post("/api/v1/domains", json={"name": "employee"})
    conflict = client.put(f"/api/v1/domains/{first}", json={"name": "employee"})
    assert conflict.status_code == 409


def test_delete_cascades_to_intents_and_examples(client, contract_domain):
    domain_id = contract_domain["domain"]["id"]
    search_id = contract_domain["search"]["id"]

    assert client.delete(f"/api/v1/domains/{domain_id}").status_code == 204
    assert client.get(f"/api/v1/domains/{domain_id}").status_code == 404
    assert client.get(f"/api/v1/domains/{domain_id}/intents/{search_id}").status_code == 404
    assert client.get("/api/v1/domains").json() == []


def test_invalid_name_is_rejected_by_validation(client):
    assert client.post("/api/v1/domains", json={"name": "  "}).status_code == 422
