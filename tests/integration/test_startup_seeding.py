"""Startup seeding.

Seeding has to happen in the server's own process: embedded Chroma is
single-writer, so a second process writing to a store the server already holds
leaves the server unable to read the new vectors until it restarts. These tests
pin the behaviour the container relies on.
"""


def test_seeding_is_off_by_default(app_factory):
    _, client = app_factory()
    assert client.get("/api/v1/domains").json() == []


def test_an_empty_store_is_seeded_at_startup(app_factory):
    _, client = app_factory({"seed_on_startup": True})

    domains = client.get("/api/v1/domains").json()
    assert {d["name"] for d in domains} == {"contract", "employee"}
    assert sum(d["example_count"] for d in domains) == 46


def test_a_seeded_service_classifies_without_a_restart(app_factory):
    """The failure this setting exists to prevent."""
    _, client = app_factory({"seed_on_startup": True})

    body = client.post(
        "/api/v1/classify",
        json={"domain": "contract", "text": "Show me all contracts with Microsoft"},
    ).json()
    assert body["intent"] == "CONTRACT_SEARCH"


def test_indexes_are_warm_immediately_after_seeding(app_factory):
    _, client = app_factory({"seed_on_startup": True})

    health = client.get("/api/v1/health/index").json()
    assert health["status"] == "ok"
    assert all(d["in_sync"] and d["state"] == "ready" for d in health["domains"])


def test_an_existing_catalogue_is_left_alone(app_factory):
    """Seeding must never touch a store that already holds data."""
    from app.main import _seed_if_empty

    app, client = app_factory({"seed_on_startup": True})
    before = client.get("/api/v1/domains").json()

    # Simulate a second boot against the same, already-populated store.
    _seed_if_empty(app.state.container)

    after = client.get("/api/v1/domains").json()
    assert [d["name"] for d in after] == [d["name"] for d in before]
    assert sum(d["example_count"] for d in after) == 46


def test_a_failing_seed_does_not_stop_the_service(app_factory, monkeypatch):
    """A broken demo catalogue must not take the API down."""
    from app.main import _seed_if_empty

    app, client = app_factory()

    class Boom:
        domains = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))

    _seed_if_empty(Boom())  # must not raise
    assert client.get("/api/v1/health").json()["status"] == "ok"
