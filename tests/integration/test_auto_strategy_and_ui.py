"""Auto strategy selection, domain personas, and the added UI routes."""

from tests.conftest import seed_contract_domain


def test_hybrid_is_the_default_and_is_reported(client, contract_domain):
    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()
    assert body["strategy"] == "hybrid_rrf"
    assert body["strategies_tried"] == []


def test_auto_answers_with_hybrid_when_hybrid_can(app_factory):
    _, client = app_factory({"default_strategy": "auto"})
    seed_contract_domain(client)

    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    ).json()

    assert body["intent"] == "CONTRACT_SEARCH"
    assert body["strategy"] == "hybrid_rrf"
    assert body["strategies_tried"] == []  # no fallback was needed


def test_auto_still_declines_a_genuinely_unanswerable_query(app_factory):
    """Falling back must not turn every query into a match."""
    _, client = app_factory({"default_strategy": "auto"})
    seed_contract_domain(client)

    body = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "what is the weather today"}
    ).json()

    assert body["intent"] == "UNKNOWN"
    assert body["strategy"] == "hybrid_rrf"  # hybrid's verdict is the one reported
    assert body["strategies_tried"]  # the alternates were tried and also declined


def test_auto_stops_immediately_when_a_domain_has_no_examples(app_factory):
    _, client = app_factory({"default_strategy": "auto"})
    client.post("/api/v1/domains", json={"name": "empty"})

    body = client.post("/api/v1/classify", json={"domain": "empty", "text": "anything"}).json()

    assert body["intent"] == "UNKNOWN"
    assert body["reason"] == "no_examples"
    assert body["strategies_tried"] == []  # no point retrying against nothing


def test_a_rescue_must_clear_a_stricter_bar_than_a_normal_answer(app_factory):
    """Without the margin, auto traded UNKNOWN detection for accuracy: on the
    evaluation set it fell from 90% to 70%, which for a tool-calling system
    means confidently invoking the wrong tool."""
    _, client = app_factory({"default_strategy": "auto", "auto_rescue_margin": 0.0})
    seed_contract_domain(client)

    permissive = client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "book me a flight to Paris"}
    ).json()

    _, strict_client = app_factory({"default_strategy": "auto", "auto_rescue_margin": 0.10})
    seed_contract_domain(strict_client)
    strict = strict_client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "book me a flight to Paris"}
    ).json()

    # The margin is what keeps an off-domain query from being rescued.
    assert strict["intent"] == "UNKNOWN"
    assert strict["confidence"] <= permissive["confidence"] or permissive["intent"] == "UNKNOWN"


def test_a_pinned_variant_still_wins_under_auto(app_factory):
    _, client = app_factory({"default_strategy": "auto"})
    seed_contract_domain(client)

    body = client.post(
        "/api/v1/classify",
        json={
            "domain": "contract",
            "text": "Find all contracts with Microsoft",
            "variant": "bm25_only",
        },
    ).json()

    assert body["strategy"] == "bm25_only"


def test_metrics_record_the_strategy_that_answered(app_factory):
    _, client = app_factory({"default_strategy": "auto"})
    seed_contract_domain(client)
    client.post(
        "/api/v1/classify", json={"domain": "contract", "text": "Find all contracts with Microsoft"}
    )

    assert 'strategy="hybrid_rrf"' in client.get("/api/v1/metrics").text


# ------------------------------------------------------------------- personas
def test_generated_instructions_state_the_role_for_the_domain(app_factory):
    _, client = app_factory({"llm_provider": "mock"})

    contract = client.post(
        "/api/v1/domains",
        json={
            "name": "contract",
            "description": "contractual agreements and renewals",
            "generate_instructions": True,
        },
    ).json()
    shop = client.post(
        "/api/v1/domains",
        json={
            "name": "shop",
            "description": "add items to cart and checkout orders",
            "generate_instructions": True,
        },
    ).json()

    assert "legal analyst" in contract["system_instructions"]
    assert "e-commerce shopping assistant" in shop["system_instructions"]
    assert contract["system_instructions"].startswith("You are acting as")


def test_an_unfamiliar_domain_still_gets_a_role(app_factory):
    _, client = app_factory({"llm_provider": "mock"})
    domain = client.post(
        "/api/v1/domains",
        json={"name": "widgets", "description": "managing widgets", "generate_instructions": True},
    ).json()
    assert domain["system_instructions"].startswith("You are acting as a widgets specialist")


def test_the_role_reaches_the_extraction_prompt(app_factory):
    """A persona nobody sends is just decoration."""
    app, client = app_factory({"llm_provider": "mock", "entity_extraction_enabled": True})
    fixture = seed_contract_domain(client)
    client.post(
        f"/api/v1/domains/{fixture['domain']['id']}/instructions/generate",
        json={"brief": "contractual agreements and obligations"},
    )

    client.post("/api/v1/classify", json={"domain": "contract", "text": "Find Microsoft contracts"})

    system, _ = app.state.container.entity_extractor._client.calls[-1]
    assert "You are acting as a contracts and legal analyst" in system


# ------------------------------------------------------------------ UI routes
def test_every_navigation_route_renders(client, contract_domain):
    for path in [
        "/",
        "/ui/intents",
        "/ui/mcp",
        "/ui/playground",
        "/ui/sessions",
        "/ui/evaluation",
        "/ui/operations",
        "/ui/api",
    ]:
        assert client.get(path).status_code == 200, path


def test_the_current_page_is_marked_in_the_nav(client):
    page = client.get("/ui/playground").text
    assert 'href="/ui/playground"\n         class="active"' in page or 'class="active"' in page


def test_the_global_intent_list_spans_domains(app_factory):
    _, client = app_factory({"llm_provider": "mock"})
    seed_contract_domain(client)
    client.post("/api/v1/domains", json={"name": "finance", "description": "invoices"})

    page = client.get("/ui/intents").text

    assert "CONTRACT_SEARCH" in page
    assert "All intents" in page


def test_operations_refresh_targets_render_on_their_own(client, contract_domain):
    health = client.get("/ui/operations/index-health")
    audit = client.get("/ui/operations/audit")

    assert health.status_code == 200 and "contract" in health.text
    assert audit.status_code == 200
    # Partials, not whole pages: they are swapped into an existing page.
    assert "<html" not in health.text


def test_the_operations_page_declares_auto_refresh(client):
    page = client.get("/ui/operations").text
    assert 'hx-trigger="every 10s"' in page
    assert "/ui/operations/index-health" in page


def test_sessions_appear_in_the_ui_and_can_be_cleared(app_factory):
    _, client = app_factory({"llm_provider": "mock"})
    seed_contract_domain(client)
    client.post(
        "/api/v1/classify",
        json={
            "domain": "contract",
            "text": "Find all contracts with Microsoft",
            "session_id": "demo-1",
        },
    )

    page = client.get("/ui/sessions").text
    assert "demo-1" in page

    cleared = client.delete("/ui/sessions/demo-1", headers={"HX-Request": "true"})
    assert "Cleared 'demo-1'" in cleared.text
    assert "demo-1" not in client.get("/ui/sessions/list").text


def test_confidence_bands_are_colour_coded_in_the_playground(app_factory):
    """Green above 80, amber 50-80, red below 50: the colour must agree with
    the number it sits next to."""
    _, client = app_factory({"llm_provider": "mock"})
    seed_contract_domain(client)

    for text in ["Find all contracts with Microsoft", "what is the weather today"]:
        confidence = client.post(
            "/api/v1/classify", json={"domain": "contract", "text": text}
        ).json()["confidence"]
        expected = "high" if confidence >= 0.8 else ("medium" if confidence >= 0.5 else "low")
        rendered = client.post(
            "/ui/playground/classify",
            data={"domain": "contract", "text": text},
            headers={"HX-Request": "true"},
        ).text
        assert f"result-confidence {expected}" in rendered, (text, confidence)
        assert f"confidence-bar {expected}" in rendered


def test_a_lexical_only_pass_is_never_an_auto_fallback(app_factory):
    """It cannot apply the similarity floor, so it matches off-domain text."""
    from app.config import Settings
    from app.services.strategies import StrategySelector

    selector = StrategySelector(Settings(chroma_mode="ephemeral"))
    assert all(s.use_dense for s in selector.auto_sequence())


# ------------------------------------------------------------------ changelog
def test_the_changelog_is_served(client):
    response = client.get("/ui/changelog")
    assert response.status_code == 200
    assert "Changelog" in response.text
    assert "text/html" in response.headers["content-type"]


def test_the_changelog_is_also_at_the_obvious_path(client):
    """People look for /changelog, not /ui/changelog."""
    assert client.get("/changelog").status_code == 200


def test_the_changelog_is_linked_from_the_navigation(client):
    assert "/ui/changelog" in client.get("/ui/playground").text


def test_the_api_reference_still_resolves_from_its_new_home(client):
    response = client.get("/ui/api")
    assert response.status_code == 200
    assert "Intent Classifier" in response.text


# -------------------------------------------------------------- result layout
def test_the_debug_tables_carry_banded_dots(app_factory):
    """Colour appears beside the number, never instead of it."""
    _, client = app_factory({"llm_provider": "mock"})
    seed_contract_domain(client)

    rendered = client.post(
        "/ui/playground/classify",
        data={"domain": "contract", "text": "Find all contracts with Microsoft"},
        headers={"HX-Request": "true"},
    ).text

    # One dot per confidence signal, plus one per ranked intent.
    assert rendered.count('class="dot ') >= 6
    assert "Confidence breakdown" in rendered and "Ranked intents after fusion" in rendered
    # The values are still printed, not replaced by colour.
    assert "s_dense" in rendered and "best similarity" in rendered


def test_a_long_intent_name_is_allowed_to_wrap(app_factory):
    """A long underscore-joined name used to run over the confidence beside it."""
    css = (
        __import__("pathlib").Path(__file__).resolve().parents[2] / "app/static/app.css"
    ).read_text()
    assert "overflow-wrap: anywhere" in css
    # Grid items default to min-width:auto, which is what stopped it shrinking.
    assert ".result-head > div { min-width: 0; }" in css


def test_the_stylesheet_is_requested_with_a_version(client):
    page = client.get("/ui/playground").text
    assert "/static/app.css?v=" in page
