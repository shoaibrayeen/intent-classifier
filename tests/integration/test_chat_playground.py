"""The chat workbench, and reopening a recorded conversation."""

from tests.conftest import seed_contract_domain


def chat_app(app_factory, **overrides):
    app, client = app_factory(
        {"llm_provider": "mock", "entity_extraction_enabled": True, **overrides}
    )
    seed_contract_domain(client)
    return app, client


def ask(client, text, session, domain="contract"):
    return client.post(
        "/ui/playground/classify",
        data={"domain": domain, "text": text, "session_id": session},
        headers={"HX-Request": "true"},
    )


# ------------------------------------------------------------------ the chat
def test_a_new_chat_starts_empty_with_a_generated_session(app_factory):
    _, client = chat_app(app_factory)
    page = client.get("/ui/playground").text

    assert "No messages yet" in page
    assert "No turn selected" in page  # the detail pane waits rather than lying
    assert 'name="session_id"' in page


def test_asking_returns_a_bubble_and_swaps_the_detail_pane(app_factory):
    _, client = chat_app(app_factory)

    body = ask(client, "Find all active contracts with Microsoft", "s1").text

    assert 'class="bubble user"' in body
    assert "CONTRACT_SEARCH" in body
    # The reasoning arrives with the answer, out of band into the left pane.
    assert 'id="detail-panel"' in body and 'hx-swap-oob="true"' in body
    assert "Confidence breakdown" in body
    assert "counterparty" in body


def test_enter_submits_the_message(app_factory):
    _, client = chat_app(app_factory)
    page = client.get("/ui/playground").text
    assert "event.key==='Enter'&&!event.shiftKey" in page
    assert "Shift+Enter for a new line" in page


def test_the_domain_dropdown_is_in_the_chat(app_factory):
    _, client = chat_app(app_factory)
    page = client.get("/ui/playground").text
    assert 'name="domain"' in page and "contract" in page


def test_an_unknown_answer_says_so_plainly(app_factory):
    _, client = chat_app(app_factory)
    body = ask(client, "what is the weather today", "s1").text
    assert "No matching intent" in body
    assert "weak match" in body or "uncertain" in body


# --------------------------------------------------------- reopening history
def test_a_recorded_conversation_reopens_with_its_turns(app_factory):
    _, client = chat_app(app_factory)
    ask(client, "Find all active contracts with Microsoft", "chat-1")
    ask(client, "which of them expire next month", "chat-1")

    page = client.get("/ui/playground?session=chat-1").text

    assert "Find all active contracts with Microsoft" in page
    assert "which of them expire next month" in page
    assert "CONTRACT_SEARCH" in page
    assert "2 turn(s)" in page


def test_reopening_shows_the_last_turns_details_not_an_empty_pane(app_factory):
    _, client = chat_app(app_factory)
    ask(client, "Find all active contracts with Microsoft", "chat-1")

    page = client.get("/ui/playground?session=chat-1").text

    assert "No turn selected" not in page
    assert "Confidence breakdown" in page
    assert "counterparty" in page


def test_reopening_selects_the_domain_the_conversation_happened_in(app_factory):
    _, client = chat_app(app_factory)
    domain_id = client.get("/api/v1/domains").json()[0]["id"]
    ask(client, "Find all contracts with Microsoft", "chat-1")

    page = client.get("/ui/playground?session=chat-1").text

    assert f'value="{domain_id}" selected' in page


def test_any_earlier_turn_can_be_reinspected(app_factory):
    _, client = chat_app(app_factory)
    ask(client, "Find all active contracts with Microsoft", "chat-1")
    ask(client, "which of them expire next month", "chat-1")

    first = client.get("/ui/playground/details/chat-1/1").text

    assert "CONTRACT_SEARCH" in first
    assert "Find all active contracts with Microsoft" in first
    assert "Confidence breakdown" in first


def test_the_details_are_what_happened_then_not_a_reclassification(app_factory):
    """Recorded, so deleting the examples afterwards cannot rewrite history."""
    _, client = chat_app(app_factory)
    ask(client, "Find all active contracts with Microsoft", "chat-1")
    domain_id = client.get("/api/v1/domains").json()[0]["id"]
    for intent in client.get(f"/api/v1/domains/{domain_id}/intents").json():
        client.delete(f"/api/v1/domains/{domain_id}/intents/{intent['id']}")

    panel = client.get("/ui/playground/details/chat-1/1").text

    assert "CONTRACT_SEARCH" in panel
    assert "Confidence breakdown" in panel


def test_a_missing_turn_is_a_404(app_factory):
    _, client = chat_app(app_factory)
    assert client.get("/ui/playground/details/nope/1").status_code == 404


def test_clearing_from_the_chat_empties_the_thread(app_factory):
    _, client = chat_app(app_factory)
    ask(client, "Find all contracts with Microsoft", "chat-1")

    cleared = client.delete("/ui/playground/session/chat-1", headers={"HX-Request": "true"})

    assert "No messages yet" in cleared.text
    assert client.get("/api/v1/sessions/chat-1").status_code == 404


# ------------------------------------------------------------ sessions page
def test_session_rows_open_the_conversation(app_factory):
    _, client = chat_app(app_factory)
    ask(client, "Find all contracts with Microsoft", "chat-1")

    page = client.get("/ui/sessions").text

    assert "/ui/playground?session=chat-1" in page
    assert ">Open<" in page


def test_a_turn_recorded_through_the_api_is_inspectable_in_the_ui(app_factory):
    """Details are kept whether or not the caller asked to see them."""
    _, client = chat_app(app_factory)
    client.post(
        "/api/v1/classify",
        json={
            "domain": "contract",
            "text": "Find all contracts with Microsoft",
            "session_id": "api-1",
        },
    )

    panel = client.get("/ui/playground/details/api-1/1").text

    assert "Confidence breakdown" in panel
    assert "Retrieval" in panel


def test_older_turns_without_a_snapshot_still_render(app_factory):
    """Turns recorded before snapshots existed must not break the pane."""
    from app.models.classification import SessionTurn

    app, client = chat_app(app_factory)
    domain_id = client.get("/api/v1/domains").json()[0]["id"]
    app.state.container.sessions.record(
        "legacy",
        domain_id,
        SessionTurn(turn=1, text="an old question", intent="CONTRACT_SEARCH", confidence=0.7),
    )

    panel = client.get("/ui/playground/details/legacy/1").text

    assert "CONTRACT_SEARCH" in panel
    assert "recorded before the full trace was kept" in panel


def test_the_empty_state_stands_down_once_a_message_arrives(app_factory):
    """New turns are appended after the placeholder, so it must hide itself."""
    css = (
        __import__("pathlib").Path(__file__).resolve().parents[2] / "app/static/app.css"
    ).read_text()
    assert ".chat-thread:has(.chat-turn) .chat-empty { display: none; }" in css
