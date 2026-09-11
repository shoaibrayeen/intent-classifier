from app.config import Settings
from app.models.classification import SessionTurn
from app.services.session_service import SessionService


class FakeRepo:
    def __init__(self):
        self.turns = []
        self.pruned = []

    def history(self, session_id, domain_id, limit):
        rows = [t for s, d, t in self.turns if s == session_id and d == domain_id]
        return rows[-limit:] if limit else rows

    def all_turns(self, session_id):
        return [t for s, _, t in self.turns if s == session_id]

    def append(self, session_id, domain_id, turn, principal=""):
        self.turns.append((session_id, domain_id, turn))

    def prune(self, session_id, keep):
        self.pruned.append((session_id, keep))
        return 0

    def expire(self, older_than):
        return 0

    def delete(self, session_id):
        before = len(self.turns)
        self.turns = [row for row in self.turns if row[0] != session_id]
        return before - len(self.turns)


def service(**overrides) -> tuple[SessionService, FakeRepo]:
    repo = FakeRepo()
    return SessionService(Settings(chroma_mode="ephemeral", **overrides), repo), repo


def turn(n, text, intent="CONTRACT_SEARCH", **entities) -> SessionTurn:
    return SessionTurn(turn=n, text=text, intent=intent, entities=entities)


def test_contextual_text_prepends_the_previous_question():
    svc, _ = service()
    history = [turn(1, "Show me Microsoft contracts?")]
    assert svc.contextual_text(history, "when do they expire") == (
        "Show me Microsoft contracts when do they expire"
    )


def test_no_context_without_history_or_when_disabled():
    svc, _ = service()
    assert svc.contextual_text([], "anything") is None
    off, _ = service(context_retrieval_enabled=False)
    assert off.contextual_text([turn(1, "x")], "y") is None


def test_repeating_the_same_question_adds_no_context():
    svc, _ = service()
    assert svc.contextual_text([turn(1, "Find contracts")], "find contracts") is None


def test_carry_over_fills_only_what_the_new_intent_accepts():
    svc, _ = service()
    history = [turn(1, "…", counterparty="Microsoft", status="ACTIVE")]
    merged, carried = svc.carry_over(history, {}, {"counterparty": {"type": "string"}})
    assert merged == {"counterparty": "Microsoft"}
    assert carried == ["counterparty"]


def test_the_new_turn_wins_over_history():
    svc, _ = service()
    history = [turn(1, "…", counterparty="Microsoft")]
    merged, carried = svc.carry_over(
        history, {"counterparty": "Oracle"}, {"counterparty": {"type": "string"}}
    )
    assert merged == {"counterparty": "Oracle"}
    assert carried == []


def test_most_recent_turn_wins_among_earlier_turns():
    svc, _ = service()
    history = [turn(1, "…", counterparty="Microsoft"), turn(2, "…", counterparty="Oracle")]
    merged, _ = svc.carry_over(history, {}, {"counterparty": {"type": "string"}})
    assert merged["counterparty"] == "Oracle"


def test_carry_over_can_be_disabled():
    svc, _ = service(entity_carry_over_enabled=False)
    merged, carried = svc.carry_over(
        [turn(1, "…", counterparty="Microsoft")], {}, {"counterparty": {"type": "string"}}
    )
    assert merged == {} and carried == []


def test_recording_prunes_to_the_configured_size():
    svc, repo = service(session_max_turns=3)
    svc.record("s1", "d1", turn(1, "a"))
    assert repo.pruned == [("s1", 3)]
    assert svc.next_turn_number(svc.history("s1", "d1")) == 2


def test_disabled_sessions_neither_read_nor_write():
    svc, repo = service(sessions_enabled=False)
    svc.record("s1", "d1", turn(1, "a"))
    assert repo.turns == []
    assert svc.history("s1", "d1") == []
