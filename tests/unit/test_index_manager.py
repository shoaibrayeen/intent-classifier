from app.models.common import Status
from app.models.example import Example, text_hash
from app.models.index import IndexState
from app.models.intent import IntentConfig
from app.services.index_manager import IndexManager


class FakeIntents:
    def __init__(self, intents: list[IntentConfig]):
        self.intents = intents

    def list_for_domain(self, domain_id: str):
        return [i for i in self.intents if i.domain_id == domain_id]


class FakeExamples:
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def list_for_domain(self, domain_id: str, intent_ids=None):
        items = [e for e in self.examples if e.domain_id == domain_id]
        if intent_ids is not None:
            items = [e for e in items if e.intent_id in intent_ids]
        return items

    def count_for_domain(self, domain_id: str) -> int:
        return len([e for e in self.examples if e.domain_id == domain_id])


def make_example(intent: IntentConfig, text: str) -> Example:
    return Example(
        domain_id=intent.domain_id,
        intent_id=intent.id,
        intent_name=intent.name,
        text=text,
        text_hash=text_hash(text),
    )


def build(examples: list[Example], intents: list[IntentConfig]) -> IndexManager:
    return IndexManager(FakeExamples(examples), FakeIntents(intents))


def test_empty_domain_builds_an_index_with_no_bm25_model():
    manager = build([], [])
    index = manager.get("d1")
    assert index.bm25 is None
    assert index.doc_count == 0
    assert manager.status("d1").state == IndexState.READY


def test_status_before_first_build_is_not_built():
    manager = build([], [])
    assert manager.status("d1").state == IndexState.NOT_BUILT


def test_rebuild_increments_the_version():
    intent = IntentConfig(name="SEARCH", domain_id="d1")
    manager = build([make_example(intent, "find contracts")], [intent])
    first = manager.get("d1")
    second = manager.rebuild("d1")
    assert second.version == first.version + 1


def test_dirty_domain_rebuilds_on_next_get():
    intent = IntentConfig(name="SEARCH", domain_id="d1")
    manager = build([make_example(intent, "find contracts")], [intent])
    version = manager.get("d1").version
    manager.mark_dirty("d1")
    assert manager.status("d1").state == IndexState.DIRTY
    assert manager.get("d1").version == version + 1
    assert manager.status("d1").state == IndexState.READY


def test_disabled_intents_are_excluded_from_the_corpus():
    active = IntentConfig(name="SEARCH", domain_id="d1")
    disabled = IntentConfig(name="OLD", domain_id="d1", status=Status.DISABLED)
    manager = build(
        [make_example(active, "find contracts"), make_example(disabled, "legacy phrasing")],
        [active, disabled],
    )
    assert manager.get("d1").doc_count == 1


def test_drop_forgets_the_domain():
    intent = IntentConfig(name="SEARCH", domain_id="d1")
    manager = build([make_example(intent, "find contracts")], [intent])
    manager.get("d1")
    manager.drop("d1")
    assert manager.status("d1").state == IndexState.NOT_BUILT
