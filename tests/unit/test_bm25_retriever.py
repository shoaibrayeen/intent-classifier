from datetime import UTC, datetime

from rank_bm25 import BM25Okapi

from app.services.bm25_retriever import retrieve
from app.services.index_manager import Bm25Index
from app.services.normalizer import tokenize

TEXTS = [
    "Find all contracts with Microsoft",
    "Which contracts expire this year?",
    "Summarize the Oracle agreement",
]


def make_index() -> Bm25Index:
    return Bm25Index(
        domain_id="d1",
        version=1,
        built_at=datetime.now(UTC),
        example_ids=["e1", "e2", "e3"],
        intent_ids=["search", "expiry", "summary"],
        intent_names=["SEARCH", "EXPIRY", "SUMMARY"],
        texts=TEXTS,
        bm25=BM25Okapi([tokenize(text) for text in TEXTS]),
    )


def test_lexical_match_ranks_first():
    hits = retrieve(make_index(), tokenize("contracts expire this year"), 5)
    assert hits[0].example_id == "e2"
    assert hits[0].rank == 1


def test_zero_scoring_documents_are_dropped():
    hits = retrieve(make_index(), tokenize("microsoft"), 5)
    assert [hit.example_id for hit in hits] == ["e1"]


def test_query_with_no_overlap_returns_nothing():
    assert retrieve(make_index(), tokenize("weather forecast tomorrow"), 5) == []


def test_index_without_a_model_returns_nothing():
    index = make_index()
    index.bm25 = None
    assert retrieve(index, tokenize("contracts"), 5) == []


def test_empty_token_list_returns_nothing():
    assert retrieve(make_index(), [], 5) == []
