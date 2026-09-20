"""BM25 fallback.

This path runs whenever no embedding provider is configured, which is the default state
of a fresh clone - so it carries the exam corpus for most first-time users.
"""

from __future__ import annotations

import pytest

from backend.lexical import BM25Index, tokenize
from backend.retrieval import retrieve, retrieve_lexical


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("I can't sleep because of the exams!") == ["can't", "sleep", "exams"]


def test_tokenize_drops_single_characters():
    assert "a" not in tokenize("a big exam")


def test_empty_index_is_safe():
    assert BM25Index([]).search("anything", 3) == []


def test_empty_query_returns_nothing():
    assert BM25Index(["exam stress is hard"]).search("", 3) == []


def test_query_with_only_unknown_terms_returns_nothing():
    index = BM25Index(["exam stress", "revision timetable"])
    assert index.search("zzzz qqqq", 3) == []


def test_ranks_the_matching_document_first():
    index = BM25Index(
        [
            "I cannot sleep because I keep worrying about failing my exams",
            "My revision timetable keeps slipping",
            "I argued with my brother about chores",
        ]
    )
    hits = index.search("I can't sleep, worried about failing", 3)
    assert hits[0][0] == 0
    assert hits[0][1] > 0


def test_scores_are_ordered_descending():
    index = BM25Index(["exam stress sleep", "exam stress", "unrelated text"])
    scores = [score for _, score in index.search("exam stress sleep", 3)]
    assert scores == sorted(scores, reverse=True)


def test_respects_k():
    index = BM25Index(["exam one", "exam two", "exam three", "exam four"])
    assert len(index.search("exam", 2)) == 2


# --- Against the real corpus --------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_substring"),
    [
        ("I can't sleep because I keep worrying about failing", "sleep"),
        ("my parents put so much pressure on me about marks", "parents"),
        ("my study schedule is falling apart", "schedule"),
    ],
)
def test_real_corpus_retrieves_topically_relevant_passages(real_corpus, query, expected_substring):
    passages = retrieve_lexical(query, k=3)
    assert passages, f"nothing retrieved for {query!r}"
    joined = " ".join(p.instruction.lower() for p in passages)
    assert expected_substring in joined


def test_retrieve_falls_back_to_lexical_when_embedder_is_unavailable(real_corpus, monkeypatch):
    """The behaviour that keeps the exam corpus working on a fresh clone."""

    def no_embedder():
        raise RuntimeError("EMBEDDING_PROVIDER=local needs sentence-transformers")

    monkeypatch.setattr("backend.retrieval.get_embedder", no_embedder)

    passages = retrieve("I can't sleep because I keep worrying about failing", k=3)
    assert passages, "fallback returned nothing - the corpus would be silently unused"
    assert "sleep" in passages[0].instruction.lower()


def test_lexical_floor_filters_weak_matches(real_corpus, monkeypatch):
    from backend.settings import get_settings

    monkeypatch.setattr(get_settings(), "lexical_min_score", 10_000.0, raising=False)
    assert retrieve_lexical("anything at all") == []
