from __future__ import annotations

import json

import numpy as np
import pytest

from backend.retrieval import CorpusIndex, format_passages, load_index, retrieve
from backend.settings import get_settings


def test_committed_corpus_still_loads(real_corpus):
    """Guards the shipped artefacts: 350 normalised 384-d vectors, aligned to metadata."""
    assert len(real_corpus) == 350
    assert real_corpus.dim == 384
    assert len(real_corpus.meta) == 350


def test_committed_vectors_are_unit_norm(real_corpus):
    norms = np.linalg.norm(real_corpus.vectors, axis=1)
    # retrieval treats the dot product as cosine similarity, which only holds if this does.
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_manifest_matches_the_vectors_on_disk():
    settings = get_settings()
    with open(settings.embeddings_dir / "manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    vectors = np.load(settings.embeddings_dir / "exam_texts.npy")
    assert manifest["dim"] == vectors.shape[1]
    assert manifest["count"] == vectors.shape[0]
    assert manifest["embedding_model"] == settings.embedding_model


def test_misaligned_corpus_is_rejected_at_construction():
    with pytest.raises(ValueError, match="misaligned"):
        CorpusIndex(
            np.zeros((3, 4), dtype=np.float32),
            [{"instruction": "a", "response": "b"}],
            "m",
        )


def test_wrong_query_dimension_is_rejected(tiny_corpus):
    """The failure this prevents returns plausible nonsense rather than an error."""
    with pytest.raises(ValueError, match="embedded by different models"):
        tiny_corpus.search(np.ones(10, dtype=np.float32), k=1)


def test_search_returns_results_ranked_by_score(tiny_corpus):
    query = np.zeros(384, dtype=np.float32)
    query[1] = 1.0  # aligns exactly with row 1
    results = tiny_corpus.search(query, k=3)

    assert results[0].instruction == "cannot sleep"
    assert results[0].score == pytest.approx(1.0)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_search_caps_k_at_corpus_size(tiny_corpus):
    assert len(tiny_corpus.search(np.eye(1, 384, dtype=np.float32)[0], k=99)) == 3


def test_numpy_search_matches_bruteforce_on_the_real_corpus(real_corpus):
    """Equivalence check against the exhaustive scan FAISS was doing."""
    rng = np.random.default_rng(0)
    query = rng.normal(size=real_corpus.dim).astype(np.float32)
    query /= np.linalg.norm(query)

    expected = np.argsort(-(real_corpus.vectors @ query))[:5].tolist()
    actual = [
        real_corpus.meta.index(
            next(m for m in real_corpus.meta if m["instruction"] == p.instruction)
        )
        for p in real_corpus.search(query, k=5)
    ]
    assert actual == expected


def test_retrieve_filters_below_threshold(tiny_corpus, fake_embedder, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_min_score", 0.99, raising=False)
    # Random fake embeddings are near-orthogonal to the basis vectors, so nothing clears
    # a 0.99 floor - the point being that weak matches are dropped, not passed on.
    assert retrieve("completely unrelated query") == []


def test_retrieve_returns_passages_when_threshold_is_permissive(
    tiny_corpus, fake_embedder, monkeypatch
):
    settings = get_settings()
    monkeypatch.setattr(settings, "retrieval_min_score", -1.0, raising=False)
    assert len(retrieve("exam worry", k=2)) == 2


def test_format_passages_is_empty_for_no_results():
    assert format_passages([]) == ""


def test_format_passages_numbers_and_scores(tiny_corpus):
    query = np.eye(1, 384, dtype=np.float32)[0]
    text = format_passages(tiny_corpus.search(query, k=2))
    assert "Context 1" in text and "Context 2" in text
    assert "similarity" in text


def test_index_is_cached_between_calls(real_corpus):
    assert load_index() is load_index()
