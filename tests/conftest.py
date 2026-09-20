from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EMERGENCY_CALLS_ENABLED", "false")
os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "1000")

from backend.providers import embeddings as embeddings_module  # noqa: E402
from backend.providers import llm as llm_module  # noqa: E402
from backend.retrieval import CorpusIndex, set_index  # noqa: E402


class FakeEmbedder:
    """Deterministic stand-in: no network, no torch, but a real vector space."""

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text.lower())) % (2**32))
            vector = rng.normal(size=self.dim).astype(np.float32)
            out.append(vector / np.linalg.norm(vector))
        return np.vstack(out)


class FakeLLM:
    """Records what it was asked, returns something recognisable."""

    def __init__(self, reply: str = "That sounds really hard. What's weighing on you most?"):
        self.reply = reply
        self.calls: list[dict] = []

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system": system})
        return self.reply


@pytest.fixture
def fake_llm():
    llm = FakeLLM()
    llm_module.set_therapist_llm(llm)
    yield llm
    llm_module.set_therapist_llm(None)


@pytest.fixture
def fake_embedder():
    embedder = FakeEmbedder()
    embeddings_module.set_embedder(embedder)
    yield embedder
    embeddings_module.set_embedder(None)


@pytest.fixture
def real_corpus():
    """The actual committed vectors - proves the shipped artefacts still load."""
    from backend.retrieval import load_index
    from backend.retrieval import set_index as _set

    _set(None)
    index = load_index(force=True)
    yield index
    _set(None)


@pytest.fixture
def tiny_corpus():
    vectors = np.eye(3, 384, dtype=np.float32)
    meta = [
        {"id": 0, "instruction": "exam nerves", "response": "break revision into steps"},
        {"id": 1, "instruction": "cannot sleep", "response": "wind down before bed"},
        {"id": 2, "instruction": "parents pressure", "response": "set boundaries kindly"},
    ]
    index = CorpusIndex(vectors, meta, "sentence-transformers/all-MiniLM-L6-v2")
    set_index(index)
    yield index
    set_index(None)
