"""Query embedding providers.

The corpus is embedded once, offline, and committed to the repo. At request time we only
ever need to embed one short query string - which is why shipping ~2GB of torch into a
250MB serverless bundle was never going to work.

Both providers target the *same* model (all-MiniLM-L6-v2), so the committed corpus
vectors stay valid either way. ``retrieval.py`` enforces that with a manifest check;
mixing vector spaces silently returns plausible-looking nonsense, so it is worth the
guard rail.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
import numpy as np
import structlog

from backend.settings import get_settings

log = structlog.get_logger(__name__)

HF_INFERENCE_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction"


@runtime_checkable
class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, shape (len(texts), dim)."""
        ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against a zero vector producing NaNs rather than just a useless result.
    norms[norms == 0] = 1.0
    return vectors / norms


class LocalEmbedder:
    """sentence-transformers on this machine. Requires the ``local`` extra."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local needs sentence-transformers. Install it with "
                "`uv sync --extra local`, or set EMBEDDING_PROVIDER=huggingface to use "
                "the hosted API instead."
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class HuggingFaceEmbedder:
    """Hosted Inference API. Same model as LocalEmbedder, so vectors are interchangeable."""

    def __init__(self, model_name: str, api_key: str | None, timeout: float) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def embed(self, texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{HF_INFERENCE_URL}/{self.model_name}",
            headers=self._headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        vectors = np.asarray(data, dtype=np.float32)
        # Some pipelines return per-token embeddings (n, tokens, dim); mean-pool those
        # to match what sentence-transformers produces.
        if vectors.ndim == 3:
            vectors = vectors.mean(axis=1)
        if vectors.ndim != 2:
            raise RuntimeError(
                f"Unexpected embedding shape {vectors.shape} from {self.model_name}. "
                "Check that the model is a sentence-transformers model."
            )
        return _normalise(vectors)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is not None:
        return _embedder

    settings = get_settings()
    if settings.embedding_provider == "huggingface":
        _embedder = HuggingFaceEmbedder(
            model_name=settings.embedding_model,
            api_key=settings.huggingface_api_key,
            timeout=settings.embedding_timeout_seconds,
        )
    else:
        _embedder = LocalEmbedder(model_name=settings.embedding_model)
    log.info(
        "embedder_ready",
        provider=settings.embedding_provider,
        model=settings.embedding_model,
    )
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Test seam."""
    global _embedder
    _embedder = embedder
