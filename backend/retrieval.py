"""Vector retrieval over the support corpus.

FAISS is gone. The corpus is 350 vectors of 384 dimensions - 537KB - and
``IndexFlatIP`` was doing an exhaustive scan of it anyway. A single numpy matmul gives
bit-identical results, is faster at this size once FAISS's call overhead is counted, and
removes a ~30MB dependency from a bundle that has a 250MB ceiling.

If the corpus ever grows past roughly 50k rows, reintroduce an ANN index behind
``search()`` - the interface is deliberately narrow so that swap stays local.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from backend.lexical import BM25Index
from backend.providers.embeddings import get_embedder
from backend.settings import get_settings

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievedPassage:
    score: float
    instruction: str
    response: str


class CorpusIndex:
    """Normalised embedding matrix plus aligned metadata."""

    def __init__(self, vectors: np.ndarray, meta: list[dict], model_name: str) -> None:
        if vectors.shape[0] != len(meta):
            raise ValueError(
                f"Corpus is misaligned: {vectors.shape[0]} vectors vs {len(meta)} "
                "metadata rows. Rebuild with `python -m scripts.build_index`."
            )
        self.vectors = vectors
        self.meta = meta
        self.model_name = model_name

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def search(self, query_vector: np.ndarray, k: int) -> list[RetrievedPassage]:
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query_vector.shape[0] != self.dim:
            raise ValueError(
                f"Query vector has dim {query_vector.shape[0]} but the corpus has "
                f"{self.dim}. The query and corpus were embedded by different models."
            )

        # Both sides are L2-normalised, so the dot product is the cosine similarity.
        scores = self.vectors @ query_vector

        k = min(k, len(self))
        # argpartition is O(n) vs argsort's O(n log n); only the top-k needs ordering.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        return [
            RetrievedPassage(
                score=float(scores[i]),
                instruction=self.meta[i]["instruction"],
                response=self.meta[i]["response"],
            )
            for i in top
        ]


_index: CorpusIndex | None = None


def _manifest_path(embeddings_dir: Path) -> Path:
    return embeddings_dir / "manifest.json"


def load_index(force: bool = False) -> CorpusIndex:
    """Load and cache the corpus. Called at startup so no user pays for a cold load."""
    global _index
    if _index is not None and not force:
        return _index

    settings = get_settings()
    emb_dir = settings.embeddings_dir
    vectors = np.load(emb_dir / "exam_texts.npy")
    with open(emb_dir / "exam_meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)

    model_name = settings.embedding_model
    manifest_file = _manifest_path(emb_dir)
    if manifest_file.exists():
        with open(manifest_file, encoding="utf-8") as fh:
            manifest = json.load(fh)
        model_name = manifest.get("embedding_model", model_name)

        # The failure this prevents is nasty: mismatched vector spaces still return
        # results, they are just quietly meaningless.
        if model_name != settings.embedding_model:
            raise RuntimeError(
                f"Corpus was embedded with '{model_name}' but EMBEDDING_MODEL is "
                f"'{settings.embedding_model}'. Query and corpus vectors are not "
                "comparable. Either restore the original model setting or re-embed the "
                "corpus with `python -m scripts.build_index`."
            )

    _index = CorpusIndex(vectors=vectors, meta=meta, model_name=model_name)
    log.info("corpus_index_loaded", count=len(_index), dim=_index.dim, model=model_name)
    return _index


def set_index(index: CorpusIndex | None) -> None:
    """Test seam."""
    global _index, _bm25
    _index = index
    _bm25 = None


_bm25: BM25Index | None = None


def load_bm25() -> BM25Index:
    """Lexical index over the same metadata. No model, no key, no network."""
    global _bm25
    if _bm25 is None:
        index = load_index()
        _bm25 = BM25Index([f"{row['instruction']} {row['response']}" for row in index.meta])
        log.info("bm25_index_built", documents=len(_bm25))
    return _bm25


def retrieve_lexical(query: str, k: int | None = None) -> list[RetrievedPassage]:
    settings = get_settings()
    k = k or settings.retrieval_top_k
    index = load_index()

    hits = load_bm25().search(query, k)
    return [
        RetrievedPassage(
            score=score,
            instruction=index.meta[i]["instruction"],
            response=index.meta[i]["response"],
        )
        for i, score in hits
        if score >= settings.lexical_min_score
    ]


def retrieve(query: str, k: int | None = None) -> list[RetrievedPassage]:
    """Return the best-matching passages for a query.

    Prefers embeddings, which handle paraphrase. Falls back to BM25 when no embedding
    provider is configured, so a fresh clone still gets grounded answers rather than
    silently losing the corpus.
    """
    settings = get_settings()
    k = k or settings.retrieval_top_k

    index = load_index()
    try:
        query_vector = get_embedder().embed([query])[0]
    except Exception as exc:
        log.info("embedding_unavailable_using_lexical", error=str(exc)[:120])
        return retrieve_lexical(query, k)

    passages = index.search(query_vector, k)

    kept = [p for p in passages if p.score >= settings.retrieval_min_score]
    # A weak best match means the corpus has nothing useful; better to give the model
    # no context than misleading context.
    if not kept:
        log.info(
            "retrieval_below_threshold",
            best_score=passages[0].score if passages else None,
            threshold=settings.retrieval_min_score,
        )
    return kept


def format_passages(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return ""
    return "\n\n".join(
        f"Context {i + 1} (relevance {p.score:.2f}):\nSituation: {p.instruction}\n"
        f"Helpful response: {p.response}"
        for i, p in enumerate(passages)
    )
