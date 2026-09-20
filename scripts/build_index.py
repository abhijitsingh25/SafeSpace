"""Rebuild the corpus embeddings and manifest.

Replaces ``build_faiss_index.py``, which hardcoded ``C:\\GenAi\\SafeSpace\\data`` and
resolved its output directory one level above the repo, so it wrote embeddings to a
sibling of the project. Paths now derive from the repo root.

No FAISS: the index is a plain normalised matrix that ``backend.retrieval`` searches
with a matmul.

    uv run --extra local python -m scripts.build_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date

import numpy as np

from backend.providers.embeddings import get_embedder
from backend.settings import get_settings


def load_corpus(path) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    meta: list[dict] = []
    skipped = 0

    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  ! skipping malformed line {line_no}")
                skipped += 1
                continue

            instruction = (obj.get("instruction") or "").strip()
            response = (obj.get("response") or "").strip()
            if not instruction or not response:
                print(f"  ! skipping incomplete record on line {line_no}")
                skipped += 1
                continue

            texts.append(f"Q: {instruction}\nA: {response}")
            meta.append({"id": len(meta), "instruction": instruction, "response": response})

    if skipped:
        print(f"  skipped {skipped} bad record(s)")
    return texts, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    out_dir = settings.embeddings_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading  {settings.corpus_path}")
    texts, meta = load_corpus(settings.corpus_path)
    print(f"Loaded   {len(texts)} records")
    if not texts:
        print("Nothing to embed - aborting.")
        return 1

    embedder = get_embedder()
    print(f"Embedding with {embedder.model_name} ...")

    batches = []
    for start in range(0, len(texts), args.batch_size):
        batch = texts[start : start + args.batch_size]
        batches.append(embedder.embed(batch))
        print(f"  {min(start + args.batch_size, len(texts))}/{len(texts)}")

    vectors = np.vstack(batches).astype(np.float32)

    # retrieval.py relies on unit norms so that a dot product *is* cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    np.save(out_dir / "exam_texts.npy", vectors)
    with open(out_dir / "exam_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    manifest = {
        "embedding_model": embedder.model_name,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "normalized": True,
        "vectors_sha256_prefix": hashlib.sha256(vectors.tobytes()).hexdigest()[:16],
        "built_at": date.today().isoformat(),
        "source": str(settings.corpus_path.relative_to(settings.corpus_path.parents[1])),
        "_comment": (
            "Query embeddings MUST come from embedding_model. Mixing vector spaces "
            "returns plausible but meaningless results."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nWrote {vectors.shape[0]} x {vectors.shape[1]} vectors to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
