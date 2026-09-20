"""Vercel serverless entrypoint.

Vercel's Python runtime looks for an ASGI app named ``app`` in this module. The heavy
lifting is all in ``backend`` - this file exists only so the platform can find it.

What makes the bundle fit the 250MB limit:
  * no torch / sentence-transformers  (EMBEDDING_PROVIDER=huggingface)
  * no faiss                          (numpy matmul in backend.retrieval)
  * no ollama                         (LLM_PROVIDER=groq)

See requirements.txt, which pins only the production subset.
"""

from backend.main import app

__all__ = ["app"]
