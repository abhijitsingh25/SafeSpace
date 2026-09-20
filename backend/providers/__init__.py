"""Pluggable inference backends.

Everything that talks to a model goes through here, so that:
  * local development can keep Ollama/MedGemma,
  * production can run on hosted APIs that fit in a serverless bundle,
  * and tests can substitute a fake without a network or a GPU.
"""

from backend.providers.embeddings import Embedder, get_embedder
from backend.providers.llm import TherapistLLM, get_therapist_llm

__all__ = ["Embedder", "TherapistLLM", "get_embedder", "get_therapist_llm"]
