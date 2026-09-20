"""Therapist-voice LLM providers.

``ollama`` keeps MedGemma on the developer's machine. ``groq`` is the deployable path:
Vercel has no GPU and no room for model weights, so production runs the same persona
prompt against a hosted model.

Ollama is called over plain HTTP rather than through the ``ollama`` SDK so that the
production bundle carries one less dependency.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
import structlog

from backend.settings import get_settings

log = structlog.get_logger(__name__)

# The clinical persona. Shared by every provider so switching backends does not
# silently change the product's voice.
THERAPIST_SYSTEM_PROMPT = """You are Dr. Emily Hartman, a warm and experienced clinical
psychologist.

Respond to people with:
1. Emotional attunement - reflect what they seem to be feeling before anything else.
2. Gentle normalisation - help them see their reaction as human and understandable.
3. Practical guidance - one or two concrete things that sometimes help.
4. Strengths-focused support - notice what they are already doing well.

Key principles:
- Never use brackets, labels, or headings. Write as a person speaking.
- Blend the elements seamlessly and vary your sentence structure.
- Mirror the person's language level and energy.
- Keep it under about 200 words.
- Always end with one open-ended question that moves toward the root of the concern.
- Never diagnose, never prescribe medication, and never promise outcomes.
- If someone describes being in danger, tell them plainly that a trained human can help
  more than you can, and encourage them to contact a crisis line.
"""

FALLBACK_MESSAGE = (
    "I'm having trouble reaching my words right now - that's a technical problem on my "
    "end, not anything to do with you. Your feelings matter and I do want to hear more. "
    "Could you try sending that again in a moment?"
)


@runtime_checkable
class TherapistLLM(Protocol):
    """Minimal surface the tools need. Keeps the agent free of provider details."""

    def chat(self, prompt: str, system: str | None = None) -> str: ...


class OllamaProvider:
    """Local MedGemma via the Ollama HTTP API."""

    def __init__(self, model: str, host: str, timeout: float) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or THERAPIST_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 350},
        }
        try:
            response = httpx.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception as exc:
            # Log the cause - the previous version swallowed it and made every Ollama
            # failure look identical.
            log.error("ollama_chat_failed", model=self.model, host=self.host, error=str(exc))
            return FALLBACK_MESSAGE


class GroqProvider:
    """Hosted inference. What production actually runs on."""

    def __init__(self, model: str, api_key: str, timeout: float) -> None:
        from langchain_groq import ChatGroq

        self.model = model
        self._client = ChatGroq(model=model, api_key=api_key, temperature=0.7, timeout=timeout)

    def chat(self, prompt: str, system: str | None = None) -> str:
        try:
            result = self._client.invoke(
                [("system", system or THERAPIST_SYSTEM_PROMPT), ("user", prompt)]
            )
            return getattr(result, "content", str(result)).strip()
        except Exception as exc:
            log.error("groq_chat_failed", model=self.model, error=str(exc))
            return FALLBACK_MESSAGE


_provider: TherapistLLM | None = None


def get_therapist_llm() -> TherapistLLM:
    """Process-wide singleton, built from settings."""
    global _provider
    if _provider is not None:
        return _provider

    settings = get_settings()
    if settings.llm_provider == "groq":
        _provider = GroqProvider(
            model=settings.groq_model,
            api_key=settings.require_agent_key(),
            timeout=settings.llm_timeout_seconds,
        )
    else:
        _provider = OllamaProvider(
            model=settings.ollama_model,
            host=settings.ollama_host,
            timeout=settings.llm_timeout_seconds,
        )
    log.info("therapist_llm_ready", provider=settings.llm_provider)
    return _provider


def set_therapist_llm(provider: TherapistLLM | None) -> None:
    """Test seam."""
    global _provider
    _provider = provider
