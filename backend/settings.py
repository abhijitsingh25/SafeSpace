"""Centralised, validated configuration.

Replaces the old gitignored ``backend/config.py``, which meant the app could not start
from a clean clone. Everything is read from the environment (or a local ``.env``), with
defaults chosen so that a developer with only ``GROQ_API_KEY`` set can run the app.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Core ---------------------------------------------------------------
    environment: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # ---- Agent LLM (the router/orchestrator) --------------------------------
    groq_api_key: str | None = None
    groq_model: str = "qwen/qwen3.8-27b"
    agent_temperature: float = 0.6
    agent_top_p: float = 0.95
    agent_max_tokens: int = 2048
    # Qwen is a reasoning model. "hidden" keeps its chain-of-thought out of the reply -
    # a user in distress should never be shown the model deliberating about them.
    agent_reasoning_effort: str = "default"
    agent_reasoning_format: str = "hidden"

    # ---- Therapist-voice LLM ------------------------------------------------
    # "ollama" keeps MedGemma locally; "groq" is the deployable path.
    llm_provider: Literal["ollama", "groq"] = "ollama"
    ollama_model: str = "alibayram/medgemma:4b"
    ollama_host: str = "http://localhost:11434"
    llm_timeout_seconds: float = 60.0

    # ---- Embeddings ---------------------------------------------------------
    # "local" uses sentence-transformers (needs the `local` extra, ~2GB of torch).
    # "huggingface" calls the hosted Inference API for the SAME MiniLM model, so the
    # committed corpus vectors stay valid and nothing has to be re-embedded.
    embedding_provider: Literal["local", "huggingface"] = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_api_key: str | None = None
    embedding_timeout_seconds: float = 30.0

    # ---- Retrieval ----------------------------------------------------------
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.25

    # ---- Safety -------------------------------------------------------------
    # Off by default: placing real phone calls should be a deliberate opt-in, never
    # something a fresh clone does by accident.
    emergency_calls_enabled: bool = False
    twilio_account_sid: str | None = None
    twilio_auth_token: str | None = None
    twilio_from_number: str | None = None
    emergency_contact: str | None = None
    twilio_voice_url: str = "http://demo.twilio.com/docs/voice.xml"
    default_crisis_region: str = "INTL"

    # ---- Therapist locator --------------------------------------------------
    # Without a key the locator says so honestly instead of inventing providers.
    serpapi_api_key: str | None = None
    locator_timeout_seconds: float = 15.0

    # ---- API ----------------------------------------------------------------
    cors_allow_origins: str = "*"
    rate_limit_per_minute: int = 20
    session_ttl_seconds: int = 60 * 60 * 24

    # ---- Paths --------------------------------------------------------------
    corpus_path: Path = DATA_DIR / "mental_health_exams.jsonl"
    embeddings_dir: Path = EMBEDDINGS_DIR
    crisis_resources_path: Path = DATA_DIR / "crisis_resources.json"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def twilio_configured(self) -> bool:
        return all(
            [
                self.twilio_account_sid,
                self.twilio_auth_token,
                self.twilio_from_number,
                self.emergency_contact,
            ]
        )

    @model_validator(mode="after")
    def _check_emergency_config(self) -> Settings:
        if self.emergency_calls_enabled and not self.twilio_configured:
            raise ValueError(
                "EMERGENCY_CALLS_ENABLED is true but Twilio is not fully configured. "
                "Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER and "
                "EMERGENCY_CONTACT, or set EMERGENCY_CALLS_ENABLED=false."
            )
        return self

    def require_agent_key(self) -> str:
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(get one free at https://console.groq.com)."
            )
        return self.groq_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
