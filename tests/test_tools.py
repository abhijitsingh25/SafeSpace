from __future__ import annotations

import pytest

from backend.settings import get_settings
from backend.tools import (
    ALL_TOOLS,
    ask_exam_stress_specialist,
    ask_mental_health_specialist,
    emergency_support,
    find_nearby_therapists,
    guided_coping_exercise,
    place_emergency_call,
)


def test_every_tool_has_a_description_the_model_can_route_on():
    for tool in ALL_TOOLS:
        assert tool.description and len(tool.description) > 40, f"{tool.name} under-described"


def test_tool_names_are_unique():
    names = [t.name for t in ALL_TOOLS]
    assert len(names) == len(set(names))


def test_system_prompt_only_references_registered_tools():
    """The original prompt named two tools that were never registered."""
    from backend.agent import SYSTEM_PROMPT

    registered = {t.name for t in ALL_TOOLS}
    for name in registered:
        assert name in SYSTEM_PROMPT, f"{name} is registered but absent from the prompt"

    # And nothing in the prompt that looks like a tool name is unregistered.
    for stale in ("locate_therapist_tool", "ask_exam_rag_tool", "ask_exam_rag"):
        assert stale not in SYSTEM_PROMPT


def test_general_support_uses_the_therapist_llm(fake_llm):
    result = ask_mental_health_specialist.invoke({"query": "I feel lonely"})
    assert result == fake_llm.reply
    assert fake_llm.calls[0]["prompt"] == "I feel lonely"


def test_exam_tool_grounds_on_retrieved_context(tiny_corpus, fake_embedder, fake_llm, monkeypatch):
    monkeypatch.setattr(get_settings(), "retrieval_min_score", -1.0, raising=False)
    ask_exam_stress_specialist.invoke({"query": "exams are stressing me"})

    call = fake_llm.calls[0]
    assert "Reference passages" in call["prompt"]
    assert "never mention" in call["system"].lower()


def test_exam_tool_falls_back_when_nothing_relevant_is_retrieved(
    tiny_corpus, fake_embedder, fake_llm, monkeypatch
):
    """Better no context than misleading context presented as grounding."""
    monkeypatch.setattr(get_settings(), "retrieval_min_score", 0.999, raising=False)
    ask_exam_stress_specialist.invoke({"query": "unrelated"})

    assert "Reference passages" not in fake_llm.calls[0]["prompt"]


# --- Therapist locator: the fabrication fix -----------------------------------------


def test_locator_without_a_key_invents_nothing(monkeypatch):
    monkeypatch.setattr(get_settings(), "serpapi_api_key", None, raising=False)
    result = find_nearby_therapists.invoke({"location": "Boston"})

    # The old implementation returned three fake practices with fake phone numbers.
    assert "Dr. Ayesha Kapoor" not in result
    assert "555" not in result
    assert "psychologytoday.com" in result
    assert "invent" in result.lower()


def test_locator_uses_real_results_when_configured(monkeypatch):
    monkeypatch.setattr(get_settings(), "serpapi_api_key", "fake-key", raising=False)
    monkeypatch.setattr(
        "backend.tools._search_therapists",
        lambda *a, **k: [
            {"title": "Riverside Counselling", "address": "12 Mill Rd", "phone": "+1 617 555 0100"}
        ],
    )
    result = find_nearby_therapists.invoke({"location": "Boston"})
    assert "Riverside Counselling" in result
    assert "verify credentials" in result


def test_locator_degrades_honestly_when_the_search_fails(monkeypatch):
    monkeypatch.setattr(get_settings(), "serpapi_api_key", "fake-key", raising=False)

    def boom(*a, **k):
        raise RuntimeError("serpapi down")

    monkeypatch.setattr("backend.tools._search_therapists", boom)
    result = find_nearby_therapists.invoke({"location": "Boston"})
    assert "isn't responding" in result
    assert "555" not in result


def test_locator_handles_empty_results(monkeypatch):
    monkeypatch.setattr(get_settings(), "serpapi_api_key", "fake-key", raising=False)
    monkeypatch.setattr("backend.tools._search_therapists", lambda *a, **k: [])
    assert "couldn't find listings" in find_nearby_therapists.invoke({"location": "Nowhere"})


# --- Coping exercises ---------------------------------------------------------------


@pytest.mark.parametrize("technique", ["breathing", "grounding", "muscle_relaxation", "worry_time"])
def test_each_exercise_has_steps(technique):
    result = guided_coping_exercise.invoke({"technique": technique})
    assert len(result) > 150
    assert result.rstrip().endswith("?")


def test_unknown_technique_falls_back_to_grounding():
    assert "5-4-3-2-1" in guided_coping_exercise.invoke({"technique": "nonsense"})


def test_technique_matching_tolerates_spacing_and_case():
    assert "Progressive" in guided_coping_exercise.invoke({"technique": "Muscle Relaxation"})


# --- Emergency ----------------------------------------------------------------------


def test_emergency_call_is_disabled_by_default():
    """A fresh clone must not be able to place real phone calls by accident."""
    placed, status = place_emergency_call()
    assert placed is False
    assert "disabled" in status.lower()


def test_emergency_call_never_raises(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "emergency_calls_enabled", True, raising=False)
    monkeypatch.setattr(settings, "twilio_account_sid", "sid", raising=False)
    monkeypatch.setattr(settings, "twilio_auth_token", "token", raising=False)
    monkeypatch.setattr(settings, "twilio_from_number", "+15550000000", raising=False)
    monkeypatch.setattr(settings, "emergency_contact", "+15550000001", raising=False)

    # A Twilio outage must not take down the turn that surfaces the hotline numbers.
    placed, status = place_emergency_call()
    assert placed is False
    assert isinstance(status, str)


def test_emergency_support_returns_hotlines_even_when_the_call_fails():
    result = emergency_support.invoke({"region": "US"})
    assert "988" in result
    # The old tool returned None here, leaving the model with nothing to say.
    assert result is not None and len(result) > 100


def test_exam_tool_degrades_when_no_embedder_is_configured(fake_llm, monkeypatch):
    """A fresh clone with only GROQ_API_KEY must still get an answer, not a 503."""

    def no_embedder(*a, **k):
        raise RuntimeError("EMBEDDING_PROVIDER=local needs sentence-transformers")

    monkeypatch.setattr("backend.tools.retrieve", no_embedder)
    result = ask_exam_stress_specialist.invoke({"query": "my exams are next week"})

    assert result == fake_llm.reply
    assert "Reference passages" not in fake_llm.calls[0]["prompt"]
