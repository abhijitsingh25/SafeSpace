"""Agent tools.

Every tool returns a string the agent can ground a reply in - including the emergency
tool, which previously returned ``None`` and left the model with nothing to say at the
single most important moment in the product.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import structlog
from langchain_core.tools import tool

from backend import safety
from backend.providers.llm import THERAPIST_SYSTEM_PROMPT, get_therapist_llm
from backend.retrieval import format_passages, retrieve
from backend.settings import get_settings

log = structlog.get_logger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"


# --- General support ---------------------------------------------------------------


@tool
def ask_mental_health_specialist(query: str) -> str:
    """Give warm, evidence-informed emotional support in a therapist's voice.

    Use this for general emotional or psychological concerns: low mood, anxiety,
    loneliness, relationships, sleep, motivation, self-esteem, or when the person just
    wants to be heard. This is the default tool for anything that is not specifically
    about exams, finding a therapist, or an emergency.

    Args:
        query: The person's message, in their own words.
    """
    return get_therapist_llm().chat(query)


# --- Exam / academic stress RAG -----------------------------------------------------


@tool
def ask_exam_stress_specialist(query: str) -> str:
    """Answer questions about exams, studying, and academic pressure using a curated corpus.

    Use this whenever the person mentions exams, tests, results, grades, studying,
    revision, academic burnout, fear of failing, school or university pressure, or
    pressure from parents or teachers about academic performance.

    Args:
        query: The person's message about their academic situation.
    """
    try:
        passages = retrieve(query)
    except Exception as exc:
        # Retrieval is unavailable - typically no embedding provider configured on a
        # fresh clone. Losing grounding is a degradation; failing the turn is not
        # acceptable, so answer with general support instead.
        log.warning("exam_rag_unavailable", error=str(exc))
        return get_therapist_llm().chat(query)

    context = format_passages(passages)

    if not context:
        # Nothing relevant retrieved - fall back to general support rather than
        # feeding the model unrelated passages and calling it "grounded".
        log.info("exam_rag_no_context", query_hash=safety.hash_text(query))
        return get_therapist_llm().chat(query)

    system_prompt = (
        THERAPIST_SYSTEM_PROMPT
        + "\n\nYou have been given passages from a curated library of responses to "
        "academic stress. Draw on them where they fit, in your own words. Never mention "
        "the passages, the library, or that anything was retrieved."
    )
    user_message = (
        f"Reference passages:\n{context}\n\n"
        f"The person said: {query}\n\n"
        "Respond to them directly, using the passages where they are relevant."
    )
    return get_therapist_llm().chat(user_message, system=system_prompt)


# --- Therapist locator --------------------------------------------------------------

_NO_LOCATOR_KEY_RESPONSE = """I can't search for therapists directly right now - that
lookup isn't configured on my end, and I'd rather tell you that than invent names and
phone numbers for you to chase.

Here's how to find someone real near you:
- **Psychology Today's directory** (psychologytoday.com/us/therapists) lets you filter by
  location, insurance, specialism, and whether they offer online sessions.
- **Your GP or primary care doctor** can refer you, and often knows who has capacity.
- **Your insurance provider's directory**, if you're insured - this avoids surprise costs.
- **If you're a student**, your school or university almost certainly has free counselling.
- **Open Path Collective** (openpathcollective.org) lists therapists offering reduced rates.

When you call, it's completely normal to ask about cost, waiting times, and whether they
work with what you're going through."""


def _search_therapists(location: str, api_key: str, timeout: float) -> list[dict]:
    response = httpx.get(
        SERPAPI_URL,
        params={
            "engine": "google_maps",
            "q": f"licensed therapist counselor near {location}",
            "type": "search",
            "api_key": api_key,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("local_results", [])[:5]


@tool
def find_nearby_therapists(location: str) -> str:
    """Look up licensed therapists and counselling services near a place.

    Use this when the person asks for a therapist, counsellor, or professional help near
    them, or asks how to find one. If you do not know their location, ask before calling
    this.

    Args:
        location: A city, town, postcode, or area name.
    """
    settings = get_settings()

    if not settings.serpapi_api_key:
        return _NO_LOCATOR_KEY_RESPONSE

    try:
        results = _search_therapists(
            location, settings.serpapi_api_key, settings.locator_timeout_seconds
        )
    except Exception as exc:
        log.error("therapist_lookup_failed", location=location, error=str(exc))
        return (
            "I tried to look that up but the search isn't responding right now. "
            + _NO_LOCATOR_KEY_RESPONSE
        )

    if not results:
        return (
            f"I couldn't find listings near {location}. That may mean the search doesn't "
            "cover the area well rather than that there's nobody there.\n\n"
            + _NO_LOCATOR_KEY_RESPONSE
        )

    lines = []
    for item in results:
        parts = [f"**{item.get('title', 'Unnamed practice')}**"]
        if item.get("address"):
            parts.append(item["address"])
        if item.get("phone"):
            parts.append(item["phone"])
        if item.get("rating"):
            parts.append(f"rated {item['rating']}/5")
        lines.append("- " + " - ".join(parts))

    return (
        f"Here's what I found near {location}. I'd suggest calling to check availability "
        "and cost before booking:\n\n" + "\n".join(lines) + "\n\n"
        "These come from a public listings search, so do verify credentials - a licensed "
        "therapist will be registered with your country's professional body."
    )


# --- Guided coping exercises --------------------------------------------------------


@tool
def guided_coping_exercise(technique: str) -> str:
    """Give step-by-step instructions for a grounding or calming exercise.

    Use this when someone is acutely anxious, panicking, overwhelmed, dissociating, or
    asks for something practical to do *right now*.

    Args:
        technique: One of "breathing", "grounding", "muscle_relaxation", or "worry_time".
            Choose "grounding" if the person feels detached or panicky, "breathing" for
            physical anxiety symptoms, "muscle_relaxation" for physical tension or
            trouble sleeping, and "worry_time" for repetitive spiralling thoughts.
    """
    exercises = {
        "breathing": (
            "**Box breathing** - it works by slowing your exhale, which tells your nervous "
            "system the emergency is over.\n\n"
            "1. Breathe in through your nose for a count of 4.\n"
            "2. Hold for 4.\n"
            "3. Out through your mouth for 4.\n"
            "4. Hold empty for 4.\n\n"
            "Repeat four times. If 4 feels like too long, use 3 - the ratio matters more "
            "than the number. It's normal to feel a bit light-headed at first."
        ),
        "grounding": (
            "**5-4-3-2-1 grounding** - this pulls your attention out of your head and back "
            "into the room.\n\n"
            "Name to yourself:\n"
            "- **5** things you can see\n"
            "- **4** things you can feel touching you\n"
            "- **3** things you can hear\n"
            "- **2** things you can smell\n"
            "- **1** thing you can taste\n\n"
            "Go slowly. If you lose track, just start the current step again - losing track "
            "isn't failing at it."
        ),
        "muscle_relaxation": (
            "**Progressive muscle relaxation** - tension often outlasts the thing that "
            "caused it, and this discharges it.\n\n"
            "Working from your feet upward, for each muscle group: squeeze tight for 5 "
            "seconds, then release all at once and notice the difference for 10 seconds.\n\n"
            "Feet, calves, thighs, stomach, hands, arms, shoulders, jaw, forehead.\n\n"
            "The release is the active part, not the squeeze. Skip anywhere that's injured."
        ),
        "worry_time": (
            "**Scheduled worry time** - this works better than trying to stop worrying, "
            "which tends to backfire.\n\n"
            "1. Pick a fixed 15-minute slot later today, not close to bedtime.\n"
            "2. When a worry shows up before then, write it down in one line and tell "
            "yourself you'll deal with it at that time.\n"
            "3. At the slot, read the list and worry deliberately for the full 15 minutes.\n\n"
            "Most people find that by the time the slot arrives, a good half of the list "
            "has lost its charge."
        ),
    }

    key = technique.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in exercises:
        key = "grounding"
    return exercises[key] + "\n\nWould you like to try it now and tell me how it goes?"


# --- Emergency ----------------------------------------------------------------------


def place_emergency_call() -> tuple[bool, str]:
    """Place the Twilio call. Returns (success, human-readable status).

    Never raises: a failure here must not abort the turn that surfaces hotline numbers.
    """
    settings = get_settings()

    if not settings.emergency_calls_enabled:
        log.info("emergency_call_skipped", reason="disabled_by_config")
        return False, "Emergency calling is disabled in this environment."

    if not settings.twilio_configured:
        log.error("emergency_call_skipped", reason="twilio_not_configured")
        return False, "Emergency calling is enabled but Twilio is not fully configured."

    try:
        from twilio.rest import Client

        client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
        call = client.calls.create(
            to=settings.emergency_contact,
            from_=settings.twilio_from_number,
            url=settings.twilio_voice_url,
        )
        log.warning("emergency_call_placed", call_sid=call.sid)
        return True, "An emergency contact has been called."
    except Exception as exc:
        # The old version let this propagate into the agent loop, so a Twilio outage
        # took down the crisis turn itself.
        log.error("emergency_call_failed", error=str(exc))
        return False, f"The emergency call could not be placed: {exc}"


@tool
def emergency_support(region: str = "") -> str:
    """Surface crisis helplines and alert the configured emergency contact.

    Use this only when someone expresses suicidal thoughts, intent to self-harm, or is
    otherwise in immediate danger.

    Args:
        region: Two-letter country code (US, UK, IN, CA, AU) if known, otherwise empty.
    """
    placed, status = place_emergency_call()
    log.warning("emergency_support_invoked", call_placed=placed, region=region or "default")

    resources = safety.format_resources(region or None)
    return (
        "I'm concerned about you and I want to make sure you're not alone with this.\n\n"
        "**Please reach out to one of these now:**\n\n"
        f"{resources}\n\n"
        f"({status})"
    )


# --- Registry -----------------------------------------------------------------------

ALL_TOOLS = [
    ask_mental_health_specialist,
    ask_exam_stress_specialist,
    find_nearby_therapists,
    guided_coping_exercise,
    emergency_support,
]


def load_crisis_resources_file() -> dict:
    path: Path = get_settings().crisis_resources_path
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
