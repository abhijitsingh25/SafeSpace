"""Deterministic safety layer.

The original build left crisis detection entirely to the agent LLM's discretion via a
single line in the system prompt. That fails open: a jailbreak, an unusual phrasing, or
one unlucky sample and nobody escalates. This module runs *before* the agent on every
inbound message and, on CRISIS, bypasses the agent completely in favour of a fixed,
human-reviewed response.

Design rules:
  * The deterministic tier is authoritative for escalation. The LLM tier can only raise
    the level, never lower it.
  * False positives are cheap (a user sees a hotline they didn't need). False negatives
    are not. Every ambiguity resolves upward.
  * Crisis copy is never model-generated. It is a template with verified hotline data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import structlog

from backend.settings import get_settings

log = structlog.get_logger(__name__)


class RiskLevel(IntEnum):
    """Ordered so that ``max()`` composes tiers correctly."""

    NONE = 0
    ELEVATED = 1
    CRISIS = 2


# --- Pattern tiers -----------------------------------------------------------------
# Explicit intent toward ending one's life or self-harm.
#
# Two details that the eval set caught and that are easy to get wrong:
#   * "kms" is also an abbreviation for kilometres, so a preceding number disqualifies it.
#   * A non-first-person reflexive ("kill herself") means the subject is someone else.
#     Those are captured in the `refl` group and downgraded, rather than missed entirely
#     as they were when the patterns only matched "myself".
_REFLEXIVE = r"(?P<refl>my|him|her|them|your|one)\s?sel(?:f|ves)"

_CRISIS_PATTERNS: tuple[tuple[str, str], ...] = (
    (rf"\bkill(?:ing)?\s+{_REFLEXIVE}\b", "kill_self"),
    (r"\bend(?:ing)?\s+(?:my|his|her|their)\s+life\b", "end_life"),
    (r"\bend(?:ing)?\s+it\s+all\b", "end_it_all"),
    (r"\btake\s+(?:my|his|her|their)\s+own\s+life\b", "take_own_life"),
    (r"\bsuicid(?:e|al)\b", "suicide"),
    (r"\bwants?\s+to\s+die\b", "want_to_die"),
    (r"\bwish(?:es)?\s+(?:i|he|she|they)\s+(?:were|was)\s+dead\b", "wish_dead"),
    (r"\bbetter\s+off\s+dead\b", "better_off_dead"),
    (
        r"\bdo(?:n'?t|es\s?n'?t)\s+want\s+to\s+(?:be\s+here|live|wake\s+up)\b",
        "dont_want_to_live",
    ),
    (rf"\b(?:hurt|harm)(?:ing)?\s+{_REFLEXIVE}\b", "self_harm"),
    (rf"\bcut(?:ting)?\s+{_REFLEXIVE}\b", "cutting"),
    (r"\boverdos(?:e|ing)\b", "overdose"),
    (rf"\bhang(?:ing)?\s+{_REFLEXIVE}\b", "hanging"),
    (r"\bjump\s+off\s+(?:a|the)\b", "jumping"),
    (r"\bslit\s+(?:my|his|her|their)\s+wrists?\b", "slit_wrists"),
    # Negative lookbehinds keep "5 kms" and "5kms" out of the crisis tier.
    (r"(?<!\d)(?<!\d )\bkms\b", "kms_slang"),
    (r"\bunalive\b", "unalive_slang"),
)

# Distress without stated intent. Warrants resources and a gentle check-in, not a bypass.
_ELEVATED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhopeless\b", "hopeless"),
    (
        r"\bcan'?t\s+(?:go\s+on|take\s+(?:it|this)\s+any\s?more|do\s+this\s+any\s?more)\b",
        "cant_go_on",
    ),
    (r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living|going\s+on)\b", "no_reason"),
    (r"\bnobody\s+(?:would|will)\s+(?:miss|care)\b", "nobody_cares"),
    (r"\bi\s+(?:am|'m|feel)\s+(?:worthless|a\s+burden)\b", "worthless"),
    (r"\bgive\s+up\s+on\s+(?:everything|life)\b", "give_up"),
    (r"\bempty\s+inside\b", "empty"),
    (r"\bself[\s-]?harm\b", "self_harm_topic"),
)

# Markers that the statement is in the past or negated. Checked in a window around the
# match, not across the whole message, so a later clause can't mask a present-tense one.
_RESOLVED_MARKERS = (
    r"used\s+to",
    r"no\s+longer",
    r"not\s+any\s?more",
    r"in\s+the\s+past",
    r"years?\s+ago",
    r"months?\s+ago",
    r"back\s+then",
    r"when\s+i\s+was\s+(?:younger|a\s+kid|a\s+teen)",
    r"never\s+(?:again|would|want)",
    r"don'?t\s+(?:feel|think)\s+(?:that|like\s+that)\s+any\s?more",
    r"i'?m\s+(?:better|okay|ok|fine|past)\s+(?:now|that)",
    r"glad\s+i\s+didn'?t",
    r"recovered\s+from",
    r"got\s+help\s+for",
)

# Markers that the subject is someone other than the user.
_THIRD_PARTY_MARKERS = (
    r"my\s+(?:friend|brother|sister|mom|mum|mother|dad|father|son|daughter|partner|"
    r"boyfriend|girlfriend|wife|husband|cousin|colleague|classmate|student|roommate)",
    r"someone\s+i\s+know",
    r"a\s+friend\s+of\s+mine",
    r"my\s+(?:friend|child|kid)s",
    r"\b(?:he|she|they)\s+(?:said|told|is|was|are|were|wants?|needs?|feels?|has|have|keeps?|"
    r"might|may|started|been)\b",
)

_WINDOW_CHARS = 90

_CRISIS_RE = [(re.compile(p, re.IGNORECASE), name) for p, name in _CRISIS_PATTERNS]
_ELEVATED_RE = [(re.compile(p, re.IGNORECASE), name) for p, name in _ELEVATED_PATTERNS]
_RESOLVED_RE = re.compile("|".join(_RESOLVED_MARKERS), re.IGNORECASE)
_THIRD_PARTY_RE = re.compile("|".join(_THIRD_PARTY_MARKERS), re.IGNORECASE)


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    rules: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    third_party: bool = False

    @property
    def is_crisis(self) -> bool:
        return self.level is RiskLevel.CRISIS

    def to_dict(self) -> dict:
        return {
            "level": self.level.name,
            "rules": list(self.rules),
            "reason": self.reason,
            "third_party": self.third_party,
        }


def _context_window(text: str, start: int, end: int) -> str:
    return text[max(0, start - _WINDOW_CHARS) : min(len(text), end + _WINDOW_CHARS)]


def classify_deterministic(message: str) -> RiskAssessment:
    """Rule-based tier. Fast, offline, and the authority on escalation."""
    if not message or not message.strip():
        return RiskAssessment(RiskLevel.NONE, reason="empty message")

    text = message.strip()
    crisis_hits: list[str] = []
    downgraded: list[str] = []
    third_party = False

    for pattern, name in _CRISIS_RE:
        for match in pattern.finditer(text):
            window = _context_window(text, match.start(), match.end())

            if _RESOLVED_RE.search(window):
                downgraded.append(f"{name}:past")
                continue

            # A reflexive pronoun that isn't first-person settles the question outright -
            # "kill herself" is about someone else no matter what surrounds it.
            groups = match.groupdict()
            reflexive = groups.get("refl")
            if reflexive and reflexive.lower() not in {"my", "one"}:
                downgraded.append(f"{name}:third_party")
                third_party = True
                continue

            if _THIRD_PARTY_RE.search(window):
                downgraded.append(f"{name}:third_party")
                third_party = True
                continue

            crisis_hits.append(name)

    if crisis_hits:
        return RiskAssessment(
            RiskLevel.CRISIS,
            rules=tuple(dict.fromkeys(crisis_hits)),
            reason="explicit risk language in first person, present tense",
        )

    elevated_hits = [name for pattern, name in _ELEVATED_RE if pattern.search(text)]
    if downgraded or elevated_hits:
        return RiskAssessment(
            RiskLevel.ELEVATED,
            rules=tuple(dict.fromkeys(downgraded + elevated_hits)),
            reason="distress language, or risk language that is past-tense/third-party",
            third_party=third_party,
        )

    return RiskAssessment(RiskLevel.NONE)


_LLM_CLASSIFIER_PROMPT = """You are a triage classifier for a mental health chat service.
Classify the user's most recent message into exactly one label:

CRISIS   - the user expresses current suicidal intent, a plan, or intent to self-harm.
ELEVATED - significant distress, hopelessness, or past/third-party risk, without current
           first-person intent.
NONE     - anything else.

Rules: judge only the user's own current state. Past-tense recovery ("I used to feel
that way") and concern about another person are ELEVATED, not CRISIS.

Reply with one word: CRISIS, ELEVATED, or NONE."""


def classify_with_llm(message: str, llm) -> RiskLevel:
    """Second tier for phrasings the lexicon misses. Never lowers the rule-based verdict."""
    try:
        result = llm.invoke([("system", _LLM_CLASSIFIER_PROMPT), ("user", message)])
        raw = getattr(result, "content", str(result)).strip().upper()
        for level in (RiskLevel.CRISIS, RiskLevel.ELEVATED):
            if level.name in raw:
                return level
        return RiskLevel.NONE
    except Exception as exc:  # pragma: no cover - defensive
        # A classifier outage must not silently downgrade anyone.
        log.warning("llm_risk_classifier_failed", error=str(exc))
        return RiskLevel.NONE


def assess(message: str, llm=None) -> RiskAssessment:
    """Compose both tiers. The result is the *maximum* risk either tier reports."""
    deterministic = classify_deterministic(message)

    # Skip the paid round-trip when the rules already escalated as far as they can.
    if deterministic.level is RiskLevel.CRISIS or llm is None:
        return deterministic

    llm_level = classify_with_llm(message, llm)
    if llm_level > deterministic.level:
        return RiskAssessment(
            llm_level,
            rules=(*deterministic.rules, "llm_classifier"),
            reason="escalated by LLM triage tier",
            third_party=deterministic.third_party,
        )
    return deterministic


# --- Crisis response ---------------------------------------------------------------


@lru_cache
def _load_resources() -> dict:
    path: Path = get_settings().crisis_resources_path
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_crisis_resources(region: str | None = None) -> dict:
    """Return the hotline block for a region, falling back to the international list."""
    data = _load_resources()["regions"]
    key = (region or get_settings().default_crisis_region or "INTL").upper()
    return data.get(key, data["INTL"])


def format_resources(region: str | None = None) -> str:
    block = get_crisis_resources(region)
    lines = "\n".join(f"- **{line['name']}** — {line['contact']}" for line in block["lines"])
    return lines


def build_crisis_response(region: str | None = None, third_party: bool = False) -> str:
    """Fixed, reviewed copy. Deliberately not generated by a model."""
    resources = format_resources(region)

    if third_party:
        return (
            "It sounds like you're worried about someone you care about, and that takes a "
            "lot of compassion. You don't have to carry that alone.\n\n"
            "If you believe they may be in immediate danger, please contact emergency "
            "services or one of these lines — they support people who are worried about "
            "someone else too:\n\n"
            f"{resources}\n\n"
            "What you can do right now: stay with them if you can, listen without trying to "
            "fix it, ask directly whether they're thinking about suicide (asking does not "
            "plant the idea), and help them connect with one of the services above.\n\n"
            "How are *you* holding up through this?"
        )

    return (
        "I'm really glad you told me, and I want to be honest with you: what you're "
        "describing sounds serious, and I'm not able to give you the kind of help you "
        "deserve right now. A trained person can.\n\n"
        "**Please reach out to one of these now — they're free, confidential, and open "
        "right now:**\n\n"
        f"{resources}\n\n"
        "If you're in immediate danger, please call your local emergency number.\n\n"
        "You don't have to explain it perfectly or have the right words. Telling someone "
        "you're struggling is enough to start. If you're able to, try not to be alone "
        "right now — is there someone nearby you could sit with while you make that call?"
    )


DISCLAIMER = (
    "SafeSpace is an AI support tool, not a therapist, and does not provide medical "
    "advice, diagnosis, or treatment. In an emergency, contact your local emergency services."
)

_ELEVATED_FOOTER_TEMPLATE = (
    "\n\n---\n_If things get heavier, you can reach a trained human any time:_\n{resources}"
)


def elevated_footer(region: str | None = None) -> str:
    return _ELEVATED_FOOTER_TEMPLATE.format(resources=format_resources(region))


def hash_text(text: str) -> str:
    """Stable, non-reversible id for audit logs, so we never log raw message content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def log_assessment(assessment: RiskAssessment, message: str, session_id: str) -> None:
    """Audit trail for every non-NONE assessment. Content is hashed, never stored raw."""
    if assessment.level is RiskLevel.NONE:
        return
    log.warning(
        "risk_assessed",
        level=assessment.level.name,
        rules=list(assessment.rules),
        third_party=assessment.third_party,
        session_id=session_id,
        message_hash=hash_text(message),
        message_length=len(message),
    )
