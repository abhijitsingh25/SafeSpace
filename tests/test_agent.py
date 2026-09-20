from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agent import SYSTEM_PROMPT, extract_reply
from backend.providers.llm import FALLBACK_MESSAGE


def test_extract_reply_picks_the_final_answer_not_an_intermediate_thought():
    """The old parser kept the last message with *any* content, including pre-tool chatter."""
    state = {
        "messages": [
            HumanMessage(content="I'm stressed about exams"),
            AIMessage(
                content="Let me look that up.",
                tool_calls=[{"name": "ask_exam_stress_specialist", "args": {}, "id": "1"}],
            ),
            ToolMessage(
                content="grounded advice",
                tool_call_id="1",
                name="ask_exam_stress_specialist",
            ),
            AIMessage(content="That sounds exhausting. What's the exam?"),
        ]
    }
    reply = extract_reply(state)
    assert reply.response == "That sounds exhausting. What's the exam?"
    assert reply.tool_called == "ask_exam_stress_specialist"


def test_extract_reply_ignores_a_message_that_is_only_requesting_a_tool():
    state = {
        "messages": [
            AIMessage(content="thinking...", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
        ]
    }
    assert extract_reply(state).response == FALLBACK_MESSAGE


def test_extract_reply_never_returns_none():
    """`None` used to reach the UI and render as the literal string 'None'."""
    assert extract_reply({"messages": []}).response == FALLBACK_MESSAGE
    assert extract_reply({}).response == FALLBACK_MESSAGE


def test_extract_reply_reports_no_tool_for_a_direct_answer():
    state = {"messages": [AIMessage(content="I'm SafeSpace, a support assistant.")]}
    reply = extract_reply(state)
    assert reply.tool_called is None
    assert reply.response.startswith("I'm SafeSpace")


def test_system_prompt_states_a_routing_policy():
    assert "ROUTING POLICY" in SYSTEM_PROMPT
    assert "exactly one tool" in SYSTEM_PROMPT


def test_system_prompt_forbids_fabricated_referrals():
    assert "Never invent names, phone numbers" in SYSTEM_PROMPT


def test_system_prompt_does_not_miscount_tools():
    """It used to announce 'three tools' above a list of four."""
    assert "three tools" not in SYSTEM_PROMPT.lower()


def test_tool_attribution_is_scoped_to_the_current_turn():
    """With memory, invoke() returns the whole conversation - an earlier turn's tool
    must not be reported as this turn's."""
    state = {
        "messages": [
            HumanMessage(content="find me a therapist in Bangalore"),
            AIMessage(
                content="",
                tool_calls=[{"name": "find_nearby_therapists", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="listings", tool_call_id="1", name="find_nearby_therapists"),
            AIMessage(content="Here are some options."),
            HumanMessage(content="what have I told you so far?"),
            AIMessage(content="You mentioned looking for a therapist."),
        ]
    }
    reply = extract_reply(state)
    assert reply.response == "You mentioned looking for a therapist."
    assert reply.tool_called is None


def test_tool_attribution_still_works_within_a_turn_that_used_one():
    state = {
        "messages": [
            HumanMessage(content="old"),
            AIMessage(content="old answer"),
            HumanMessage(content="I'm panicking"),
            AIMessage(
                content="",
                tool_calls=[{"name": "guided_coping_exercise", "args": {}, "id": "2"}],
            ),
            ToolMessage(content="box breathing", tool_call_id="2", name="guided_coping_exercise"),
            AIMessage(content="Try box breathing."),
        ]
    }
    reply = extract_reply(state)
    assert reply.tool_called == "guided_coping_exercise"
    assert reply.response == "Try box breathing."
