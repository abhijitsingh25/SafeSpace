"""The LangGraph agent, its routing policy, and conversation memory.

Two things changed materially from the original ``ai_agent.py``:

1. **The system prompt names tools that actually exist.** It previously referenced
   ``locate_therapist_tool`` and ``ask_exam_rag_tool``, neither of which was registered,
   and announced "three tools" above a list of four. The model was being asked to route
   to names it had never been given.
2. **Conversations have memory.** A checkpointer keyed by ``thread_id`` means the agent
   can remember what someone told it thirty seconds ago - which, for something presenting
   itself as a therapist, is the difference between a product and a demo.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.providers.llm import FALLBACK_MESSAGE
from backend.settings import get_settings
from backend.tools import ALL_TOOLS

log = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are SafeSpace, a mental health support assistant. You are warm,
unhurried, and you listen more than you advise.

ROUTING POLICY - pick exactly one tool per turn:

- Exams, studying, results, grades, academic burnout, fear of failing, or pressure from
  school, university, teachers or parents about academic performance
  -> ask_exam_stress_specialist

- A request for a therapist, counsellor, or professional help near a place
  -> find_nearby_therapists. If you do not know where they are, ask first.

- Acute anxiety, panic, feeling overwhelmed or detached, or a request for something
  practical to do right now
  -> guided_coping_exercise

- Suicidal thoughts, intent to self-harm, or immediate danger
  -> emergency_support, immediately, before anything else

- Everything else emotional or psychological
  -> ask_mental_health_specialist

HOW TO WRITE:
- Reply in the tool's voice. Do not summarise the tool output or mention that a tool ran.
- Never invent names, phone numbers, clinics, or credentials. If a tool did not give you
  a real referral, say you could not look one up.
- Never diagnose, and never give medication advice.
- Refer back to what the person told you earlier in the conversation. It matters to them
  that you remembered.
- If someone is just chatting or asking who you are, answer directly and briefly without
  calling any tool.
"""


@dataclass
class AgentReply:
    response: str
    tool_called: str | None = None


_graph = None
_checkpointer = None


def get_checkpointer():
    """Conversation memory store.

    ``MemorySaver`` is per-process. That is correct locally and behind a single long-lived
    server, but on Vercel each invocation may hit a cold instance, so memory will be
    lossy there. Set ``DATABASE_URL`` and swap in ``PostgresSaver`` for durable memory -
    see README "Deploying to Vercel".
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def build_graph(model=None):
    settings = get_settings()

    if model is None:
        from langchain_groq import ChatGroq

        kwargs = {
            "model": settings.groq_model,
            "api_key": settings.require_agent_key(),
            "temperature": settings.agent_temperature,
            "max_tokens": settings.agent_max_tokens,
            "model_kwargs": {"top_p": settings.agent_top_p},
        }
        # Only reasoning models accept these, and Groq 400s if they are sent to a model
        # that does not support them.
        if settings.agent_reasoning_effort:
            kwargs["reasoning_effort"] = settings.agent_reasoning_effort
        if settings.agent_reasoning_format:
            kwargs["reasoning_format"] = settings.agent_reasoning_format

        model = ChatGroq(**kwargs)

    return create_react_agent(
        model,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=get_checkpointer(),
    )


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("agent_graph_built", tools=[t.name for t in ALL_TOOLS])
    return _graph


def set_graph(graph) -> None:
    """Test seam."""
    global _graph
    _graph = graph


def extract_reply(state: dict) -> AgentReply:
    """Pull the user-facing answer out of a finished graph state.

    The old implementation kept the last message with *any* content, which could be an
    intermediate thought preceding a tool call. The final answer is the last AIMessage
    that has content and is not itself requesting a tool.
    """
    messages = state.get("messages", []) if isinstance(state, dict) else []

    # With a checkpointer, invoke() returns the entire accumulated conversation, not
    # just this turn. Without this trim, `tool_called` reports a tool from an earlier
    # turn on any turn that called none.
    last_human = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human = index
    messages = messages[last_human + 1 :]

    tool_called: str | None = None
    final: str | None = None

    for message in messages:
        calls = getattr(message, "tool_calls", None)
        if calls:
            tool_called = calls[-1].get("name") if isinstance(calls[-1], dict) else None
        if isinstance(message, AIMessage) and message.content and not calls:
            final = message.content if isinstance(message.content, str) else str(message.content)

    return AgentReply(response=(final or FALLBACK_MESSAGE).strip(), tool_called=tool_called)


def run_agent(message: str, session_id: str) -> AgentReply:
    """Run one turn. Blocking - callers must keep it off the event loop."""
    config = {"configurable": {"thread_id": session_id}}
    state = get_graph().invoke({"messages": [("user", message)]}, config=config)
    reply = extract_reply(state)
    log.info("agent_turn_complete", session_id=session_id, tool_called=reply.tool_called)
    return reply


def stream_agent(message: str, session_id: str):
    """Yield ``(kind, payload)`` events for SSE.

    ``kind`` is one of ``tool`` or ``final``.
    """
    config = {"configurable": {"thread_id": session_id}}
    final: str | None = None
    tool_called: str | None = None

    for chunk in get_graph().stream(
        {"messages": [("user", message)]}, config=config, stream_mode="updates"
    ):
        for node, update in chunk.items():
            for message_obj in (update or {}).get("messages", []) or []:
                name = getattr(message_obj, "name", None)
                if node == "tools" and name:
                    tool_called = name
                    yield ("tool", name)
                calls = getattr(message_obj, "tool_calls", None)
                if isinstance(message_obj, AIMessage) and message_obj.content and not calls:
                    final = message_obj.content

    yield ("final", {"response": (final or FALLBACK_MESSAGE).strip(), "tool_called": tool_called})
