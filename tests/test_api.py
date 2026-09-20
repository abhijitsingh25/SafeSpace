from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from backend import agent as agent_module
from backend.main import app


class FakeGraph:
    """Stands in for the compiled LangGraph, recording thread ids so memory is testable."""

    def __init__(self, reply="I hear you. What's been hardest?", tool=None):
        self.reply = reply
        self.tool = tool
        self.threads: list[str] = []
        self.messages_seen: list[str] = []

    def _record(self, payload, config):
        self.threads.append(config["configurable"]["thread_id"])
        self.messages_seen.append(payload["messages"][0][1])

    def invoke(self, payload, config=None):
        self._record(payload, config)
        messages = []
        if self.tool:
            messages.append(
                AIMessage(content="", tool_calls=[{"name": self.tool, "args": {}, "id": "1"}])
            )
        messages.append(AIMessage(content=self.reply))
        return {"messages": messages}

    def stream(self, payload, config=None, stream_mode=None):
        self._record(payload, config)
        if self.tool:
            msg = AIMessage(content="tool output")
            msg.name = self.tool
            yield {"tools": {"messages": [msg]}}
        yield {"agent": {"messages": [AIMessage(content=self.reply)]}}


@pytest.fixture
def graph():
    fake = FakeGraph()
    agent_module.set_graph(fake)
    yield fake
    agent_module.set_graph(None)


@pytest.fixture
def client(graph):
    with TestClient(app) as test_client:
        yield test_client


# --- Health -------------------------------------------------------------------------


def test_health_reports_readiness(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["corpus_loaded"] is True
    assert body["corpus_passages"] == 350
    assert body["emergency_calls_enabled"] is False


def test_crisis_resources_endpoint_needs_no_message(client):
    body = client.get("/crisis-resources", params={"region": "US"}).json()
    assert any("988" in line["contact"] for line in body["lines"])


# --- Normal turns -------------------------------------------------------------------


def test_ask_returns_a_reply_and_a_session(client, graph):
    body = client.post("/ask", json={"message": "I feel low today"}).json()
    assert body["response"] == graph.reply
    assert body["risk_level"] == "NONE"
    assert body["crisis"] is False
    assert body["session_id"]
    assert "not a therapist" in body["disclaimer"].lower()


def test_session_id_is_threaded_through_to_the_graph(client, graph):
    """This is what gives the conversation memory - it had none before."""
    client.post("/ask", json={"message": "my name is Sam", "session_id": "sess-1"})
    client.post("/ask", json={"message": "what did I say?", "session_id": "sess-1"})
    assert graph.threads == ["sess-1", "sess-1"]


def test_absent_session_id_generates_a_fresh_one(client, graph):
    first = client.post("/ask", json={"message": "hello"}).json()["session_id"]
    second = client.post("/ask", json={"message": "hello"}).json()["session_id"]
    assert first != second


def test_empty_message_is_rejected(client):
    assert client.post("/ask", json={"message": ""}).status_code == 422


def test_overlong_message_is_rejected(client):
    assert client.post("/ask", json={"message": "x" * 5000}).status_code == 422


def test_agent_failure_returns_503_not_a_stack_trace(client):
    class Broken:
        def invoke(self, *a, **k):
            raise RuntimeError("groq is down")

    agent_module.set_graph(Broken())
    response = client.post("/ask", json={"message": "hello"})
    assert response.status_code == 503
    assert "try again" in response.json()["detail"]


# --- Crisis path --------------------------------------------------------------------


def test_crisis_message_bypasses_the_agent_entirely(client, graph):
    body = client.post("/ask", json={"message": "I want to kill myself", "region": "US"}).json()

    assert body["crisis"] is True
    assert body["risk_level"] == "CRISIS"
    assert body["tool_called"] == "crisis_protocol"
    assert "988" in body["response"]
    # The agent must not have run: crisis copy is fixed, never model-generated.
    assert graph.threads == []


def test_crisis_response_is_localised(client):
    body = client.post("/ask", json={"message": "I want to die", "region": "IN"}).json()
    assert "14416" in body["response"]


def test_third_party_concern_does_not_trigger_the_crisis_bypass(client, graph):
    body = client.post("/ask", json={"message": "my friend wants to kill herself"}).json()
    assert body["crisis"] is False
    assert body["risk_level"] == "ELEVATED"
    assert graph.threads, "the agent should still handle this turn"


def test_elevated_turns_append_hotlines_to_the_agent_reply(client, graph):
    body = client.post(
        "/ask", json={"message": "I feel completely hopeless", "region": "US"}
    ).json()
    assert body["risk_level"] == "ELEVATED"
    assert body["response"].startswith(graph.reply)
    assert "988" in body["response"]


def test_past_tense_risk_is_not_escalated(client, graph):
    body = client.post(
        "/ask", json={"message": "I used to feel suicidal but I'm better now"}
    ).json()
    assert body["crisis"] is False
    assert graph.threads


# --- Streaming ----------------------------------------------------------------------


def _events(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_stream_emits_session_then_final(client, graph):
    response = client.post("/ask/stream", json={"message": "hi there"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _events(response.text)
    assert events[0][0] == "session"
    assert events[-1][0] == "final"
    assert events[-1][1]["response"] == graph.reply


def test_stream_reports_the_tool_that_ran(client):
    agent_module.set_graph(FakeGraph(tool="guided_coping_exercise"))
    events = _events(client.post("/ask/stream", json={"message": "I'm panicking"}).text)
    kinds = [kind for kind, _ in events]
    assert "tool" in kinds
    assert dict(events)["tool"]["tool"] == "guided_coping_exercise"


def test_stream_crisis_path_skips_the_agent(client, graph):
    events = _events(
        client.post("/ask/stream", json={"message": "I want to kill myself", "region": "US"}).text
    )
    final = dict(events)["final"]
    assert final["crisis"] is True
    assert "988" in final["response"]
    assert graph.threads == []


# --- Rate limiting ------------------------------------------------------------------


def test_rate_limit_returns_429_but_still_gives_hotlines(client, monkeypatch):
    from backend import main as main_module
    from backend.settings import get_settings

    main_module._request_times.clear()
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2, raising=False)

    client.post("/ask", json={"message": "one"})
    client.post("/ask", json={"message": "two"})
    response = client.post("/ask", json={"message": "three"})

    assert response.status_code == 429
    # Throttling must never be a reason someone in distress sees nothing useful.
    assert response.json()["crisis_resources"]["lines"]
    main_module._request_times.clear()
