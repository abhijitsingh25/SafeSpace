"""SafeSpace HTTP API.

Request path for every turn:

    inbound message
      -> deterministic risk assessment          (backend.safety)
      -> CRISIS?  return fixed crisis copy, agent never runs
      -> otherwise run the agent with memory    (backend.agent)
      -> ELEVATED? append hotline footer
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend import safety
from backend.agent import run_agent, stream_agent
from backend.logging_config import configure_logging
from backend.retrieval import load_index
from backend.settings import REPO_ROOT, get_settings

log = structlog.get_logger(__name__)


# --- Schemas ------------------------------------------------------------------------


class Query(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(
        default=None,
        description="Opaque, client-generated conversation id. Enables memory.",
    )
    region: str | None = Field(
        default=None, max_length=4, description="Country code for crisis resources."
    )


class Answer(BaseModel):
    response: str
    tool_called: str | None = None
    session_id: str
    risk_level: str
    crisis: bool = False
    disclaimer: str = safety.DISCLAIMER


# --- Rate limiting ------------------------------------------------------------------
# In-process and therefore per-instance. Good enough for a single server; on Vercel this
# needs to move to Upstash Redis to be meaningful across instances (see README).
_request_times: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(client_key: str, limit: int) -> bool:
    now = time.monotonic()
    window = _request_times[client_key]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        return True
    window.append(now)
    return False


# --- App ----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    log.info(
        "safespace_starting",
        environment=settings.environment,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        emergency_calls_enabled=settings.emergency_calls_enabled,
    )

    # Warm the corpus at startup so request #1 doesn't pay for the load.
    try:
        index = load_index()
        log.info("corpus_warm", passages=len(index))
    except Exception as exc:
        # Retrieval is one feature, not the whole app - degrade rather than fail to boot.
        log.error("corpus_warm_failed", error=str(exc))

    # Crisis resources must parse at boot. If they don't, refuse to start: serving the
    # app without working hotline data is worse than being down.
    safety.get_crisis_resources()

    yield
    log.info("safespace_stopping")


app = FastAPI(
    title="SafeSpace API",
    version="0.2.0",
    description="Agentic mental health support with deterministic crisis guardrails.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def bind_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    started = time.monotonic()
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.unbind_contextvars("request_id", "path")
    response.headers["x-request-id"] = request_id
    log.info("request_complete", duration_ms=round((time.monotonic() - started) * 1000))
    return response


# --- Routes -------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    settings = get_settings()
    try:
        passages = len(load_index())
        corpus_ok = True
    except Exception:
        passages, corpus_ok = 0, False

    return {
        "status": "ok",
        "version": app.version,
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "corpus_loaded": corpus_ok,
        "corpus_passages": passages,
        "emergency_calls_enabled": settings.emergency_calls_enabled,
        "therapist_lookup_configured": bool(settings.serpapi_api_key),
    }


@app.get("/crisis-resources")
async def crisis_resources(region: str | None = None) -> dict:
    """Hotlines, available without sending a message. Useful for a always-visible UI link."""
    return safety.get_crisis_resources(region)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _preflight(query: Query, request: Request) -> tuple[str, safety.RiskAssessment] | JSONResponse:
    settings = get_settings()

    if _rate_limited(_client_key(request), settings.rate_limit_per_minute):
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many messages in a short time. Please wait a moment.",
                # Even a throttled user gets the hotlines.
                "crisis_resources": safety.get_crisis_resources(query.region),
            },
        )

    session_id = query.session_id or uuid.uuid4().hex
    assessment = safety.assess(query.message)
    safety.log_assessment(assessment, query.message, session_id)
    return session_id, assessment


@app.post("/ask", response_model=Answer)
async def ask(query: Query, request: Request):
    pre = _preflight(query, request)
    if isinstance(pre, JSONResponse):
        return pre
    session_id, assessment = pre

    # CRISIS bypasses the agent entirely. The response is fixed, reviewed copy - not
    # something a model composes under pressure.
    if assessment.is_crisis:
        from backend.tools import place_emergency_call

        placed, status = await run_in_threadpool(place_emergency_call)
        log.warning("crisis_bypass", session_id=session_id, call_placed=placed, status=status)
        return Answer(
            response=safety.build_crisis_response(query.region, assessment.third_party),
            tool_called="crisis_protocol",
            session_id=session_id,
            risk_level=assessment.level.name,
            crisis=True,
        )

    try:
        # run_in_threadpool keeps the blocking LangGraph call off the event loop. The
        # previous `async def` calling `graph.stream()` directly served one user at a time.
        reply = await run_in_threadpool(run_agent, query.message, session_id)
    except Exception as exc:
        log.exception("agent_failed", session_id=session_id, error=str(exc))
        return JSONResponse(
            status_code=503,
            content={
                "detail": "I couldn't reach my words just now - a technical problem on "
                "my end. Please try again in a moment.",
                "session_id": session_id,
            },
        )

    response_text = reply.response
    if assessment.level is safety.RiskLevel.ELEVATED:
        response_text += safety.elevated_footer(query.region)

    return Answer(
        response=response_text,
        tool_called=reply.tool_called,
        session_id=session_id,
        risk_level=assessment.level.name,
    )


@app.post("/ask/stream")
async def ask_stream(query: Query, request: Request):
    """Server-sent events.

    Required rather than optional on Vercel: a buffered response can exceed the function
    timeout, while a stream keeps bytes flowing.
    """
    pre = _preflight(query, request)
    if isinstance(pre, JSONResponse):
        return pre
    session_id, assessment = pre

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def generate():
        yield sse("session", {"session_id": session_id, "risk_level": assessment.level.name})

        if assessment.is_crisis:
            from backend.tools import place_emergency_call

            placed, status = await run_in_threadpool(place_emergency_call)
            log.warning("crisis_bypass", session_id=session_id, call_placed=placed)
            yield sse(
                "final",
                {
                    "response": safety.build_crisis_response(query.region, assessment.third_party),
                    "tool_called": "crisis_protocol",
                    "crisis": True,
                },
            )
            return

        # Pull the blocking generator one item at a time in the threadpool, so events
        # reach the client as they happen instead of after the whole turn completes.
        events = stream_agent(query.message, session_id)
        done = object()

        def next_event():
            try:
                return next(events)
            except StopIteration:
                return done

        while True:
            try:
                item = await run_in_threadpool(next_event)
            except Exception as exc:
                log.exception("agent_stream_failed", session_id=session_id, error=str(exc))
                yield sse("error", {"detail": "Something went wrong. Please try again."})
                return

            if item is done:
                return

            kind, payload = item
            if kind == "tool":
                yield sse("tool", {"tool": payload})
            else:
                if assessment.level is safety.RiskLevel.ELEVATED:
                    payload["response"] += safety.elevated_footer(query.region)
                payload["crisis"] = False
                yield sse("final", payload)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the static chat UI locally. On Vercel these files are served by the CDN instead.
_web_dir = REPO_ROOT / "web"
if _web_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
