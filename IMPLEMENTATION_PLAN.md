# SafeSpace — Gap Analysis & Implementation Plan

_Original analysis: 2026-09-20, commit `6ce18f3`. **Phases 0–3 and 5.1–5.6 are now
implemented**; see the status table below and the README for how to run it._

## Implementation status

| Phase | Status | Notes |
|---|---|---|
| **0 — Make it run** | Done | `pydantic-settings`, `.env.example`, real package layout, fixed index builder, `/health`, task runners |
| **1 — Safety** | Done | Deterministic classifier + crisis bypass, localised hotlines, guarded Twilio, fabricated referrals removed, UI disclaimer, eval set in CI |
| **2 — Correctness** | Done | Memory via checkpointer, prompt/tool names fixed, event loop unblocked, hardened frontend, structlog, `extract_reply` rewritten, 118 tests + CI |
| **3 — Efficiency** | Done | FAISS→numpy, provider abstractions, startup warmup, SSE streaming |
| **3.6 — Caching** | Not done | Deferred with 4.1; needs the same managed store |
| **4 — Features** | 4.3 only | `guided_coping_exercise` shipped. The rest need persistence + an identity decision |
| **5 — Vercel** | Config ready | `vercel.json`, `api/index.py`, prod `requirements.txt`, bundle-size CI gate. Not yet deployed |

**Known gaps carried forward** (both documented in the README): conversation memory and
rate limiting are per-instance, so serverless cold starts lose threads and the effective
rate limit scales with instance count. Both need a managed store — Postgres for the
checkpointer, Redis for the counter.

---

## Original analysis

### 1. Verdict up front

The repo is a working prototype of a good idea, but **it does not currently run from a clean
clone**, and the two things that make it interesting locally (Ollama/MedGemma + local
sentence-transformers) are exactly the two things that **cannot be deployed to Vercel**.

So the work splits into three tracks:

| Track | Goal | Blocking? |
|---|---|---|
| **A. Make it run** | Fix hard blockers so `uv sync` + run works for anyone | Yes — blocks everything |
| **B. Make it safe & correct** | It's a mental-health app with a crisis path; treat that as priority one | Yes — blocks public deploy |
| **C. Make it deployable** | Replace local-only inference with hosted inference | Yes — blocks Vercel |

Features and efficiency work sit on top of those.

---

## 2. Gap analysis

### 2.1 Hard blockers — the app cannot start today

| # | Gap | Evidence |
|---|---|---|
| B1 ✅ | **`backend/config.py` does not exist** and is gitignored. Every entrypoint dies on import. | [.gitignore:11](.gitignore#L11), [backend/ai_agent.py:50](backend/ai_agent.py#L50), [backend/tools.py:54](backend/tools.py#L54) — `ls backend/` shows only 3 files |
| B2 ✅ | **`sentence-transformers` and `faiss` are in neither `pyproject.toml` nor `uv.lock`.** A clean `uv sync` produces an env where `backend/tools.py` fails at import. | [pyproject.toml:7-21](pyproject.toml#L7-L21) vs [backend/tools.py:7-8](backend/tools.py#L7-L8) |
| B3 ✅ | **`build_faiss_index.py` has a hardcoded path to a machine that isn't this one** (`C:\GenAi\SafeSpace\...`), and its `BASE_DIR` resolves one level too high — it would write embeddings to `C:\nexus\embeddings`, not the repo. | [build_faiss_index.py:9](build_faiss_index.py#L9), [build_faiss_index.py:12](build_faiss_index.py#L12) |
| B4 ✅ | **Imports are cwd-relative** (`from tools import ...`, `from config import ...`). Only works if you `cd backend` first. No package, no `__init__.py`. | [backend/ai_agent.py:3](backend/ai_agent.py#L3), [backend/main.py:7](backend/main.py#L7) |
| B5 ✅ | **README instructs `pip install -r requirements.txt`** — that file doesn't exist. It also says `python main.py` / `streamlit run app.py`; neither path is right. | [README.md:50-71](README.md#L50-L71) |

### 2.2 Safety gaps — highest priority after "it runs"

This is the part I'd fix before anything else, because the README advertises safety
machinery that **is not in the code**.

| # | Gap | Evidence |
|---|---|---|
| S1 ✅ | **There is no risk classifier.** README claims "Automatically detects suicidal intent or high-risk phrases." In reality crisis detection is left entirely to the LLM's discretion via a single prompt line. A jailbreak, a typo, a non-English phrasing, or a bad sample and the emergency path silently never fires. | [README.md:18](README.md#L18) vs [backend/ai_agent.py:76](backend/ai_agent.py#L76) |
| S2 ✅ | **The crisis path calls Twilio but tells the user nothing.** `emergency_call_tool` returns `None`, so the agent gets an empty ToolMessage and has no grounded content to respond with. **No hotline number is ever shown to the user.** A phone call to one preconfigured contact is not a substitute for surfacing 988 / local crisis lines in the reply. | [backend/ai_agent.py:16-22](backend/ai_agent.py#L16-L22), [backend/tools.py:56-62](backend/tools.py#L56-L62) |
| S3 ✅ | **`call_emergency()` has no error handling and no audit trail.** If Twilio 4xx's (bad creds, unverified number, no balance) the exception propagates into the agent loop and the crisis turn fails open. Nothing is logged. | [backend/tools.py:56-62](backend/tools.py#L56-L62) |
| S4 ✅ | **The therapist locator returns fabricated therapists and fake phone numbers**, presented to a distressed user as real referrals. README claims SerpAPI/Google Maps; there is no such call. This is the most user-harmful line in the repo. | [backend/ai_agent.py:35-40](backend/ai_agent.py#L35-L40) vs [README.md:28](README.md#L28) |
| S5 ✅ | **No disclaimer in the product surface.** The disclaimer lives only at the bottom of the README. The chat UI shows nothing. | [frontend.py:8-9](frontend.py#L8-L9) |
| S6 ✅ | **No "fallback prompts" / guardrails** as claimed in README. The only fallback is a generic except-string in `query_medgemma`. | [README.md:20](README.md#L20), [backend/tools.py:47-48](backend/tools.py#L47-L48) |
| S7 ⬜ | **No data-retention or PII posture** for what is, by definition, sensitive health-adjacent conversation content. No policy, no redaction, no opt-out. Becomes mandatory the moment you persist sessions (see 4.1) or deploy publicly. | — |

### 2.3 Correctness bugs

| # | Gap | Evidence |
|---|---|---|
| C1 ✅ | **No conversation memory.** Every request sends only `[system, current user message]`. The Streamlit UI keeps history locally and never transmits it, so the "therapist" forgets you mid-sentence — fatal for the product's core premise. | [backend/main.py:19](backend/main.py#L19), [frontend.py:23](frontend.py#L23) |
| C2 ✅ | **Tool names in `SYSTEM_PROMPT` don't match registered tools.** Prompt says `locate_therapist_tool` and `ask_exam_rag_tool`; actual names are `find_nearby_therapists_by_location` and `ask_exam_rag`. It also says "three tools" then lists four. This measurably degrades routing. | [backend/ai_agent.py:70-80](backend/ai_agent.py#L70-L80) vs [backend/ai_agent.py:53-58](backend/ai_agent.py#L53-L58) |
| C3 ✅ | **`ask_exam_rag_tool` is defined *after* the graph is built and is never registered** — dead code. The raw function `ask_exam_rag` is what's actually passed. | [backend/ai_agent.py:117-120](backend/ai_agent.py#L117-L120) |
| C4 ✅ | **`async def ask` calls blocking `graph.stream()`** — this blocks the event loop for the whole LLM round-trip, so the server handles one user at a time regardless of worker config. | [backend/main.py:18-22](backend/main.py#L18-L22) |
| C5 ✅ | **Frontend crashes on any backend error.** `response.json()["response"]` with no status check, no timeout, no try/except. A 500 or a cold backend throws `KeyError`/`ConnectionError` into the UI. | [frontend.py:23-25](frontend.py#L23-L25) |
| C6 ✅ | **`final_response` can be `None`** and is rendered verbatim as the string `"None"` to the user. | [backend/ai_agent.py:83](backend/ai_agent.py#L83), [frontend.py:25](frontend.py#L25) |
| C7 ✅ | **`except Exception as e` swallows `e` without logging** — debugging a MedGemma failure is guesswork. | [backend/tools.py:47-48](backend/tools.py#L47-L48) |
| C8 ✅ | **`parse_response` keeps the last message with any content**, including intermediate agent chatter; no guarantee it's the final answer. | [backend/ai_agent.py:95-101](backend/ai_agent.py#L95-L101) |
| C9 ⬜ | **Meta/corpus length mismatch risk**: `exam_meta.json` has 350 entries; the JSONL has 350 records across 349 newline-terminated lines (last line unterminated). Harmless today, a silent off-by-one the next time the index is rebuilt differently. | `wc -l` vs meta length |

### 2.4 Efficiency gaps

| # | Gap | Impact |
|---|---|---|
| E1 ✅ | **FAISS for 350 vectors.** The index is 537 KB — a `(350, 384)` matrix. A numpy dot product over normalized vectors is exact, faster than the FAISS call overhead at this scale, and deletes a ~30 MB dependency. `IndexFlatIP` is doing a brute-force scan anyway. | Removes a dep; no accuracy loss |
| E2 ✅ | **`torch` + `sentence-transformers` (~2 GB) loaded to embed one short string per request.** | Blocks Vercel entirely (250 MB limit) |
| E3 ✅ | **No model warmup.** `_ensure_index_loaded()` lazy-loads on first request, so user #1 eats a multi-second SentenceTransformer init. | First-request latency |
| E4 ✅ | **No streaming to the user.** The agent already streams internally (`graph.stream`), then `parse_response` throws the stream away and returns one blob. Users stare at a spinner for the full generation. | Perceived latency — the biggest UX win available |
| E5 ⬜ | **Two sequential LLM calls per RAG turn** (Groq agent → Ollama MedGemma), each with its own cold path. | Latency |
| E6 ✅ | **No caching anywhere** — no embedding cache, no response cache for common queries ("I'm stressed about exams" will be asked constantly). | Cost + latency |
| E7 ✅ | **No connection reuse** — a fresh `requests.post` per Streamlit interaction, no shared async client. | Minor |
| E8 ⬜ | **Corpus embeddings are recomputed from scratch** on every index build with no incremental path. | Dev-loop friction |

### 2.5 Engineering-hygiene gaps

No tests. No CI. No linting/formatting config. No Dockerfile. No `.env.example`. No structured
logging. No `/health` endpoint. No CORS config. No rate limiting. No auth. No request IDs.
No error tracking. Two entrypoints (`backend/main.py`, `frontend.py`) with no single command to
bring the stack up.

### 2.6 The Vercel problem — read this before planning the deploy

Three hard incompatibilities, each needing an architectural decision, not a config tweak:

1. **Ollama / MedGemma cannot run on Vercel.** Serverless functions have no GPU, no model
   weights, no persistent process. `backend/tools.py` calls `ollama.chat()` in two places
   ([tools.py:34](backend/tools.py#L34), [tools.py:128](backend/tools.py#L128)) — both must go
   behind a provider interface.
2. **`torch`/`sentence-transformers`/`faiss` exceed the 250 MB unzipped function limit.**
   Roughly 2 GB against a 250 MB ceiling. Not fixable by trimming.
3. **Streamlit cannot be deployed on Vercel.** It needs a long-lived server with websockets.
   Vercel serves static assets plus serverless functions.

Also: no writable filesystem (so sessions must go to a managed store), and a 60 s function cap on
Hobby / 300 s on Pro fluid compute (so streaming is required, not optional).

---

## 3. Implementation plan

### Phase 0 — Make it run from a clean clone
_Goal: clone, `uv sync`, one command, working chat turn. No behaviour changes._

**0.1** Replace `backend/config.py` with **`backend/settings.py` using `pydantic-settings`**, reading
from env with validation and clear startup errors for missing required keys. Commit a
**`.env.example`** listing every key. Keep `.env` gitignored; drop the `config.py` ignore rule.

**0.2** Add the missing deps to `pyproject.toml`: `numpy`, `pydantic-settings`, `httpx`. **Do not**
re-add `faiss` — 3.1 removes it. Keep `sentence-transformers` as an **optional extra**
(`[project.optional-dependencies] local`) used only by the index builder and local-dev embedding,
never by the deployed function.

**0.3** Make `backend/` a real package (`__init__.py`; imports become `from backend.tools import ...`),
move `build_faiss_index.py` to `scripts/`, and fix its path handling to resolve from the repo root
instead of the hardcoded `C:\GenAi\...`.

**0.4** Add a task runner (`Makefile` / `tasks.ps1`) with `dev` (API + UI), `index` (rebuild
embeddings), `test`, `lint`.

**0.5** Rewrite the README install/run section to match reality; add `/health` to the API.

**Exit criterion:** a fresh clone with only `.env` filled in serves a chat turn.

### Phase 1 — Safety hardening
_Goal: the crisis path is deterministic, auditable, and actually helps the user._

**1.1 Deterministic pre-LLM risk classifier** in `backend/safety.py`, running on every inbound
message *before* the agent sees it. Two tiers: a fast lexicon/regex layer for explicit intent, plus
a cheap LLM classifier for ambiguous phrasing. Returns `RiskLevel.{NONE,ELEVATED,CRISIS}`. On
`CRISIS`, **bypass the agent entirely** and return a fixed, human-reviewed crisis response — never
a generated one. This is the fix for S1 and the single most important change in this document.

**1.2 Always surface crisis resources in text.** Ship `data/crisis_resources.json` keyed by country
(988 US, 116 123 UK Samaritans, AASRA India +91-9152987821, etc.) and render hotlines in the reply
itself. The Twilio call becomes a *supplement*, not the whole intervention (S2).

**1.3 Make `call_emergency()` safe**: wrap in try/except, return a status string the agent can use,
structured-log every invocation (timestamp, risk level, trigger-text hash — not raw text), and gate
it behind an explicit `EMERGENCY_CALLS_ENABLED` flag so it's off by default in dev (S3).

**1.4 Delete the fake therapist list.** Replace with a real Google Places / SerpAPI lookup; when the
API key is absent, return an honest "I can't look that up right now, here's how to find a licensed
therapist" — never invented names and numbers (S4).

**1.5 Persistent disclaimer in the UI**, plus a first-turn notice and a footer on every assistant
message noting this is not medical advice (S5).

**1.6 Safety eval set**: ~40 hand-written cases (explicit intent, veiled intent, metaphor, negation —
"I used to feel suicidal but I'm better now" must NOT escalate — and other languages) run in CI.
Track false-negative rate as the metric that matters. False positives are cheap; false negatives
are not.

### Phase 2 — Correctness

**2.1 Conversation memory** via LangGraph's checkpointer with a `thread_id` per session. Add
`session_id` to the request/response contract; the frontend persists it. Fixes C1 — the
highest-value functional fix in the repo.

**2.2 Fix the tool-name mismatches** in `SYSTEM_PROMPT`, register `ask_exam_rag` as a proper `@tool`
with a good description, delete the dead `ask_exam_rag_tool` (C2, C3). Rewrite the system prompt to
describe *routing policy* rather than a numbered tool list the model can misread.

**2.3 Un-block the event loop**: switch to `graph.astream()` with async tools, or make the endpoint
`def` so FastAPI runs it in the threadpool (C4).

**2.4 Harden the frontend**: status checks, timeouts, `.get()` with fallbacks, a friendly error
bubble instead of a stack trace (C5, C6).

**2.5 Structured logging** (`structlog`) with request IDs; log the swallowed exception in
`query_medgemma` (C7).

**2.6 Fix `parse_response`** to take the last AI message with no pending tool calls, rather than the
last message with any content (C8).

**2.7 Tests**: pytest + `httpx.ASGITransport`, with the LLM and Twilio mocked. Cover routing per
tool, the crisis bypass, memory across turns, and RAG retrieval shape. Wire up GitHub Actions
running `ruff` + `pytest` on PR.

### Phase 3 — Efficiency

**3.1 Replace FAISS with numpy** cosine similarity over the `(350, d)` normalized matrix. Identical
results, one less dependency, faster at this scale (E1). Keep the interface unchanged so you can
swap back if the corpus grows past ~50k rows.

**3.2 Provider abstraction for inference** — `backend/providers/` with `OllamaProvider` (local dev)
and `GroqProvider` / hosted-MedGemma (prod), selected by env. This unblocks Vercel and also makes
the LLM mockable in tests (E2).

**3.3 Same abstraction for embeddings.** Prod embeds the *query* via a hosted embedding API; the
corpus is embedded once offline and committed as `.npy`. Critical detail: **the corpus must be
re-embedded with whatever model prod uses** — you cannot mix MiniLM corpus vectors with a different
model's query vector.

**3.4 Warm the index at startup** via FastAPI `lifespan` instead of lazy-loading on request one (E3).

**3.5 Stream responses end-to-end** — SSE from FastAPI, token rendering in the UI. Biggest
perceived-latency win available, and mandatory under Vercel's function timeout (E4).

**3.6 Cache**: an embedding cache keyed by query hash and a short-TTL response cache for common
queries, both behind a cache interface (in-memory locally, Upstash Redis in prod) (E6).

**3.7 Reuse HTTP connections** with a shared `httpx.AsyncClient` on app state (E7).

### Phase 4 — New features
Ordered by value per unit of effort for this specific product.

**4.1 Session persistence + history** — conversations survive reload and resume where you left off.
Prerequisite for most of what follows (Postgres/Neon via SQLModel).

**4.2 Mood check-in & trend tracking** — a 1–5 check-in per session, stored, charted over time, and
fed back into agent context ("you've mentioned feeling low three days running"). Turns a stateless
chatbot into something with continuity.

**4.3 Guided coping exercises as a tool** — box breathing, 5-4-3-2-1 grounding, progressive muscle
relaxation, delivered as timed interactive UI components rather than walls of text. Cheap to build,
genuinely useful, and doesn't depend on the LLM being good.

**4.4 Safety plan builder** — a Stanley-Brown-style plan (warning signs, internal coping strategies,
people to contact, professionals, making the environment safe), saved and retrievable. Real clinical
practice, and it pairs naturally with the crisis tier from Phase 1.

**4.5 Rolling conversation summarization** — summarize older turns to keep long sessions inside the
context window and cut token cost. Pairs with 4.1.

**4.6 Journaling with sentiment trends** — free-text entries, sentiment scored, surfaced alongside mood.

**4.7 Structured screeners (PHQ-9 / GAD-7)** — validated, public-domain instruments. Worth doing, but
only with unambiguous framing that these are *screening* scores and not diagnoses, and with an
automatic handoff to the crisis path when PHQ-9 item 9 is non-zero.

**4.8 Real therapist locator** (completes 1.4) with filters for modality, insurance, telehealth.

**4.9 Multilingual support** — detect language, respond in kind, localize crisis resources by region.
Widens reach considerably; note the safety classifier must be evaluated per language.

**4.10 Expand the RAG corpus beyond exam stress** — 350 records on one topic is thin. Add sleep,
relationships, grief, workplace burnout, with provenance metadata per record.

**4.11 Escalation audit view** — an internal page listing crisis triggers and outcomes. Needed to
operate this responsibly at any scale.

### Phase 5 — Vercel deployment
Depends on Phases 0–3; 3.2 and 3.3 are the gate.

**5.1 Decide the topology.** Recommended:
- **Next.js (App Router) frontend on Vercel** — replaces Streamlit for production; keep
  `frontend.py` as the local dev/demo UI.
- **FastAPI on Vercel Python functions** under `api/`, or split the API to Fly/Render if you want
  to keep the heavier local-model path alive in prod.
- Streamlit is not deployable on Vercel — this is a replace-or-relocate decision, not a config one.

**5.2 Prod inference = Groq** (already a dependency, already used for the agent, and fast). The
"MedGemma" persona moves to a system prompt on a hosted model. This trades away the medically-tuned
weights — if MedGemma specifically matters, the alternative is a hosted MedGemma endpoint (Vertex AI
/ HF Inference) behind the same provider interface from 3.2.

**5.3 Prod retrieval = committed `.npy` + numpy**, no torch, no faiss. The whole corpus is 537 KB; it
ships inside the function bundle and loads in milliseconds.

**5.4 Managed state**: Neon/Vercel Postgres for sessions, mood, journals; Upstash Redis for cache and
rate limiting.

**5.5 Platform config**: `vercel.json`, env vars per environment, CORS locked to the frontend origin,
rate limiting on `/ask`, and streaming responses to stay inside the function timeout.

**5.6 Observability**: Sentry for errors, plus a lightweight metrics path for escalation counts and
p95 latency.

**5.7 Pre-launch gate** — do not deploy publicly until the safety eval passes, the fake therapist data
is gone, the disclaimer is in the UI, and a data-retention policy exists for stored conversations.

---

## 4. Suggested sequencing

| Order | Phase | Why here |
|---|---|---|
| 1 | Phase 0 | Nothing else is testable until the app starts |
| 2 | Phase 1 | Safety debt compounds; cheapest to fix before features land on top |
| 3 | Phase 2.1–2.3 | Memory and the event loop are the two fixes users notice immediately |
| 4 | Phase 3.2–3.3 | Provider abstraction gates both testing and deployment |
| 5 | Phase 2.7 | Tests, once the seams from 3.2 make mocking possible |
| 6 | Phase 3 rest | Streaming, caching, numpy retrieval |
| 7 | Phase 5.1–5.3 | Get a minimal thing live on Vercel |
| 8 | Phase 4 | Features, on a foundation that can hold them |

## 5. Open decisions

1. **Is MedGemma essential, or is the persona enough?** Keeping MedGemma means paying for a hosted
   endpoint or not deploying on Vercel. Dropping it means Groq everywhere and a much simpler deploy.
2. **Streamlit or Next.js?** Next.js for a Vercel-native deploy; keeping Streamlit means hosting the
   UI elsewhere (Render, Streamlit Community Cloud) and running a split stack.
3. **Accounts or anonymous sessions?** Everything in Phase 4 (mood history, journals, safety plans)
   needs identity. Anonymous device-scoped sessions are a lighter middle ground.
4. **What is the data-retention policy** for conversation content? Needs an answer before 4.1
   persists anything.
