# SafeSpace

An agentic mental health support assistant with **deterministic crisis guardrails**.

Built with LangGraph and FastAPI. Runs MedGemma locally via Ollama for development, and
hosted inference in production so it fits inside a serverless bundle.

> **SafeSpace is not a therapist.** It does not provide medical advice, diagnosis, or
> treatment. It is a support tool and a technical demonstration. In an emergency, contact
> your local emergency services.

---

## What it does

| Route | Handled by |
|---|---|
| Suicidal ideation, self-harm intent | **Crisis protocol** — fixed copy, agent bypassed entirely |
| Exams, studying, academic pressure | `ask_exam_stress_specialist` — RAG over a 350-passage corpus |
| Panic, overwhelm, "what do I do right now" | `guided_coping_exercise` — breathing, grounding, PMR, worry time |
| "Find me a therapist" | `find_nearby_therapists` — real search, or an honest "I can't look that up" |
| Everything else emotional | `ask_mental_health_specialist` — therapist-voice LLM |

## Safety model

This is the part worth reading before anything else.

**Crisis detection does not depend on the LLM deciding to escalate.** Every inbound
message goes through [`backend/safety.py`](backend/safety.py) *before* the agent sees it:

1. **Deterministic tier** — a lexicon of explicit-intent patterns, with handling for
   past-tense recovery ("I used to feel suicidal, I'm better now") and third-party
   concern ("my friend wants to kill herself"). Neither escalates to CRISIS; both are
   still treated as ELEVATED.
2. **LLM tier** *(optional)* — catches phrasings the lexicon misses. It can only **raise**
   the risk level, never lower it. It is skipped entirely when the rules already say
   CRISIS, and a classifier outage cannot silently downgrade anyone.

On **CRISIS**, the agent never runs. The response is fixed, human-reviewed copy assembled
from verified hotline data in [`data/crisis_resources.json`](data/crisis_resources.json) —
never model-generated. The Twilio call is a *supplement* to showing the user a number they
can call, not a replacement for it.

On **ELEVATED**, the agent responds normally and hotlines are appended to its reply.

The eval set lives in [`tests/test_safety.py`](tests/test_safety.py) and runs in CI. The
metric that matters is the **false-negative rate on CRISIS** — a missed escalation is the
failure with real-world consequences. False positives cost someone an unneeded hotline
number, which is an acceptable trade. **When you find a phrasing it gets wrong, add it to
that file.**

Two related guarantees:
- **Emergency calls are off by default** (`EMERGENCY_CALLS_ENABLED=false`). A fresh clone
  cannot place a real phone call by accident.
- **Nothing fabricates referrals.** Without `SERPAPI_API_KEY`, the locator says it cannot
  search and points to real directories, rather than inventing practices and numbers.

---

## Quick start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <your-repo-url> SafeSpace
cd SafeSpace

cp .env.example .env          # then add GROQ_API_KEY (free: console.groq.com)
uv sync --extra dev           # runtime + test deps, no torch

uv run uvicorn backend.main:app --reload --port 8000
```

Open <http://localhost:8000> — the API serves the chat UI.

On Windows, `.\tasks.ps1 install` then `.\tasks.ps1 api` does the same. On macOS/Linux,
`make install` and `make api`.

### The one key you actually need

`GROQ_API_KEY` plus `LLM_PROVIDER=groq` is enough to run everything. No Ollama, no torch,
no other accounts. What you give up is the exam corpus grounding — see the table.

The `.env.example` defaults (`LLM_PROVIDER=ollama`, `EMBEDDING_PROVIDER=local`) are the
*full-fidelity* local setup, not the cheapest one. Each optional piece degrades on its
own rather than breaking the app:

| If this is missing | What happens |
|---|---|
| Ollama not installed (`LLM_PROVIDER=ollama`) | Every reply is the fallback message. **Set `LLM_PROVIDER=groq` instead.** |
| `sentence-transformers` not installed (`EMBEDDING_PROVIDER=local`) | Retrieval falls back to BM25 — corpus still works, weaker on paraphrase |
| `HUGGINGFACE_API_KEY` unset (`EMBEDDING_PROVIDER=huggingface`) | Same BM25 fallback |
| `SERPAPI_API_KEY` unset | Locator explains how to find a therapist instead of searching |
| Twilio unset | Emergency calls stay disabled; **hotlines are still always shown** |

The crisis path, coping exercises, exam retrieval, memory, and the chat UI need no keys
at all.

### Running MedGemma locally

The default `LLM_PROVIDER=ollama` expects [Ollama](https://ollama.com):

```bash
ollama pull alibayram/medgemma:4b
```

Set `LLM_PROVIDER=groq` to skip Ollama entirely and run everything hosted.

### Rebuilding the corpus index

Only needed if you change `data/mental_health_exams.jsonl`:

```bash
uv sync --extra local                        # pulls sentence-transformers (~2GB)
uv run --extra local python -m scripts.build_index
```

### The Streamlit UI

Kept for local demos. Streamlit needs a long-lived websocket server, so it is not what
gets deployed.

```bash
uv run --extra local streamlit run frontend.py
```

---

## Architecture

```
web/                    Static chat UI (no build step) — served by Vercel's CDN in prod
frontend.py             Streamlit dev UI
api/index.py            Vercel serverless entrypoint
backend/
  main.py               FastAPI app: routing, SSE, rate limiting, CORS
  safety.py             Risk classification + crisis copy      <- read this one
  agent.py              LangGraph agent, routing policy, memory
  tools.py              The five tools
  retrieval.py          numpy cosine search, with a BM25 fallback
  lexical.py            BM25 — keeps the corpus usable with no model or key
  providers/
    llm.py              Ollama (local) | Groq (prod)
    embeddings.py       sentence-transformers (local) | HF Inference API (prod)
  settings.py           Validated config
scripts/build_index.py  Rebuild embeddings + manifest
tests/                  118 tests, including the safety eval set
```

### Why there is no FAISS

The corpus is 350 vectors of 384 dimensions — 537KB. `IndexFlatIP` was doing an
exhaustive scan of it anyway, so a single numpy matmul gives identical results, is faster
once FAISS's call overhead is counted, and removes a ~30MB dependency from a bundle with
a 250MB ceiling. See [`backend/retrieval.py`](backend/retrieval.py). If the corpus grows
past roughly 50k rows, reintroduce an ANN index behind `CorpusIndex.search()` — the
interface is narrow so that swap stays local.

### Vector-space safety

The corpus is embedded offline and committed; only the *query* is embedded at request
time. Both providers target the same MiniLM model, and
[`embeddings/manifest.json`](embeddings/manifest.json) records which. If
`EMBEDDING_MODEL` ever drifts from the corpus, startup fails loudly — because mismatched
vector spaces do not error, they just quietly return meaningless results.

---

## Configuration

Every key is documented in [`.env.example`](.env.example). The only one required to run
locally is `GROQ_API_KEY`.

| Key | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` local, `groq` for deployment |
| `EMBEDDING_PROVIDER` | `local` | `local` needs the `local` extra; `huggingface` for serverless |
| `EMERGENCY_CALLS_ENABLED` | `false` | Must be explicitly enabled; validated against Twilio config at startup |
| `SERPAPI_API_KEY` | — | Absent → locator declines honestly instead of inventing referrals |
| `RATE_LIMIT_PER_MINUTE` | `20` | Per-instance; see the Vercel notes below |
| `DEFAULT_CRISIS_REGION` | `INTL` | `US`, `UK`, `IN`, `CA`, `AU` also available |

---

## Demo and deployment

- **[DEMO.md](DEMO.md)** — a 5-minute runbook: what to type, in what order, and what to
  point at.
- **[DEPLOY.md](DEPLOY.md)** — step-by-step Vercel deploy, every environment variable,
  and the two limitations to understand first.

## Deploying to Vercel

Three things made the original build undeployable, and all three are resolved:

| Blocker | Resolution |
|---|---|
| Ollama needs a GPU and a persistent process | `LLM_PROVIDER=groq` |
| torch + sentence-transformers + faiss ≈ 2GB vs a 250MB limit | Hosted embeddings, numpy retrieval — see `requirements.txt` |
| Streamlit needs websockets | Static UI in `web/`, served by the CDN |

```bash
npm i -g vercel
vercel

vercel env add GROQ_API_KEY production
vercel env add HUGGINGFACE_API_KEY production
vercel env add LLM_PROVIDER production          # groq
vercel env add EMBEDDING_PROVIDER production    # huggingface
vercel env add ENVIRONMENT production
vercel env add CORS_ALLOW_ORIGINS production    # your domain, not *

vercel --prod
```

CI enforces the bundle size, so a stray `torch` in `requirements.txt` fails the build
rather than the deploy.

### Two limitations to know about before you go live

Both are single-instance assumptions that serverless breaks. Neither is hard to fix, but
both are currently **unfixed**:

1. **Conversation memory is per-instance.** `MemorySaver` in
   [`backend/agent.py`](backend/agent.py) lives in process memory, so a cold start loses
   the thread. Swap in `PostgresSaver` with a Neon/Vercel Postgres `DATABASE_URL` for
   durable memory.
2. **Rate limiting is per-instance**, so the effective limit is `RATE_LIMIT_PER_MINUTE ×
   instances`. Move the counter to Upstash Redis to make it real.

---

## Development

```bash
make test        # or: uv run pytest -q
make safety      # the crisis eval set on its own
make check       # lint + format + tests, exactly what CI runs
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, format, the full
suite, the safety eval as a separately-named step, and the bundle-size check. It installs
**without** the `local` extra, so CI proves the app works under the same no-torch
constraint the serverless function runs under.

---

## Data and privacy

Be deliberate here — this is health-adjacent conversation content.

**Currently:** conversations are held in process memory only and lost on restart. Nothing
is written to disk or to a database. Risk assessments are logged with a **SHA-256 hash**
of the message, never the raw text — enough to correlate an incident, not enough to read
what someone wrote.

**Before you persist anything** (which mood tracking, journaling, and saved safety plans
all require), decide and publish: what is stored, for how long, who can read it, how a
user deletes it, and what happens to crisis-flagged conversations. Persisting first and
writing the policy later is the wrong order for this kind of data.

---

## Roadmap

[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) has the full gap analysis and the
remaining phases — session persistence, mood tracking, safety plan builder, screeners,
multilingual support, and corpus expansion beyond exam stress.

---

*Not a substitute for professional care. If you are struggling, please talk to someone.*
