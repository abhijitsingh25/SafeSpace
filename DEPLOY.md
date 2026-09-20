# Deploying SafeSpace to Vercel

Written for: whoever is running the deploy (you). Assumes the repo is committed and
`make check` passes.

---

## Before you start: two things to decide

**1. Is this a preview or is it public?**

Deploy a **preview** first (`vercel`, no `--prod`). It gives you a private URL, costs
nothing, and shows you real cold-start behaviour. Only promote to production once you've
clicked through it yourself.

**2. Are you putting this in front of real people?**

If yes, read [Known limitations](#known-limitations) first. There are two, both are
single-instance assumptions that serverless breaks, and one of them means the assistant
can forget someone mid-conversation. Fine for a demo; not fine for strangers in distress.

---

## Step 1 — Install the tooling

```powershell
choco install nodejs -y          # or: https://nodejs.org
npm i -g vercel
vercel --version
```

## Step 2 — Log in

```powershell
vercel login
```

This opens a browser for OAuth. **This is the step nobody can do on your behalf** — it
needs your account credentials in a real browser session.

If you want to hand the deploy to someone (or something) else, create a scoped token at
<https://vercel.com/account/tokens> and pass `--token=...` on every command instead. Treat
that token like a password; it can deploy to your account.

## Step 3 — Link the project

```powershell
cd C:\nexus\SafeSpace
vercel link
```

Answer: create a new project, name it `safespace`, accept the detected settings. This
writes `.vercel/` (already gitignored).

## Step 4 — Set environment variables

Production needs different values than your `.env`, because Ollama and torch don't exist
in a serverless function.

```powershell
vercel env add GROQ_API_KEY production
vercel env add GROQ_MODEL production            # qwen/qwen3.8-27b
vercel env add LLM_PROVIDER production          # groq          <- NOT ollama
vercel env add EMBEDDING_PROVIDER production    # huggingface   <- NOT local
vercel env add HUGGINGFACE_API_KEY production
vercel env add ENVIRONMENT production           # production
vercel env add CORS_ALLOW_ORIGINS production    # https://your-app.vercel.app
```

| Variable | Value | Why |
|---|---|---|
| `GROQ_API_KEY` | your key | Required. Rotate the one that was in `.env.example`. |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Llama models 404 on this account |
| `LLM_PROVIDER` | `groq` | Ollama needs a GPU and a persistent process |
| `EMBEDDING_PROVIDER` | `huggingface` | `local` pulls ~2GB of torch, over the 250MB limit |
| `HUGGINGFACE_API_KEY` | free token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens) | Read-only scope is enough |
| `ENVIRONMENT` | `production` | Switches logging to JSON |
| `CORS_ALLOW_ORIGINS` | your domain | **Not `*`** once public |

**Optional.** Leave these unset and the app degrades honestly rather than breaking:

| Variable | Without it |
|---|---|
| `SERPAPI_API_KEY` | Locator explains how to find a therapist instead of searching |
| `TWILIO_*` + `EMERGENCY_CALLS_ENABLED` | No phone call. **Hotlines are still always shown.** |

If you skip `HUGGINGFACE_API_KEY`, retrieval falls back to BM25 — the exam corpus still
works, just with weaker paraphrase matching. That is a reasonable choice for a demo.

## Step 5 — Deploy

```powershell
vercel                  # preview - do this first
vercel --prod           # promote once you've clicked through it
```

## Step 6 — Verify

```powershell
curl https://your-app.vercel.app/health
```

Expect `corpus_loaded: true` and `llm_provider: "groq"`. Then in the browser, check all
four in order:

1. **"I'm stressed about my finals"** → routes to `ask_exam_stress_specialist`
2. **"what did I just tell you?"** → recalls it (proves memory survived)
3. **"I'm panicking right now"** → returns a breathing exercise
4. **"I want to kill myself"** → crisis card, hotlines, `tool_called: crisis_protocol`

If #4 doesn't show hotlines, **stop and do not go public.** That is the one path that
must never fail.

---

## Known limitations

Both are documented in the README and neither is hard to fix — but both are currently
**unfixed**, and you should know which one matters for your use.

### Memory is per-instance

`MemorySaver` lives in process memory. Vercel may route the next request to a cold
instance, which drops the conversation thread. For a normal app that's a papercut. For
something presenting itself as a listener, forgetting someone mid-conversation is exactly
the failure this rework set out to fix.

**Fix:** a Neon/Vercel Postgres `DATABASE_URL` and `PostgresSaver` in
[`backend/agent.py`](backend/agent.py). Roughly an hour.

**For a short demo on one warm instance, you will probably not hit this.** For real users,
you will.

### Rate limiting is per-instance

The counter is in process memory, so the effective limit is
`RATE_LIMIT_PER_MINUTE x instances`. Move it to Upstash Redis before the URL is public.

---

## Cost

Groq's free tier and Vercel's Hobby tier cover a demo. Two things to watch:

- **Groq free-tier rate limits.** Retry backoff is visible in local testing; under demo
  load you may see slow turns.
- **Hobby caps functions at 60s.** Responses stream, so this is unlikely to bite, but a
  slow model plus a retry can get close.

---

## Rollback

```powershell
vercel rollback                              # previous deployment
vercel ls                                    # list, then:
vercel promote <deployment-url>
```

Deployments are immutable, so rollback is instant and safe.
