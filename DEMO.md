# SafeSpace demo runbook

Written for: whoever is presenting. ~5 minutes.

---

## Start it

```powershell
cd C:\nexus\SafeSpace
.\tasks.ps1 api
```

Open <http://localhost:8000>. Nothing else is needed — no Ollama, no torch, no second
terminal. Set the region selector (top right) to match your audience so the hotlines are
the right ones.

**Warm it up before you present.** Send one throwaway message and delete the tab. The
first request builds the agent graph and the BM25 index, which adds a couple of seconds
that you don't want during a demo.

---

## The run, in order

Each message is chosen to trigger a different path. Paste them as-is.

### 1. General support

> I've been feeling really lonely since I moved city

Routes to `ask_mental_health_specialist`. Sets up the memory beat later — this is the
fact it will recall.

### 2. Retrieval

> I can't sleep because I keep worrying about failing my exams

Routes to `ask_exam_stress_specialist`. **Point at the terminal**: `bm25_index_built`,
then a reply grounded in a near-verbatim corpus passage. Worth saying out loud that this
runs with no embedding model, no API key, and no network — it's a fallback that kept a
headline feature alive for anyone who clones the repo.

### 3. Acute distress

> I'm having a panic attack right now

Routes to `guided_coping_exercise` and returns box breathing. The point: not everything
needs to be generated. Some things should be the same correct answer every time.

### 4. Memory

> what have I told you about so far?

Recalls **both** earlier turns and usually connects them ("carrying finals *and* the
loneliness of a new city"). Note that `tool_called` is `None` — it answered from
conversation state, no tool.

The original build sent only `[system, current message]` on every request, so it forgot
everything between turns.

### 5. The crisis path — this is the one that matters

> I want to kill myself

**Say what happens before you send it**, because the interesting part is invisible:

- The message is classified **before the agent ever sees it**
- On CRISIS the agent is **bypassed entirely** — no LLM writes this reply
- The copy is fixed and reviewed; the hotlines come from a verified data file
- The terminal shows `risk_assessed` then `crisis_bypass`, with the message **hashed**,
  never logged raw

Then send it. Point at the red-bordered card and the real numbers.

**The contrast worth stating:** before this rework, crisis detection was one line in a
prompt asking the model to please call a tool. If the model didn't, nothing happened —
and the user was shown no phone number at all, ever. The only action was a Twilio call to
one preconfigured contact.

### 6. Honest failure (optional, 20 seconds)

> can you find me a therapist in Bangalore

With no `SERPAPI_API_KEY` it says it can't look that up and points to real directories.

Worth showing because the original returned three invented therapists with fake `555`
numbers to a distressed user. A system that says "I don't know" is more trustworthy than
one that makes things up — especially this one.

---

## If someone asks

**"What stops it escalating on 'I used to feel suicidal but I'm better now'?"**
Past-tense and third-party phrasings are detected and downgraded to ELEVATED — resources
shown, agent still handles the turn, no crisis bypass. There are 18 such cases in the
eval set.

**"How do you know the classifier works?"**
`.\tasks.ps1 safety` — 62 tests, runs in CI as its own step. It caught four real bugs
during development, including `kms` firing on "I walked 5 kms today" and "kill herself"
not matching at all.

**"Why no FAISS / why is it fast?"**
350 vectors of 384 dimensions. FAISS was doing an exhaustive scan anyway; a numpy matmul
is identical and removes 30MB from a bundle with a 250MB ceiling.

**"Can it be deployed?"**
Yes — see [DEPLOY.md](DEPLOY.md). Config is committed. Two known limitations (memory and
rate limiting are per-instance) are documented rather than hidden.

---

## Known rough edges — name them before someone finds them

- **Groq free-tier rate limits.** Under rapid-fire input you may see a slow turn. Don't
  spam the input during the demo.
- **Rate limit is 20/min.** Raise `RATE_LIMIT_PER_MINUTE` in `.env` if you're demoing to
  a room that will all try it at once.
- **Exam retrieval is BM25, not embeddings**, unless you've run `uv sync --extra local`.
  Keyword-strong queries work well; heavy paraphrase is weaker.
- **Memory is in-process.** Restarting the server clears every conversation.

---

## One-line framing

> It started as a prototype where the crisis path depended on the model choosing to call
> a tool, and the therapist referrals were invented. Now crisis detection runs before the
> model and can't be talked out of it, the referrals are real or honestly absent, and
> there are 134 tests including an eval set for the one failure that actually matters.
