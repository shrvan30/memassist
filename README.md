# MemAssist — a MemGPT-style assistant with infinite memory, at $0/month

A personal AI assistant that **edits its own memory** with tool calls, paging
information between its finite context window ("RAM") and external storage
("disk") — the [MemGPT](https://arxiv.org/abs/2310.08560) architecture
(Packer et al., 2023). The LLM layer is a **free-tier failover router** across
four OpenAI-compatible providers, so the whole system runs at no cost.

```
You:  "I'm Shravan, I work on ML infra and I hate long emails."
      → agent calls core_memory_append(block="human", …)
      → restart the app, tell it nothing, ask "what do you know about me?"
      → it still knows. That's the whole thesis.
```

**Current status: Phase 1 complete and in real use.** Phases 2–4 are specified
and scaffolded but not built — see [Build phases](#build-phases) for exactly
what exists and what doesn't.

---

## Table of contents

- [The idea](#the-idea)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Memory tiers](#memory-tiers)
- [The tool contract](#the-tool-contract)
- [The agent loop](#the-agent-loop)
- [The failover router](#the-failover-router)
- [Data flow, one turn](#data-flow-one-turn)
- [Database schema](#database-schema)
- [Setup](#setup)
- [Run](#run)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Repo layout](#repo-layout)
- [Build phases](#build-phases)
- [Known deviations from the spec](#known-deviations-from-the-spec)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## The idea

An LLM's context window is finite; a relationship with an assistant is not.
MemGPT's insight is to treat the LLM like an OS process: give it a small
working set in "RAM" (the context window), unbounded "disk" (external stores),
and — critically — **let the model itself issue the page-in/page-out calls** as
ordinary function calls. The illusion of unlimited memory falls out of the
agent managing its own memory hierarchy.

MemAssist implements that faithfully, and adds one constraint of its own: it
must cost nothing to run. That constraint is not cosmetic — it shapes the
design. Because the serving model rotates across four free tiers mid-conversation,
the system needs a stable persona block to hold its voice steady, flat tool
schemas that four different tool-calling dialects all accept, a context-pressure
signal computed against *whichever* provider answered, and a local embedder so
the archival vector space never fragments.

---

## Tech stack

### Language & runtime

| | |
|---|---|
| **Python** | 3.11+ (`requires-python = ">=3.11"`) — uses PEP 604 unions and `from __future__ import annotations` throughout |
| **Typing** | Type hints everywhere; `typing.Protocol` for the two architectural seams (`LLMRouter`, `MemoryInterface`) |
| **Packaging** | setuptools via `pyproject.toml`, editable install (`pip install -e ".[dev]"`) |

### LLM layer

| Component | Choice | Why |
|---|---|---|
| **SDK** | `openai>=1.40` | All four providers speak the OpenAI chat-completions format, so one client shape works for all — they differ only in `base_url`, `api_key`, `model` |
| **Providers** | Gemini → Groq → OpenRouter → Mistral | Priority-ordered free tiers; see [the chain](#provider-chain) |
| **Chain config** | `pyyaml>=6.0` reading [`llm/providers.yaml`](llm/providers.yaml) | Provider chain is data, not code — swap a model slug without touching Python |
| **Budget ledger** | SQLite (`provider_usage` table) | Survives restarts; resets at UTC midnight by construction |

> **Hard rule:** every LLM call goes through `llm/router.py`. No agent code ever
> constructs a provider client directly.

### Memory & storage

| Tier | Technology | Notes |
|---|---|---|
| **Core memory** | SQLite `core_blocks` | Two editable blocks (`persona`, `human`), 2000-char cap each, injected into the system prompt every turn |
| **Recall memory** | SQLite `messages` | Complete append-only event log, indexed on session/created_at/event_type, keyword + date search |
| **Archival memory** | `chromadb>=0.5`, persistent client, cosine space | Semantic search over passages |
| **Embeddings** | **Local only** — Phase 1 ships a deterministic offline hashing embedder (512-dim, blake2b + sign bit, L2-normalized) | Target is `sentence-transformers` / `bge-small-en-v1.5` (384-dim). Never provider-hosted — see [why](#why-local-embeddings) |

### Validation & schemas

| | |
|---|---|
| **`pydantic>=2.5`** | One input model per tool, used to validate model-supplied arguments at dispatch time |
| **Hand-written JSON schemas** | What the LLM actually sees. Colocated with the Pydantic models in [`memory_server/schemas.py`](memory_server/schemas.py) so the pair can't drift |
| **Flat schemas, always** | No nested objects, no `$ref`. Nested schemas are the single most common source of cross-provider tool-calling failures |

### Interface

| | |
|---|---|
| **`streamlit>=1.30`** | Phase 1 chat UI with a live memory sidebar, context-usage bar, and a provider badge on every reply |
| **`python-dotenv>=1.0`** | Loads `.env` at import time in [`config.py`](config.py); optional at runtime |

### Testing

| | |
|---|---|
| **`pytest>=7.4`** | 56 tests, **no API keys, no network**, sub-second |
| **Injection seams** | `client_factory` (router transport), `now_fn` (ledger clock), `embed_fn` (archival embedder) — all three exist so the time- and network-dependent logic is deterministic under test |

### Planned (later phases)

| Phase | Technology |
|---|---|
| 2 | `mcp` SDK / FastMCP (stdio transport); async MCP client bridge; `sentence-transformers` |
| 3 | PostgreSQL + pgvector; FastAPI with SSE streaming; Next.js + Tailwind; Docker Compose; MCP over Streamable HTTP |
| 4 | Langfuse tracing (tagged by provider); deep-memory-retrieval eval harness; per-provider tool-calling CI |

---

## Architecture

```
app/streamlit_app.py       Chat UI + live memory/provider sidebar
        │
assembly.py                Composition root — wires everything from config.py
        │
agent/                     Knows only two protocols. No storage, no SDKs.
  loop.py                  Heartbeat loop, tool dispatch, pressure injection
  prompts.py               System-prompt assembly
  token_budget.py          Pure context-usage math
        │                              │
   router.chat()                  MemoryInterface
        │                              │
llm/                             memory_server/
  router.py    failover + tagging   memory_tools.py   the six tools; validated dispatch
  budgets.py   provider_usage       schemas.py        Pydantic ↔ OpenAI tool defs
  errors.py    error classification storage/sqlite.py core + recall
  providers.yaml  the chain         storage/chroma.py archival + local embedder
```

**Two seams, both enforced by `Protocol` types** in
[`agent/loop.py`](agent/loop.py):

- `LLMRouter` — `chat()`, `context_window()`, `min_context_window()`. The agent
  cannot see providers.
- `MemoryInterface` — `render_core_memory()`, `memory_stats()`, `dispatch()`,
  `record_event()`. The agent cannot see SQL or vectors.

This is what makes the later phases local changes: Phase 2 swaps the
`MemoryInterface` implementation for an MCP client; Phase 3 swaps SQLite for
Postgres underneath it. Neither touches `agent/loop.py`.

---

## Memory tiers

| Tier | Storage | In context? | Tools |
|---|---|---|---|
| **Core** | SQLite `core_blocks` — `persona` + `human` | **Always** — injected into the system prompt each turn | `core_memory_append`, `core_memory_replace` |
| **Recall** | SQLite `messages` — full log, tagged `served_by` | On demand | `conversation_search`, `conversation_search_date` |
| **Archival** | Chroma vector store | On demand, semantic | `archival_memory_insert`, `archival_memory_search` |

### Core memory

The self-editing surface. Two blocks:

- **`persona`** — the agent's identity and behavior. This is what holds its
  voice stable when the underlying model changes from Gemini to Llama
  mid-conversation.
- **`human`** — durable facts about you: name, role, preferences, goals.

Both are capped (default 2000 chars). Writes that would overflow are **refused
with an instruction, not an exception**:

> `Update would exceed the 2000-character limit for the 'human' block (result
> would be 2104 chars). Summarize or move detail into archival memory instead.`

Likewise `core_memory_replace` requires a verbatim `old_text` match and, on
miss, returns text telling the model to re-read and retry. Every error is a
teaching signal the model can act on during its next heartbeat — nothing raises
into the loop.

### Recall memory

An append-only event log with a richer vocabulary than a plain chat transcript:

- `role ∈ {user, assistant, tool, system_event}`
- `event_type ∈ {message, tool_call, tool_result, pressure_warning, internal}`
- `served_by ∈ {gemini, groq, openrouter, mistral, NULL}`

Keyword search is AND-of-terms, case-insensitive substring, newest-first,
paginated (5/page), and **defaults to `event_type='message'`** so an agent
searching its own past isn't buried in its own tool traffic. Date search accepts
`YYYY-MM-DD` and expands bare dates to cover the whole day.

### Archival memory

A persistent Chroma collection in cosine space. Passages carry
`{created_at, source}` metadata. Chroma has no query offset, so paging fetches
`top_k * (page + 1)` results and slices locally — fine at MVP scale.

#### Why local embeddings

Phase 1 ships a deterministic, dependency-free hashing embedder: tokenize →
blake2b each token → index from the first 4 bytes, sign from a bit of the 5th →
accumulate → L2-normalize. No model download, no extra key, stable across
processes (so a passage embedded last week still matches today's query).

Embeddings are computed in `chroma.py` and passed to Chroma **explicitly**, with
the collection created as `embedding_function=None`. That deliberately sidesteps
Chroma's default embedder (which would download an ONNX model on first use) and
its version-fragile embedding-function serialization.

The rule this all serves: **archival vectors must never come from a chat
provider.** The chat provider rotates across four services; a rotating embedding
space would silently fragment archival memory into mutually unsearchable
islands. `ArchivalStore(embed_fn=…)` is the injection point for upgrading to
`bge-small-en-v1.5` without touching a caller.

---

## The tool contract

Seven tools, all returning plain strings. Signatures are frozen; parameter
schemas are flat.

```python
send_message(text)                                     # ONLY user-facing channel
core_memory_append(block: "persona"|"human", content)
core_memory_replace(block, old_text, new_text)
conversation_search(query, page=0)
conversation_search_date(start, end, page=0)           # YYYY-MM-DD
archival_memory_insert(content)
archival_memory_search(query, top_k=5, page=0)
```

Two conventions carry most of the weight:

**`send_message` is the only output channel.** Any prose the model writes
outside a tool call is internal monologue and never reaches you. This is
straight from MemGPT and it's what makes the inner reasoning safe to keep in
context.

**`request_heartbeat: bool` on every *memory* tool.** Setting it true means "give
me control back after this runs so I can chain another action before replying" —
search, then save, then answer. `send_message` deliberately has no heartbeat:
it's terminal. Chaining is capped at 5 rounds because **every heartbeat is a
real API request** billed against a provider's free-tier budget.

### Dispatch is the robustness boundary

`MemoryTools.dispatch()` **never raises**:

| Failure | Result |
|---|---|
| Unknown tool name | `"Error: 'foo' is not a memory tool."` |
| Pydantic `ValidationError` | `"Error: invalid arguments for …"` |
| Handler exception | `"Error: <tool> failed: …"` |
| Oversized result | Truncated to 4000 chars with a marker |

Tool results are treated as **untrusted text**: length-capped, never evaluated,
never interpolated into code. Every failure mode becomes something the model can
read and recover from on the next turn.

---

## The agent loop

One `step(user_text)` per user turn ([`agent/loop.py`](agent/loop.py)):

1. **Record** the user message to recall; push onto the in-context FIFO queue.
2. **Pressure check — once, at the top of the turn.** If
   `last_input_tokens / limit ≥ 0.7`, build a `[MEMORY PRESSURE]` warning, log it
   as a `system_event`, and inject it into the queue.
3. **Heartbeat loop (≤ 5 rounds):**
   - Rebuild the system prompt *fresh*: base instructions + rendered core memory
     + live stats (`Recall: 44 messages · Archival: 12 passages · Context:
     12,043 / 32,000 tokens (38%)`).
   - `router.chat([system, *messages], tools, tool_choice="auto")`.
   - Record `served_by`. Set `last_input_tokens` from the real
     `usage.prompt_tokens`, falling back to a ~4-chars/token estimate. Recompute
     the limit against **that provider's** window.
   - Echo the assistant turn into history in strict OpenAI shape (`content` may
     be `None` when the model only emitted tool calls — that's valid).
   - No tool calls → break.
   - Dispatch each call. `send_message` is intercepted in the loop (it's the
     user channel, not a memory tool); everything else goes to
     `memory.dispatch()`. **Every `tool_call` gets a matching `role:"tool"`
     message** — required before the next model call or providers reject the
     history.
   - **Continue condition:** stop if a message was sent and no heartbeat was
     requested; stop at the cap; otherwise loop.
4. **Safety net.** If the model wrote prose but never called `send_message`
   (common on weaker fallback models), surface that text anyway rather than
   showing you nothing.

### Context-window sizing

The pressure threshold is computed against the **active provider's** window,
capped by `MEMASSIST_CONTEXT_LIMIT`:

```python
limit = min(router.context_window(served_by), CONTEXT_LIMIT)
```

Before the first response of a session there is no active provider, so it uses
`min_context_window()` across the whole chain — the smallest window is the safe
planning default, since you don't know who will answer next. (Gemini offers 1M;
OpenRouter and Mistral offer 32K. Planning for 1M and landing on Mistral is how
you get a hard context error.)

---

## The failover router

Every LLM call in the system goes through `Router.chat()`.

```
for cfg in chain_for(lane):                        # priority-sorted, lane-filtered
    if no api key / cooling down / daily budget spent:
        skip, record reason
    try:
        raw = client.create(**kwargs)              # 1 jittered retry on 5xx
    except RateLimit(429):    cooldown 60s              → next provider
    except Quota(402):        cooldown to UTC midnight  → next provider
    except ServerError(5xx):                            → next provider
    except Auth(401/403):                               → next provider
    ledger.record_request(name, tokens)
    return ChatResult(served_by=name, …)
raise AllProvidersExhausted(attempts)              # carries per-provider reasons
```

### Provider chain

Configured as data in [`llm/providers.yaml`](llm/providers.yaml):

| Pri | Provider | Model | Window | Lanes | Role |
|---|---|---|---|---|---|
| 1 | **gemini** | `gemini-2.0-flash` | 1,048,576 | interactive | Primary — huge free tier |
| 2 | **groq** | `llama-3.3-70b-versatile` | 131,072 | interactive | Fast fallback |
| 3 | **openrouter** | a `:free` model | 32,768 | interactive | Emergency lane |
| 4 | **mistral** | `mistral-small-latest` | 32,768 | interactive **+ background** | Last resort + batch |

> Free-tier limits change constantly. The budgets in the YAML are conservative
> planning defaults for proactive skipping, not authoritative quotas — **verify
> at signup.**

### Key mechanisms

**Proactive skipping.** Never spend a request discovering a 429 you could have
predicted. Before calling, the router checks: key present? cooling down? daily
request limit hit? daily token limit hit? Any "no" and it moves on without
burning the request.

**Persistent ledger** ([`llm/budgets.py`](llm/budgets.py)). The `provider_usage`
table is keyed `(provider, UTC date)`. Daily reset is *emergent rather than
scheduled*: a new UTC day is simply a new row with zero counts and no cooldown.
No cron, no cleanup job. `now_fn` is injected so every clock-dependent path is
deterministic under test.

**Error classification** ([`llm/errors.py`](llm/errors.py)) — the subtle part:

| Status | Category | Action |
|---|---|---|
| 429 | `RateLimitError` | 60s cooldown, next provider |
| 402 | `QuotaExceededError` | Cooldown to UTC midnight, next provider |
| 5xx | `ProviderServerError` | Retry once with jittered backoff, then next |
| 401 / 403 | `ProviderAuthError` | Skip this provider (misconfigured) |
| connection / timeout | `ProviderServerError` | Retry, then next |
| **other 4xx** | **re-raised unchanged** | See below |

Other 4xx deliberately **do not** trigger failover. A malformed request would
fail identically on all four providers; masking it behind a cascade turns "your
schema has a bug" into a mystifying `AllProvidersExhausted`. So request bugs
surface immediately.

With one carefully-scoped exception: Groq/Llama reports a *model-side* tool-call
failure as HTTP 400 `tool_use_failed`. The request is valid — it works on other
providers — so those are string-matched and reclassified as retryable. That
exception is narrow on purpose; broadening it would re-hide real bugs.

**Transport injection.** `Router(…, client_factory=…)` means the entire failover
algorithm is unit-tested against fake clients with zero network. That's why
there are 21 router tests and no HTTP cassettes.

**Lane routing.** `chat_background()` filters the chain to
`lanes: [background]` — Mistral only. 2 RPM is irrelevant for batch work, and
Mistral's monthly token quota dwarfs the others.

**Provider tagging.** Every result carries `served_by`, which propagates into the
`messages` table, the reply badge, and the sidebar status panel. Excellent demo
value and genuinely the fastest way to debug "why did that answer sound
different?"

### Cross-provider consistency rules

These are load-bearing, all visible in code:

- **Fixed `temperature=0.3` everywhere.** Consistency across a rotating model
  pool matters more than per-provider tuning.
- **`tools`/`tool_choice` are omitted entirely when there are no tools** — some
  providers reject a null `tools` field outright.
- **Flat tool schemas.** No nested objects. Four tool-calling dialects, one
  format that all of them handle.
- **The Gemini pin is deliberate.** Gemini 3.x (`gemini-flash-latest`,
  `gemini-3-*`) requires round-tripping a Google-specific `thought_signature` on
  every function call, which breaks the flat cross-provider format with
  `400 INVALID_ARGUMENT`. Hence `gemini-2.0-flash`. If that slug is ever retired,
  pick another *non-thinking* Flash model.

---

## Data flow, one turn

```
user types → loop.step()
  ├─ record_event(user, message)                      → messages table
  ├─ if ≥70% context: inject [MEMORY PRESSURE]        → messages table
  └─ heartbeat loop (≤5):
       system = base instructions + core_blocks + live stats
       router.chat
         ├─ skip cooling / exhausted providers
         ├─ call provider → ChatResult(served_by=…)
         └─ ledger.record_request(tokens)             → provider_usage table
       dispatch tool calls:
         send_message      → user + record_event(assistant, message, served_by)
         core_memory_*     → SQLite core_blocks  (visible in NEXT turn's prompt)
         conversation_*    → SQLite messages     (LIKE / date search)
         archival_*        → Chroma              (local hashing embedding)
       each result → role:"tool" message
       request_heartbeat decides: loop again, or stop
```

### What survives a restart

Worth being precise about, since it's the headline claim:

- The **in-context FIFO** (`AgentLoop.messages`) lives in Streamlit session
  state and is **lost on restart**. It is never rehydrated from recall.
- **Core memory survives** and is re-injected into the system prompt on the very
  first turn of the new session. *That* is why "restart the app and it still
  knows your name" works — and why the system prompt pushes the agent to promote
  durable facts into the `human` block.
- **Recall and archival survive too**, but they're pull-only: the agent must
  decide to go search them.

So the demo isn't a persistence trick — it's the MemGPT thesis in miniature.
What the agent chose to promote into core memory is what it wakes up knowing;
everything else it has to go looking for.

---

## Database schema

```sql
CREATE TABLE core_blocks (
  id          INTEGER PRIMARY KEY,
  name        TEXT UNIQUE CHECK(name IN ('persona','human')),
  content     TEXT NOT NULL,
  char_limit  INTEGER DEFAULT 2000,
  updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
  id          INTEGER PRIMARY KEY,
  session_id  TEXT NOT NULL,
  role        TEXT NOT NULL,   -- user | assistant | tool | system_event
  event_type  TEXT,            -- message | tool_call | tool_result
                               -- | pressure_warning | internal
  served_by   TEXT,            -- gemini | groq | openrouter | mistral | NULL
  content     TEXT NOT NULL,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_created ON messages(created_at);
CREATE INDEX idx_messages_event   ON messages(event_type);

CREATE TABLE provider_usage (
  provider       TEXT NOT NULL,
  usage_date     DATE NOT NULL,   -- UTC
  requests       INTEGER DEFAULT 0,
  tokens         INTEGER DEFAULT 0,
  cooldown_until TIMESTAMP,
  PRIMARY KEY (provider, usage_date)
);
```

Archival passages live in Chroma with `{id, created_at, source}` metadata.
Phase 3 migrates them to
`archival_passages(id, content, embedding vector(384), sensitive, created_at)` —
384 dims to match `bge-small-en-v1.5`.

`SQLiteStore` opens one connection with `check_same_thread=False` and serializes
every write behind a lock, because Streamlit re-runs the script on different
threads within a session. Schema init is idempotent and includes a small
migration for DBs created before `served_by` existed.

---

## Setup

Requires **Python 3.11+** and **at least one free provider key**. Gemini alone
works; Gemini + Groq is the ideal Phase 1 pair (you can watch failover happen).

```bash
# 1. Install
pip install -e ".[dev]"          # or: pip install -r requirements.txt

# 2. Get free keys — no credit card
#    Gemini     → https://aistudio.google.com/apikey
#    Groq       → https://console.groq.com/keys
#    OpenRouter → https://openrouter.ai/keys
#    Mistral    → https://console.mistral.ai/api-keys

# 3. Configure
cp .env.example .env             # then paste your key(s)
```

`.env` holds `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
`MISTRAL_API_KEY`. It is gitignored and must never be committed. Providers with
no key are skipped automatically — the app runs on whatever you have.

> **Windows note:** if `python` isn't on PATH, use the full interpreter path
> (e.g. `C:\Users\<you>\miniconda3\python.exe`), or pass it to make:
> `make dev PY="C:/Users/<you>/miniconda3/python.exe"`.

---

## Run

```bash
make dev        # or: python -m streamlit run app/streamlit_app.py
make test       # or: python -m pytest        (56 tests, no keys, no network)
make mcp        # Phase 2 MCP server (currently a stub — exits 1)
make clean      # remove caches and local data/
```

If you launch with no keys set, the UI shows an onboarding gate and accepts a
Gemini key inline for the session.

### The sidebar

| Panel | Shows |
|---|---|
| **Core memory** | Live `persona` and `human` blocks — watch them change as the agent edits itself |
| **Context usage** | Progress bar against the active provider's window; warns at the pressure threshold |
| **Memory tiers** | Recall message count, archival passage count |
| **Providers** | ✅/⛔ per provider with the reason (`cooling_down`, `daily_requests_exhausted`, `no_api_key`) and seconds remaining |

Every assistant reply is badged `⚡ served by groq`. **Reset conversation**
clears the in-context window but keeps all saved memory — the fastest way to see
the persistence claim demonstrated.

---

## Configuration reference

All knobs live in [`config.py`](config.py), each overridable by env var:

| Env var | Default | Meaning |
|---|---|---|
| `MEMASSIST_TEMPERATURE` | `0.3` | Fixed across every provider, for voice stability |
| `MEMASSIST_CONTEXT_LIMIT` | `32000` | Planning cap for the pressure signal. `0` disables the cap and uses the real provider window. Keeping it modest makes the eviction mechanic observable in a short chat |
| `MEMASSIST_PRESSURE_THRESHOLD` | `0.7` | Fraction of the limit that triggers the warning |
| `MEMASSIST_MAX_HEARTBEATS` | `5` | Max chained tool rounds per turn. Each is a real API request |
| `MEMASSIST_CORE_BLOCK_CHAR_LIMIT` | `2000` | Per-block cap for core memory |
| `MEMASSIST_ARCHIVAL_TOP_K` | `5` | Default archival results |
| `MEMASSIST_SEARCH_PAGE_SIZE` | `5` | Recall search page size |
| `MEMASSIST_TOOL_RESULT_CHAR_CAP` | `4000` | Truncation cap on all tool results |
| `MEMASSIST_DB_PATH` | `data/memassist.db` | SQLite location |
| `MEMASSIST_CHROMA_PATH` | `data/chroma` | Chroma persistence directory |
| `MEMASSIST_PROVIDERS_YAML` | `llm/providers.yaml` | Provider chain config |

`DEFAULT_PERSONA` and `DEFAULT_HUMAN` seed core memory on first run only —
after that the agent owns those blocks.

---

## Testing

**56 tests. No API keys. No network. Sub-second.**

That's possible because of three deliberate injection seams:

| Seam | Makes testable |
|---|---|
| `Router(client_factory=…)` | The entire failover algorithm, against fake clients |
| `BudgetLedger(now_fn=…)` | Cooldown expiry, UTC-midnight rollover, daily reset |
| `ArchivalStore(embed_fn=…)` | Vector behavior without a model download |

Plus one rule: **memory functions are deterministic and contain no LLM calls**,
so they're unit-tested directly.

| Suite | Count | Covers |
|---|---|---|
| `test_router.py` | 21 | Failover per status code, proactive skipping, budget exhaustion, lane routing, tool normalization, 400-propagation, transient-400 failover, YAML loading |
| `test_sqlite.py` | 9 | Block seeding, append/replace, char-limit overflow, verbatim-miss errors, keyword + date search, pagination |
| `test_memory_tools.py` | 9 | Validated dispatch, invalid enums, unknown tools, truncation, stats, core rendering |
| `test_budgets.py` | 6 | Usage recording, cooldown expiry, UTC-day rollover clearing both usage *and* midnight cooldowns |
| `test_chroma.py` | 5 | Embedding determinism + normalization, insert/search/count, empty store, persistence across reopen |
| `test_loop.py` | 5 | Tool-then-send, heartbeat cap, text fallback, pressure injection, per-provider window sizing |
| `test_prompts.py` + `test_token_budget.py` | 6 | Prompt assembly, usage math |

Not yet covered: **live per-provider tool-calling integration tests** (Phase 4).
That gap is exactly what produced the Gemini 3.x `thought_signature` surprise and
the Groq `tool_use_failed` surprise — both now encoded as a model pin and a
string match, but discovered the hard way.

---

## Repo layout

```
memassist/
  CLAUDE.md              Project instructions for AI coding agents
  PROJECT_SPEC.md        Full architecture + phased build plan — read before implementing
  README.md              This file
  .mcp.json              MCP server registration (Phase 2 target)
  .env / .env.example    Provider keys

  config.py              Every knob, env-overridable
  assembly.py            Composition root
  conftest.py            Makes the repo root importable for tests

  llm/
    router.py            Failover algorithm, normalization, provider tagging
    budgets.py           provider_usage ledger — proactive skipping
    errors.py            Exception classification (what fails over, what doesn't)
    providers.yaml       The chain, as data
  agent/
    loop.py              Heartbeat loop, dispatch, pressure injection
    prompts.py           System-prompt assembly
    token_budget.py      Pure usage math
  memory_server/
    __main__.py          Phase 2 FastMCP entry point (stub)
    memory_tools.py      The six tools + validated dispatch
    schemas.py           Pydantic models ↔ OpenAI tool definitions
    storage/
      sqlite.py          Core + recall
      chroma.py          Archival + local embedder
  app/
    streamlit_app.py     Phase 1 UI
  tests/                 56 tests
  data/                  SQLite DB + Chroma store (gitignored)
  Makefile
```

---

## Build phases

### Phase 1 — MVP ✅ **complete**

**Goal:** a working MemGPT agent on free tiers, with the memory mechanics
visible.

Delivered:

- SQLite schema: `core_blocks`, `messages`, `provider_usage`
- Core memory with char limits and instructive error messages
- Recall memory with keyword + date search and pagination
- Chroma archival memory with a local offline embedder
- All seven tools, Pydantic-validated, flat schemas
- Failover router across all four providers with a persistent budget ledger and
  proactive skipping
- Heartbeat-driven agent loop with per-provider context sizing and
  memory-pressure injection
- Streamlit chat with a live memory sidebar, context bar, provider status panel,
  and a `served_by` badge on every reply
- 56 tests, no keys, no network

**Definition of done — both verified:**
1. Tell it your name, restart the app → it still knows.
2. Remove `GEMINI_API_KEY` → replies seamlessly arrive from Groq.

The live database confirms failover in production use: 44 user turns, 32 tool
calls, served by groq (57), openrouter (19), and mistral (2) as the chain
degraded across days.

---

### Phase 2 — MCP + full provider chain ⬜ **not built**

**Goal:** the memory tools become a real Model Context Protocol server, so the
*same* memory is usable by this agent **and** by Claude Code / Claude Desktop.

Planned work:

- **FastMCP server** (`memgpt-memory`, stdio transport) in
  `memory_server/__main__.py`, exposing the six memory tools — everything except
  `send_message`, which is the agent's own channel, not a memory operation.
- **MCP client bridge** in the agent loop: `list_tools()` → convert to OpenAI
  tool schemas → on tool use, `session.call_tool()` → feed the result back as a
  tool result. `MemoryTools` stays as the server-side implementation; only the
  `MemoryInterface` implementation the loop holds changes.
- **Verify via [`.mcp.json`](.mcp.json)** that Claude Code can drive the same
  memory server. This is the demo-worthy payoff: your assistant's long-term
  memory becomes a tool any MCP client can use.
- **Proactive-skipping polish** — notably enforcing `rpm_limit`, which is parsed
  today but not checked (see [deviations](#known-deviations-from-the-spec)).
- **`sentence-transformers` embeddings** (`bge-small-en-v1.5`, 384-dim) replacing
  the hashing embedder via the existing `embed_fn` seam. Requires a re-embed
  migration of existing passages.

Security note carried into this phase: **tool results are untrusted text.** Never
eval, always length-cap. That already holds in Phase 1 and must survive the move
across a process boundary.

**Done when:** the Streamlit agent and Claude Code both read and write the same
core/recall/archival memory through the MCP server.

---

### Phase 3 — production stack ⬜ **not built**

**Goal:** move off single-file storage and a single-user UI.

Planned work:

- **PostgreSQL + pgvector** replacing SQLite + Chroma.
  `archival_passages(id, content, embedding vector(384), sensitive, created_at)`
  — the 384 dims are why the embedder swap belongs in Phase 2, before the data
  migrates.
- **FastAPI backend** with **SSE streaming**, replacing Streamlit's
  request/rerun cycle. Token-by-token replies, plus a live event stream for
  memory operations.
- **Next.js + Tailwind** frontend with a live **memory-inspector panel**: watch
  core blocks mutate, archival passages appear, and the provider badge flip —
  in real time, mid-turn. This is the version worth showing people.
- **MCP over Streamable HTTP** instead of stdio, so the memory server is a
  network service rather than a subprocess.
- **Docker Compose**: `web`, `api`, `memory-mcp`, `db`.

The Phase 1 seams are what make this tractable: `agent/loop.py` doesn't change,
because it never knew what storage was.

---

### Phase 4 — background intelligence & evaluation ⬜ **not built**

**Goal:** memory that improves itself, and evidence that it works.

Planned work:

- **Mistral background consolidation lane.** `router.chat_background()` already
  exists and routes to Mistral only. Phase 4 adds `jobs/consolidate.py`: nightly
  summarization, archival compaction, merging redundant passages. 2 RPM is
  irrelevant for batch work.
- **The `sensitive` flag — a hard prerequisite.** Mistral's free tier trains on
  prompts. Archival passages need a `sensitive` boolean, and background jobs must
  send only non-sensitive content. **Do not wire the background lane until this
  lands** — it's the privacy gate, not a nice-to-have.
- **Deep-memory-retrieval eval harness.** Seed facts in session 1, query them in
  session N, measure recall. This is the only way to know whether the memory
  architecture actually works rather than merely appearing to.
- **Langfuse tracing, tagged by provider.** Which provider served which turn,
  what it cost in tokens, where latency went. The `served_by` tag already
  threads through everything — this surfaces it as observability.
- **Per-provider tool-calling CI tests.** Integration-test every tool schema
  against every provider. The two hard-won lessons already in the codebase
  (Gemini's `thought_signature`, Groq's `tool_use_failed`) argue strongly for
  catching the next one automatically.

---

### Non-goals (v1)

Multi-user auth, voice, RAG over user documents, fine-tuning, paid tiers.

---

## Known deviations from the spec

Honest accounting — these are the gaps between [`PROJECT_SPEC.md`](PROJECT_SPEC.md)
and what actually runs today.

| # | Gap | Impact | Phase |
|---|---|---|---|
| 1 | **MCP server is a stub** — `__main__.py` prints a notice and exits 1. `.mcp.json` registers it, so a Claude Code launch fails today | Phase 2 headline feature absent | 2 |
| 2 | **`rpm_limit` parsed but never enforced.** `_availability()` checks daily limits and cooldown only; RPM breaches are handled reactively via the 429 → 60s path | Wasted requests, notably on Mistral at 2 RPM | 2 |
| 3 | **Memory pressure warns but never evicts.** `loop.messages` grows unbounded; nothing truncates the FIFO. The spec's "~50% for the FIFO queue" split isn't implemented | If the model ignores the warning you eventually hit a hard context error rather than degrading gracefully | 2 |
| 4 | **Hashing embedder, not `bge-small-en-v1.5`.** 512-dim bag-of-tokens rather than 384-dim semantic | Archival recall is lexical, not semantic. Also a dimension mismatch with the planned `vector(384)` column | 2 |
| 5 | **No `sensitive` flag** on archival passages | Blocks the Mistral background lane on privacy grounds | 4 |
| 6 | **Pressure warning injected as `role:"user"`**, not a system message — pragmatic, since mid-conversation system messages behave inconsistently across the four providers | The recall log's `system_event` row and the in-context role disagree | — |
| 7 | Gemini pinned to `gemini-2.0-flash`, not the spec's `gemini-2.5-flash` | Deliberate and documented in the YAML — 3.x breaks flat tool schemas | — |
| 8 | `jobs/` directory from the spec layout doesn't exist | `chat_background()` has no caller yet | 4 |

---

## Troubleshooting

**`AllProvidersExhausted`** — the exception carries a per-provider reason
(`no_api_key`, `cooling_down`, `daily_requests_exhausted`, `rate_limited`,
`quota_exceeded`, `auth_error`). Check the sidebar's provider panel; most often
it's every key missing, or a shared daily budget genuinely spent. Cooldowns clear
on their own — at UTC midnight for quota, after 60s for rate limits.

**A 400 propagates instead of failing over** — that's intentional. A malformed
request would fail on all four providers, so surfacing it beats masking it.
Usually a tool schema problem.

**Replies stop mid-work** — you probably hit the heartbeat cap (5). Either the
task genuinely needs more chained calls (raise `MEMASSIST_MAX_HEARTBEATS`,
knowing each round costs a request), or the model is looping on tool calls
without ever calling `send_message`.

**Archival memory unavailable** — the sidebar shows a warning if Chroma failed
to init. This degrades gracefully by design: [`assembly.py`](assembly.py) catches
the failure and the assistant runs on core + recall alone.

**Agent forgot something you told it** — check whether it was ever written to
core memory (visible in the sidebar). If it only lived in the conversation, it's
in recall and must be *searched* for, not remembered. That's the architecture
working as designed, not a bug — though it's also a fair prompt-engineering
signal that the persona block should push harder on saving.

---

## References

- **MemGPT: Towards LLMs as Operating Systems** — Packer et al., 2023,
  [arXiv:2310.08560](https://arxiv.org/abs/2310.08560). The source of the memory
  hierarchy, self-editing tools, and heartbeat mechanics.
- **Letta** (formerly `memgpt`) — reference implementation for memory and
  heartbeat patterns. *Consult, never copy.*
- **LiteLLM Router** — reference for provider failover patterns. *Consult, never
  copy.*
- **Model Context Protocol** — https://modelcontextprotocol.io (Phase 2).

See [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full architecture, exact tool
definitions, and the complete phased build plan.
