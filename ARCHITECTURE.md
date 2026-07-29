# ARCHITECTURE — MemAssist (as-built)

This document describes MemAssist **as the code actually is** at the end of the
Phase 1.5 fix sprint (deterministic bench 100/100). It is a developer map: the
file layout, the data flows through one turn, and the failure-handling designs —
in particular the Gemini `ProviderPermanentError` fix.

- Rationale and the phased plan live in [`PROJECT_SPEC.md`](PROJECT_SPEC.md).
- User-facing setup/usage lives in [`README.md`](README.md).
- Scored capabilities and the fix-by-fix run log live in [`BENCHMARKS.md`](BENCHMARKS.md).

---

## 1. File map

```
memassist/
  config.py               Central settings (env-overridable); loads .env
  assembly.py             Composition root — build_memory / build_router / build_loop
  conftest.py             Puts repo root on sys.path for tests

  agent/                  LLM-facing loop. Imports NO storage and NO provider SDK.
    loop.py               AgentLoop: turn cycle, tool dispatch, pressure + FIFO eviction
    prompts.py            System-prompt assembly (identity, tiers, provenance, pressure)
    token_budget.py       Pure context-usage math (approx_tokens, is_under_pressure, …)

  llm/                    The failover router. The ONLY place a provider is called.
    router.py             Router.chat(): priority chain, cooldowns, disable, served_by
    errors.py             classify_provider_error() → the router's failure taxonomy
    budgets.py            BudgetLedger over the provider_usage table (UTC-daily reset)
    providers.yaml        The provider chain as data (slug, base_url, budgets, lanes)

  memory_server/          Memory tiers + the six tools (the MemoryInterface impl).
    memory_tools.py       MemoryTools: the six tools, validated dispatch, provenance
    schemas.py            Pydantic input models ↔ flat OpenAI function schemas
    __main__.py           Phase 2 FastMCP entry point (stub)
    storage/
      sqlite.py           core_blocks + messages (recall); date-bound validation
      chroma.py           ArchivalStore: persistent Chroma, cosine, source metadata
      embedder.py         Local bge-small-en-v1.5, 384-d, L2-normalized (lazy load)
      migrate_embeddings.py   One-time 512-d → 384-d re-embed migration

  app/streamlit_app.py    Phase 1 chat UI: live memory sidebar + provider badge

  bench/                  Deterministic, offline benchmark (100 pts). `python -m bench`
  tests/                  pytest suite (60 tests, no API keys, no provider calls)

  Makefile  pyproject.toml  requirements.txt  ci.yml
  .mcp.json  mcp_servers.yaml   MCP registration (server is Phase 2)
```

### Two seams, both `Protocol` types in `agent/loop.py`

The loop depends on nothing concrete. It talks to exactly two interfaces:

| Protocol | Methods | Concrete impl | Swapped by |
|---|---|---|---|
| `LLMRouter` | `chat`, `context_window`, `min_context_window` | `llm.router.Router` | (stable) |
| `MemoryInterface` | `render_core_memory`, `memory_stats`, `dispatch`, `record_event` | `memory_server.memory_tools.MemoryTools` | Phase 2 → MCP client |

Because the loop sees only these, later phases are local changes: Phase 2 swaps
the `MemoryInterface` implementation for an MCP client; Phase 3 swaps SQLite for
Postgres beneath it. Neither edits `agent/loop.py`.

`assembly.py` is the only module that wires concrete parts together from
`config.py`.

---

## 2. Data flow — one interactive turn

Entry point: `AgentLoop.step(user_text) -> list[str]` (the strings sent to the
user via `send_message`). The Streamlit app and the benchmark both drive this
same method.

```
user_text
   │  record ("user","message") to recall; append to FIFO self.messages
   │  if under_pressure(): inject "[MEMORY PRESSURE] …" + record pressure_warning
   ▼
┌─ heartbeat loop (cap = max_heartbeats, default 5) ───────────────────────────┐
│  system = BASE_SYSTEM_PROMPT + rendered core memory + memory stats + usage%   │
│  result = router.chat([system, *self.messages], tools=ALL_TOOLS)              │
│     └─ on errors.AllProvidersExhausted → return PROVIDERS_EXHAUSTED_MESSAGE    │
│  self.served_by = result.served_by ; update last_input_tokens / last_limit    │
│  append assistant message (content + tool_calls) to FIFO                       │
│                                                                                │
│  for each tool_call:                                                           │
│     send_message        → collect text as output; record ("assistant",…)       │
│     memory tool         → text = memory.dispatch(name, args); append tool msg  │
│        └─ archival_memory_insert while under pressure → set `offloaded` flag   │
│                                                                                │
│  if offloaded: _evict_offloaded()   # the paging half of MemGPT (see §5)       │
│                                                                                │
│  stop when: a message was sent and no heartbeat requested, OR cap hit,         │
│             OR the model produced plain text with no tool call (surfaced)      │
└───────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  every event (user / assistant / tool_call / tool_result / pressure_warning /
      eviction) is persisted to the recall log, tagged with served_by.
outputs
```

Key invariants:
- **`send_message` is the only path to the user.** Any other assistant text is
  internal (recorded as `event_type="internal"`), surfaced only as a fallback if
  the model ends a turn without calling `send_message`.
- **Tool results are strings, always.** `MemoryTools.dispatch` validates arguments
  through the tool's Pydantic model and turns every failure into an `"Error: …"`
  string the model can read and retry — it never raises into the loop.
- **Context % is measured against the ACTIVE provider's window** (`_limit_for`),
  capped by the planning limit, because Groq/Mistral windows ≪ Gemini's.

---

## 3. Memory tiers & the tool contract

| Tier | Store | In context? | Tools |
|---|---|---|---|
| Core | SQLite `core_blocks` (`persona`, `human`) | Always — rendered into the system prompt each turn | `core_memory_append`, `core_memory_replace` |
| Recall | SQLite `messages` (append-only, `served_by`) | On demand | `conversation_search`, `conversation_search_date` |
| Archival | Chroma (cosine, 384-d) | On demand (semantic) | `archival_memory_insert`, `archival_memory_search` |

Tool schemas are **flat** (no nested objects) so all four provider tool-calling
dialects accept them (`schemas.py`). Each tool has a Pydantic model for
validation and a hand-written JSON schema for the model, colocated so they can't
drift.

**Provenance.** `core_memory_append` and `archival_memory_insert` carry a flat
`source: "stated" | "inferred"` (default **`inferred`**). Human-block lines are
stored tagged — `"Works as a nurse. [stated]"` — and archival passages record
`source` in Chroma metadata. The `persona` block is deliberately untagged (it is
the agent's own identity). If the model copies the rendered tag into `content`,
`_PROVENANCE_TAG_RE` strips it — the `source` argument is the only authority.

---

## 4. Context pressure & FIFO eviction (the MemGPT paging mechanic)

`token_budget.py` is pure math. Each turn the loop compares the last prompt's
input tokens against `min(active_provider_window, planning_limit)`. Crossing the
threshold (default 0.7) sets `under_pressure()`.

Phase 1 did the *summarizing* half (offload to archival) but never the *paging*
half, so the FIFO only grew and the warning re-fired forever. As-built now
(`agent/loop.py`):

1. Under pressure the agent calls `archival_memory_insert` to summarize the
   oldest turns; a successful insert sets an `offloaded` flag.
2. After the tool round, `_evict_offloaded()` drops the oldest ~half of the FIFO
   (never below `MIN_MESSAGES_BEFORE_EVICTION = 8`), leaves a `[CONTEXT EVICTED]`
   marker at the head, and records an `eviction` event to recall for audit.
3. `_safe_cut()` advances the cut past any orphaned `tool` messages — evicting an
   assistant message while keeping its tool replies produces a transcript every
   OpenAI-compatible provider rejects.
4. `_recompute_usage()` re-estimates tokens from the *retained* queue (the
   provider's `prompt_tokens` describes the pre-eviction prompt); the real count
   supersedes the estimate on the next call.

Net effect: offloading frees context in the same turn and pressure clears.

---

## 5. The failover router

`Router.chat(messages, tools)` walks the priority-ordered chain from
`providers.yaml` (Gemini → Groq → OpenRouter → Mistral). For each provider:

```
if not _availability(cfg):            # skip, with a reason
    record reason; continue
try:
    raw = _call_with_retry(cfg, …)    # 5xx: retry once, then raise ProviderServerError
except RateLimitError:                # transient 429
    ledger.cooldown_for_seconds(60);  record "rate_limited";  continue
except QuotaExceededError:            # 402
    ledger.cooldown_until_utc_midnight();  record "quota_exceeded";  continue
except ProviderServerError:           # 5xx / transport / transient tool_use_failed
    record "server_error";  continue
except ProviderPermanentError as e:   # see §6 — NEVER cooled
    self._disabled[cfg.name] = str(e);  log ERROR;  continue
record usage; return ChatResult(served_by=cfg.name, …)
# nothing served:
raise AllProvidersExhausted(attempts)  # loop turns this into a friendly message
```

`_availability(cfg)` skips a provider that (in order) has no key, is in
`_disabled` (→ `"permanently_unavailable (…)"`), is cooling down, or is over its
daily request/token budget. All of this surfaces in `provider_status()` (the UI
badge) and in the `AllProvidersExhausted.attempts` map.

- **Budgets persist** in the `provider_usage` SQLite table (`budgets.py`) and
  reset at UTC midnight by construction (usage is keyed by UTC date).
- **`served_by`** is stamped on the `ChatResult` and recorded on every event.
- **Background lane:** `chat_background()` routes straight to Mistral (spec §3.2).
- Cross-provider consistency: fixed temperature, flat tool schemas, and a **local**
  embedder so provider rotation never fragments the archival vector space.

---

## 6. The Gemini `ProviderPermanentError` design (fix 1)

### The bug it fixes ("Gemini 0-req")

Google returns **HTTP 429 for two very different conditions**:

1. *"You are going too fast"* — a real, transient rate limit.
2. *"This model has no free-tier allowance on your project"* — the quota-metric
   line reads `limit: 0`.

`providers.yaml` originally pinned `gemini-2.0-flash`, which has **zero** free-tier
allowance on this project, so **every** call returned a `limit: 0` 429. The
classifier treated all 429s as a transient `RateLimitError`, so the router put
priority-1 Gemini on a rolling 60s cooldown — forever. Gemini looked merely
"busy", served **zero** requests through all of Phase 1, and never once reported
why. A wrong model slug (404) had the same invisibility.

### The design

The distinction is made **at classification time**, in
`llm/errors.py::classify_provider_error`, and a new exception type carries it:

```
ProviderError
├─ RateLimitError        429, transient           → cooldown 60s, fail over
├─ QuotaExceededError    402                       → cooldown until UTC midnight
├─ ProviderServerError   5xx / transport / 400 tool_use_failed → retry once, fail over
└─ ProviderPermanentError   NEVER cooled down
   └─ ProviderAuthError  401 / 403
```

`ProviderPermanentError` is returned for:
- a **429 whose body matches `_ZERO_QUOTA_MARKERS`** (`"limit: 0"`, …) — zero
  allowance, not backpressure;
- a **404 or a model-not-found marker** — wrong/retired/disabled slug;
- **401 / 403** (`ProviderAuthError`) — bad or missing credentials.

Anything else keeps its existing meaning; an unclassified 4xx (e.g. a genuine
400) is **re-raised**, never silently failed over across all four providers.

### What the router does with it

In `Router.chat`, a `ProviderPermanentError` is **not cooled down** — a cooldown
implies "try again later", which is false and is exactly what hid the bug. Instead
the router:

1. adds the provider to `self._disabled` for the life of the process,
2. logs at **ERROR** with the reason (e.g. *"zero free-tier quota for this model
   (HTTP 429, limit: 0) — change the model slug in providers.yaml or enable
   billing"*), and
3. keeps failing over, so the remaining providers still serve the turn.

`provider_status()` then reports the provider as
`permanently_unavailable (<reason>)`, and if nothing else can serve, the distinct
error type propagates in `AllProvidersExhausted`.

**Deliberate deviation from the brief:** the spec said permanent faults should
"raise loudly, no cooldown". Raising immediately would let one misconfigured
provider kill a turn the other three could serve, so the router *disables + logs
at ERROR + fails over* — loud and uncooled, but not fatal.

### The other half of the fix

Classification alone doesn't make Gemini answer — the slug was wrong.
`providers.yaml` is repointed to **`gemini-2.5-flash-lite`**, verified against the
live endpoint to return 200 with **standard flat `tool_calls`** (Gemini 3.x
models like `gemini-flash-latest` were rejected: they require round-tripping a
Google-specific `thought_signature` that breaks the flat cross-provider tool
format). Re-verify with `make bench LIVE=1`.

---

## 7. Storage & persistence

`core_blocks` (persona/human + char limit) and `messages` (recall log with a
`served_by` column and an idempotent add-column migration) live in one SQLite DB;
`provider_usage` (the budget ledger) lives in the same DB. Date-range recall
search validates its bounds via `_normalize_date_bound` and raises `ValueError`
on malformed input, which `conversation_search_date` turns into a correctable
`"Error: …"` string (the T3b fix).

Archival passages live in a persistent Chroma collection (cosine space). Vectors
come from `embedder.py` — a local **`BAAI/bge-small-en-v1.5`** (384-d,
L2-normalized, lazily loaded and cached per process, no API key). The Phase 1
signed-hashing embedder is retained as an injectable zero-dependency fallback.
`migrate_embeddings.py` rebuilds the collection when the vector width changes
(512-d → 384-d), from the stored passage text — no LLM, no regenerated content.

---

## 8. Verification

- **`make test`** → pytest, 60 tests, no API keys and no provider calls. The
  archival tests exercise the real local bge-small embedder (cached after a
  one-time download), so a cold run pays that load once.
- **`make bench`** → the deterministic 100-point suite in `bench/` (every
  provider call is a scripted fake; every check gets a fresh temp SQLite +
  Chroma). Reproducible, so a score delta is attributable to a source change.
- **`make bench LIVE=1`** → adds one real request per configured provider,
  reported but **never scored**, so CI stays deterministic.

Current: **bench 100/100**, pytest green.

---

## 9. Known deviations & not-yet-built

- Embeddings are local by design (never provider-hosted) so rotation can't
  fragment the vector space.
- `ProviderPermanentError` disables-and-fails-over rather than raising immediately
  (see §6).
- **Phase 2+ (not built):** the FastMCP server (`memory_server/__main__.py` is a
  stub), the async MCP client bridge, Postgres/pgvector, FastAPI/Next.js, and
  Langfuse tracing. The `MemoryInterface` seam is what keeps those additive.
