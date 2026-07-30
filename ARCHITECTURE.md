# ARCHITECTURE — MemAssist (as-built)

This document describes MemAssist **as the code actually is** at the end of
Phase 4 (deterministic bench 110/110, on either storage backend). It is a developer map: the
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

  agent/                  Thin adapter + prompt/budget helpers. No storage, no SDK.
    loop.py               AgentLoop: step()/resume(), state views over the checkpoint
    prompts.py            System-prompt assembly (tiers, provenance, untrusted rule)
    token_budget.py       Pure context-usage math (approx_tokens, is_under_pressure, …)

  graph/                  LangGraph owns CONTROL FLOW only (spec §4).
    state.py              AgentState (turn state) + Deps (router/memory/limits/allowlist)
    nodes.py              The 8 nodes: build_prompt … security_gate … respond
    graph.py              StateGraph wiring, pressure edge, heartbeat cycle, recursion cap

  security/               The AI security layer (spec §6).
    sanitizer.py          Untrusted results -> marked data; 7 injection patterns
    guards.py             Core-memory lockout, source=external, allowlist, path jail
    injections/*.yaml     T11 red-team corpus — read by BOTH bench and CI

  mcp_client.py           External MCP servers from mcp_servers.yaml; trust zones

  llm/                    The failover router. The ONLY place a provider is called.
    router.py             Router.chat(): priority chain, cooldowns, disable, served_by
    errors.py             classify_provider_error() → the router's failure taxonomy
    budgets.py            BudgetLedger over the provider_usage table (UTC-daily reset)
    providers.yaml        The provider chain as data (slug, base_url, budgets, lanes)

  memory_server/          Memory tiers + the six tools (the MemoryInterface impl).
    memory_tools.py       MemoryTools: the six tools, validated dispatch, provenance
    schemas.py            Pydantic input models ↔ flat OpenAI function schemas
    __main__.py           FastMCP server (stdio): the 6 tools, for Claude Code
    storage/
      sqlite.py           core_blocks + messages (recall); date-bound validation
      chroma.py           ArchivalStore: persistent Chroma, cosine, source metadata
      postgres.py         PostgresStore: the same surface, Postgres dialect
      pgvector_store.py   PgVectorStore: vector(384), cosine, SQL paging
      migrate_to_postgres.py  SQLite+Chroma -> Postgres, idempotent
      embedder.py         Local bge-small-en-v1.5, 384-d, L2-normalized (lazy load)
      migrate_embeddings.py   One-time 512-d → 384-d re-embed migration

  api/                    FastAPI service (Phase 4).
    main.py               SSE /chat, approve/deny, memory-inspector reads
    sessions.py           session id -> AgentLoop -> checkpointer thread
  web/                    Next.js + Tailwind UI; PARITY.md tracks the cutover

  bench/                  Deterministic, offline benchmark (110 pts). `python -m bench`
  tests/                  pytest suite (120 tests, no API keys, no provider calls)
  workspace/              The filesystem server's jail — nothing else is reachable

  Dockerfile  web/Dockerfile  docker-compose.yml
  Makefile  pyproject.toml  requirements.txt  .github/workflows/ci.yml
  .mcp.json  mcp_servers.yaml   MCP registration (server is Phase 2)
```

### Two seams, both `Protocol` types in `agent/loop.py`

The loop depends on nothing concrete. It talks to exactly two interfaces:

| Protocol | Methods | Concrete impl | Swapped by |
|---|---|---|---|
| `LLMRouter` | `chat`, `context_window`, `min_context_window` | `llm.router.Router` | (stable) |
| `MemoryInterface` | `render_core_memory`, `memory_stats`, `dispatch`, `record_event` | `memory_server.memory_tools.MemoryTools` | (stable) |
| `ExternalToolset` | `names`, `trust_of`, `is_gated`, `jail_of`, `call` | `mcp_client.ExternalTools` | (stable; stdio or HTTP) |

Because the loop sees only these, later phases are local changes: Phase 2 swaps
the `MemoryInterface` implementation for an MCP client; Phase 3 swaps SQLite for
Postgres beneath it. Neither edits `agent/loop.py`.

`assembly.py` is the only module that wires concrete parts together from
`config.py`.

---

## 2. Data flow — one interactive turn

Entry point: `AgentLoop.step(user_text) -> list[str]` (the strings sent to the
user via `send_message`). The Streamlit app and the benchmark both drive this
same method. Since Phase 2 the steps below are graph **nodes** (§10), not inline
loop code, but the sequence is unchanged.

```
user_text
   │  record ("user","message") to recall; append to the FIFO in graph state
   ▼
┌─ heartbeat cycle (cap = max_heartbeats, default 5) ──────────────────────────┐
│  build_prompt    system = BASE_SYSTEM_PROMPT + core memory + stats + usage%   │
│  pressure_check  first pass and >= threshold? -> inject_warning               │
│  call_llm        router.chat([system, *messages], tools)                      │
│     └─ on errors.AllProvidersExhausted -> PROVIDERS_EXHAUSTED_MESSAGE          │
│                  served_by / input_tokens / limit updated; assistant appended  │
│                                                                                │
│  security_gate   per tool_call: allowlist, core-memory lockout, source=        │
│                  external, path jail; interrupt() for write-gated tools        │
│  dispatch_tools  send_message  -> collect output; record ("assistant",…)       │
│                  memory tool   -> memory.dispatch(name, args)                  │
│                  external tool -> external.call(...); VERBATIM result to recall │
│                     └─ archival_memory_insert under pressure -> evict (§4)      │
│  sanitize_results untrusted results rewritten as marked data; saw_untrusted    │
│                                                                                │
│  stop when: a message was sent and no heartbeat requested, OR cap hit,         │
│             OR the model produced plain text with no tool call (surfaced)      │
└───────────────────────────────────────────────────────────────────────────────┘
   │
   ▼  every event (user / assistant / tool_call / tool_result / pressure_warning /
      eviction / security) is persisted to the recall log, tagged with served_by.
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

- **`make test`** → pytest, 120 tests, no API keys, no provider calls and no
  MCP subprocesses. Unit tests inject the hashing embedder as a test double, so
  CI needs no bge-small download; the benchmark keeps the real model, which is
  what T5 measures.
- **`make bench`** → the deterministic 110-point suite in `bench/` (every
  provider call is a scripted fake; every check gets a fresh temp SQLite +
  Chroma). Reproducible, so a score delta is attributable to a source change.
- **`make bench LIVE=1`** → adds one real request per configured provider,
  reported but **never scored**, so CI stays deterministic.
- **CI** (`.github/workflows/ci.yml`): ruff → pytest → gitleaks → pip-audit.
  The audit ignores PYSEC-2026-311 (chromadb): the RCE needs the `/api/v2` HTTP
  endpoint with `trust_remote_code`, and Chroma runs embedded here with no
  server. No fixed release exists, so without the ignore the gate can never pass.

Current: **bench 110/110** (T1–T8 100, T11 10), pytest green, CI green.

---

## 9. Known deviations & not-yet-built

- Embeddings are local by design (never provider-hosted) so rotation can't
  fragment the vector space.
- `ProviderPermanentError` disables-and-fails-over rather than raising immediately
  (see §6).
- **`memgpt-memory` is dispatched in-process**, not over its own stdio server.
  It is `trust=internal` by definition, so a subprocess hop buys no isolation
  while adding latency and non-determinism to the benchmark. The stdio server is
  real and registered in `.mcp.json` — it exists for external clients.
- **`mcp` is pinned `<2`**: `langchain-mcp-adapters` 0.3.1 imports
  `RequestContext` from `mcp.shared.context`, which mcp 2.0 removed.
- **Schema budget.** The filesystem server alone exports 14 tools, so 16
  external schemas now ride in every prompt. Spec §5.1 caps active servers at 3
  for exactly this reason; trimming to the tools actually used is the obvious
  next economy.
- **Phase 5 (not built):** the Mistral consolidation lane with the `sensitive`
  filter (T10), Langfuse tracing tagged by provider, and the CD deploy
  pipeline. The `MemoryInterface` and `ExternalToolset` seams are what keep
  those additive.
- **The checkpointer is still `InMemorySaver`.** Turn state does not survive
  an API restart and a second replica would not see it — both move together
  when a Postgres checkpointer lands. Saved memory (core/recall/archival) is
  unaffected: that is already in the database.

---

## 10. Phase 2 — LangGraph (the turn cycle moved)

`AgentLoop` used to *be* the turn cycle. It is now a 93-line adapter: `step()`
seeds graph state, invokes, returns. The cycle lives in `graph/nodes.py`.

```
START -> build_prompt -> pressure_check --(>=threshold)-> inject_warning --+
             ^              |(under)                                       |
             |              v                                              |
             |           call_llm <----------------------------------------+
             |              |
             |    +- tool calls? --- no --> respond -> END
             |    v yes
             |  security_gate -> dispatch_tools -> sanitize_results
             |                                        |
             +---- heartbeat < cap and not done? ------+
```

Two deliberate deviations from the spec diagram:

- **The cycle re-enters at `build_prompt`, not `call_llm`.** A
  `core_memory_append` in round 1 has to reach the model in round 2, which means
  re-rendering the prompt. `pressure_check` guards on `heartbeat_count == 0`, so
  the warning still fires exactly once per turn.
- **`recursion_limit` is computed from `max_heartbeats`** (48 at the default).
  The counter in `dispatch_tools` enforces policy; the limit is the backstop for
  a routing bug the counter cannot see. LangGraph's default of 25 would trip at
  about 4 heartbeats.

### Turn state lives in a checkpointer

`InMemorySaver`, keyed by a per-instance thread id. `loop.messages`,
`last_input_tokens`, `last_limit` and `served_by` are **views** over the current
checkpoint; writing goes through `reset()` (rotates the thread, dropping
accumulated checkpoints) or `seed_context()`. Assigning to `loop.messages` would
mutate a throwaway copy, which is why the mutators are explicit.

This exists for `interrupt()`: suspending a turn and resuming it later only
works if the state it resumes into was persisted.

---

## 11. Phase 3 — external tools and the security layer

### Trust zones (spec 6.1)

`mcp_servers.yaml` is the authority. `trust_of(tool)` resolves through the
owning server, and tools are fetched **per server** so every tool's owner — and
therefore its trust zone — is known rather than guessed.

| Server | Transport | Trust | Notes |
|---|---|---|---|
| `memgpt-memory` | in-process | internal | Same `MemoryTools` the stdio server wraps. A subprocess hop would add latency and non-determinism for no isolation benefit. |
| `ddg-search` | `uvx duckduckgo-mcp-server` | **untrusted** | `search`, `fetch_content` |
| `filesystem` | `npx @modelcontextprotocol/server-filesystem ./workspace` | **untrusted** | 14 tools, 4 write-gated |

MCP schemas are flattened to scalar-only properties before reaching the router —
nested objects and `$ref` are the most reliable way to break tool calling on one
provider but not another. A tool-name collision across servers **raises**: an
ambiguous owner is an ambiguous trust decision.

### The two seams

**`sanitize_results` -> `security/sanitizer.py` (spec 6.2).** Every untrusted
result is, in this order: marker-escaped (on RAW input), pattern-neutralized,
length-capped, then wrapped in `<untrusted_content>` with a header restating the
rule. The system prompt's UNTRUSTED CONTENT section is what makes the markers
mean something.

The verbatim original goes to recall memory in `dispatch_tools`, *before*
sanitizing — the audit trail reads what actually arrived, the model only ever
sees the sanitized copy (the next `call_llm` runs strictly after this node).

**`security_gate` -> `security/guards.py` (spec 6.3).** Runs before anything
executes:

1. **Deny by default** on the node allowlist (declared schemas + external names
   + the built-in contract).
2. **Core memory closed** once `saw_untrusted` latches for the turn. It does not
   try to judge whether a given call "looks related" — by the time the model is
   choosing tools it has already read the content.
3. **Archival forced to `source='external'`**, so origin travels with the
   passage. A model-supplied `source='stated'` is overridden.
4. **Path jail** re-checked on this side of the subprocess boundary, across
   every path-carrying argument (`path`, `source`, `destination`, `paths`).
5. **`interrupt()`** for write-gated tools.

Order is load-bearing twice: marker-stripping runs before pattern flagging, and
the jail check runs before the approval gate — a traversal is refused outright,
never offered to a human who might wave it through.

The node is split around `interrupt()`: pure evaluation above it (that code
re-runs on resume), every side effect below it, so audit events are written once.

### Human-in-the-loop

`step()` returns normally with an empty reply while suspended;
`loop.pending_approval` carries the request and `loop.resume(approved)` finishes
the turn. Streamlit renders the exact arguments with Approve/Deny, and a
suspended turn owns the UI until answered. A denial returns text telling the
model not to retry or route around it.

Verified live against the real filesystem server: the file does not exist before
approval and does after.

---

## 12. Phase 4 — the production stack

### Storage: two backends, one surface

`assembly.build_stores()` is the only module that knows which backend is
running. Config follows the DSN — setting `MEMASSIST_POSTGRES_DSN` is enough.

| | SQLite + Chroma | Postgres + pgvector |
|---|---|---|
| Core + recall | `sqlite.py` | `postgres.py` |
| Archival | Chroma, cosine | `vector(384)`, `<=>`, HNSW |
| Budget ledger | one file | same DSN |
| Paging | score all, slice locally | `LIMIT/OFFSET` in SQL |
| Setup | none | a database |

Differences are dialect, never behaviour: `created_at` is a real `timestamptz`
in Postgres but is formatted back to `YYYY-MM-DD HH:MM:SS` on read, so both
backends hand callers identical rows. `tests/test_storage_backends.py` runs one
suite against both (each Postgres test in a throwaway schema), and the benchmark
scores 110/110 either way.

**The ledger matters more than it looks.** In a container the SQLite file is
ephemeral, so a Postgres deployment that left the ledger on disk would hand the
router a fresh, empty budget on every restart and cheerfully re-spend a free
tier it had already exhausted.

`migrate_to_postgres.py` re-embeds archival passages **from stored text** rather
than copying vectors: text is the source of truth, re-embedding is deterministic
with the same local model, and it stays correct even if the two stores ever
disagreed on dimensionality. Idempotent, and it never deletes from the source.

### API: sessions are checkpointer threads

One session id maps to one `AgentLoop`, and each loop owns one checkpointer
thread — so a turn suspended on an approval is still suspended when the next
HTTP request arrives. Memory is shared across sessions because it is the
*user's* memory; only the in-context window is per-session, which is exactly the
MemGPT split.

One turn at a time per session (409 otherwise): concurrent turns would
interleave writes into one thread. Sending a message while an approval is
outstanding is also a 409.

**What "SSE token streaming" means here.** The user-facing reply is the
*argument* of a `send_message` tool call, not the model's `content` field —
content is internal monologue the user must never see (§3). Streaming provider
tokens would therefore stream the wrong text, and streaming tool-call argument
deltas across four providers is exactly the fragility the flat-schema rule
exists to prevent. So what streams is the **turn**: every tool call, tool
result, eviction and security decision arrives live, and the reply is chunked
into `token` events. The live feed is sourced from `record_event` — the audit
stream was already being written, so mirroring it to subscribers needed no graph
changes at all.

### Deployment

`docker compose up --build` brings up web → api → memory-mcp → postgres, each
waiting on its dependency being **healthy** rather than merely started. The
Postgres probe passes `-d` as well as `-U`, because `pg_isready` reports ready
while `initdb` is still running.

Two image decisions worth naming:

- **bge-small is baked into a build LAYER**, not a volume. A volume would still
  pay the ~130 MB download once per fresh environment and would make container
  start depend on Hugging Face being reachable.
- **CPU-only torch is installed first**, from the PyTorch CPU index. The default
  wheel is 527 MB and drags ~2 GB of `nvidia_*` wheels behind it — none of which
  a container that embeds 384-d vectors on CPU can use.

### Tool-schema economy

Every tool schema rides in every prompt, on every turn, for the life of the
session. The registry's `tools:` allowlist filters at load, so the *registry*
decides what the context budget is spent on rather than whatever a server
happens to export: the filesystem server's 14 tools became 4, and 16 external
schemas became 6. Deny-by-default is unaffected — a dropped tool is simply
absent rather than soft-blocked.
