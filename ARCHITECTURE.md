# Architecture

How MemAssist is built. The design it was written against is
[PROJECT_SPEC.md](PROJECT_SPEC.md); what it scores is [BENCHMARKS.md](BENCHMARKS.md).

## File map

```
config.py           Settings, all env-overridable
assembly.py         Composition root — the only module that knows which
                    storage backend and checkpointer are in use

agent/              Adapter over the graph; no storage, no provider SDKs.
                    loop.py · prompts.py · token_budget.py

graph/              Control flow only — never a provider call.
                    state.py (AgentState + Deps) · nodes.py · graph.py

llm/                The only place a provider is called.
                    router.py · errors.py · budgets.py · providers.yaml

memory_server/      Memory tiers and the six tools.
                    memory_tools.py · schemas.py · __main__.py (MCP server)
  storage/          sqlite.py · chroma.py · postgres.py · pgvector_store.py ·
                    embedder.py (local bge-small, 384-d) · migrations

security/           sanitizer.py · guards.py · sensitivity.py · injections/
mcp_client.py       External MCP servers, trust zones
jobs/consolidate.py Background recall → archival summarization
observability.py    Langfuse tracing; inert unless keys are set
api/                FastAPI: SSE turns, approvals    web/  Next.js UI
bench/              115-point suite + unscored stress tier
```

## One turn

`AgentLoop.step(text)` returns the messages sent to the user.

```
user text
  │  recorded to recall; appended to the in-context queue
  ▼
┌ heartbeat cycle, max 5 ─────────────────────────────────────────────┐
│ build_prompt       system prompt = core memory + stats + usage      │
│ pressure_check     first pass and ≥70%? → inject_warning            │
│ call_llm           router.chat(...); sets served_by, input_tokens   │
│ security_gate      allowlist, core-memory lockout, path jail,       │
│                    interrupt() for gated tools                      │
│ dispatch_tools     send_message → user; memory tools; external      │
│                    tools (result written to recall verbatim)        │
│ sanitize_results   external results rewritten as marked-up data     │
│ stop when a reply was sent and no heartbeat was requested           │
└─────────────────────────────────────────────────────────────────────┘
  ▼
every event is written to the recall log, tagged with served_by
```

The cycle re-enters at `build_prompt`, so a memory write in one round is visible
in the next; `pressure_check` fires only on the first pass, so the warning
appears once per turn. `send_message` is the only path to the user. Tool results
are always strings — validation failures become `"Error: …"` the model can read
and retry, never an exception.

## Memory tiers

| Tier | Storage | In context | Tools |
|---|---|---|---|
| Core | `core_blocks` (persona, human) | Always | `core_memory_append`, `core_memory_replace` |
| Recall | `messages`, append-only | On demand | `conversation_search`, `conversation_search_date` |
| Archival | Chroma or pgvector, cosine, 384-d | On demand | `archival_memory_insert`, `archival_memory_search` |

Tool schemas are flat — no nested objects — so all four provider dialects accept
them, and each tool's Pydantic model and JSON schema sit in one file so they
cannot drift. Facts carry a `source` (`stated`, `inferred`, `external`);
archival passages also carry `sensitive`, set by content inspection at insert.

**Context pressure.** Crossing 70% injects a warning to summarize into archival.
A successful archival write then evicts the oldest half of the queue, leaving a
marker. Above `hard_evict_fraction` (95%) eviction happens regardless, with a
different marker saying the messages were not summarized. Eviction skips past
orphaned tool replies, which providers reject.

## Failover router

`Router.chat()` walks the priority chain in `providers.yaml`:

```
skip if: no key · disabled · cooling down · over daily budget
429 transient      → 60s cooldown, try next
402 quota          → cooldown until UTC midnight, try next
5xx / transport    → retry once, then try next
permanent          → disable for this process, log at ERROR, try next
nothing served     → AllProvidersExhausted → a plain sentence to the user
```

Budgets persist in `provider_usage`, keyed by UTC date; `served_by` is recorded
on every event; `chat_background()` reaches only Mistral, the sole provider on
the `background` lane.

## Security

| Server | Trust | Notes |
|---|---|---|
| `memgpt-memory` | internal | Dispatched in-process; the MCP server exists for external clients |
| `ddg-search` | untrusted | `search`, `fetch_content` |
| `filesystem` | untrusted | Restricted to `./workspace`, writes require approval |

**`sanitize_results` → `sanitizer.py`.** Untrusted results are marker-escaped on
the raw input, pattern-neutralized, length-capped, then wrapped in
`<untrusted_content>`. The unmodified original goes to the recall log first, so
an audit reads what arrived while the model only sees the sanitized copy.

**`security_gate` → `guards.py`.** Deny by default against the node's allowlist;
core memory closes for the rest of a turn once untrusted content enters it;
archival writes are forced to `source=external`; every path-carrying argument is
re-checked against the jail. The jail check runs before the approval prompt, so
a path escape is refused rather than offered to a human.

`security/injections/*.yaml` holds the attack corpus, read by both the benchmark
and the test suite so a case is written once.

## The two seams

`agent/loop.py` depends on `Protocol` types, not implementations:

| Protocol | Methods | Implementation |
|---|---|---|
| `LLMRouter` | `chat`, `context_window`, `min_context_window` | `llm.router.Router` |
| `MemoryInterface` | `render_core_memory`, `memory_stats`, `dispatch`, `record_event` | `memory_server.memory_tools.MemoryTools` |

`ExternalToolset` covers the MCP surface the graph uses. `assembly.py` is the
only module that picks concrete implementations, which is why swapping SQLite
for Postgres touches one file.

## Storage backends

SQLite + Chroma, or Postgres + pgvector, selected by `MEMASSIST_POSTGRES_DSN`.
Differences are dialect, not behaviour: timestamps are formatted identically on
read, and one test suite runs against both. The graph checkpointer follows the
same choice, so on Postgres a turn suspended awaiting approval survives a
restart — the session id is the thread id, so a rebuilt process reaches the same
state. The budget ledger lives with the rest of the data; on SQLite in a
container it would reset each restart and re-spend an exhausted free tier.

## Notable decisions

**A 429 is not always a rate limit.** Google returns 429 both for "too fast" and
for "this model has no free-tier allowance" (`limit: 0`). Treating the second as
transient put the primary provider on a rolling cooldown forever, where it
looked merely busy and served nothing. `ProviderPermanentError` is never cooled
down — the provider is disabled for the process and logged at ERROR — but it
fails over rather than raising, so one misconfigured provider cannot end a turn
the other three could serve.

**SSE streams turn events, not model tokens.** The user-visible reply is the
argument of a `send_message` tool call, not the model's `content` field, which
holds internal reasoning the user must not see. Streaming provider tokens would
stream the wrong text, and streaming tool-call argument deltas differs across
the four providers. So the stream carries tool calls, results, evictions and
security decisions as they happen, and chunks the finished reply. The feed is
built from `record_event`, which was already being written for the audit log.

**The web healthcheck probes `127.0.0.1`, not `localhost`.** Next.js listens on
`0.0.0.0` (IPv4 only) while the Alpine image maps `localhost` to both `127.0.0.1`
and `::1`. busybox wget tries `::1` first and reports a healthy server as
refused. The API probe is unaffected because that image uses curl, which prefers
IPv4.

**Postgres publishes on host port 15432**, because 5432 is commonly already
held and a first run should not fail on a port conflict. Services inside the
network still use `postgres:5432`.

**Embeddings are computed locally**, so rotating providers cannot fragment the
vector space.

## Verification

`make test` (no keys, no network), `make bench` (115 points, deterministic),
`make stress` (unscored). CI runs lint, both suites against both backends, a
secret scan, a dependency audit and the web build.
