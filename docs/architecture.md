# Architecture

Four services, one turn cycle, three memory tiers, two interchangeable
storage backends.

## The running system

```
browser ── Next.js web (3000) ── FastAPI api (8000) ──► LangGraph turn cycle
                                                          │
                                          ┌───────────────┼────────────────┐
                                          ▼               ▼                ▼
                                    failover router   memory MCP      external MCP
                                    gemini→groq→      server (8090)   (ddg-search,
                                    openrouter→       6 memory tools  jailed filesystem,
                                    mistral                │          human-approved writes)
                                          │               ▼
                                    provider APIs    Postgres (15432 host / 5432 internal)
                                    ($0 free tiers)  or SQLite + Chroma
```

Compose starts them health-gated: postgres first (`pg_isready -d` — the
database, not just the server, closing the initdb race), then memory-mcp,
then api, then web (probed via explicit `127.0.0.1` — Next.js binds
IPv4-only and busybox wget prefers IPv6).

## One message's path

1. `POST /chat` opens an SSE stream; `AgentLoop.step()` (`agent/loop.py`,
   a 93-line adapter) seeds graph state.
2. The turn cycle runs (`graph/`): **build_prompt** (instructions +
   rendered core blocks + memory stats + today's date) -> **pressure_check**
   (70% threshold, first pass only) -> **call_llm** (through the router) ->
   on tool calls: **security_gate** -> possibly a **human interrupt** ->
   **dispatch_tools** -> **sanitize_results** -> back to build_prompt.
   Up to 5 heartbeats; `recursion_limit = 8x+8` backstops routing bugs.
3. Every heartbeat re-enters at build_prompt so a memory edit in round 1
   is visible in round 2.
4. `send_message` ends the turn -> **respond** persists the reply with
   `served_by` and chunks it to the browser. If the model never replied,
   respond composes an honest fallback from the turn's own findings — a
   turn can never return an empty reply.

## State vs dependencies

`AgentState` (serializable: messages, heartbeat count, findings, pending
approval) flows between nodes and is checkpointed — on Postgres, a turn
suspended on a human approval survives an API restart. `Deps` (stores,
router, MCP clients) is bound once and never serialized. That split is
what makes interrupts resumable.

## Module map

| Directory | Responsibility |
|---|---|
| `llm/` | Router, error taxonomy, budget ledger — [failover-router.md](failover-router.md) |
| `graph/` + `agent/` | Turn cycle, prompt assembly, context pressure |
| `memory_server/` | The three tiers behind an MCP server — [memory.md](memory.md) |
| `security/` | Sanitizer, guards, sensitivity filter — [security.md](security.md) |
| `jobs/` | Background consolidation via the Mistral lane |
| `bench/` | The regression gate — [benchmarks.md](benchmarks.md) |
| `api/` + `web/` | FastAPI + Next.js — [api.md](api.md) |
| `deploy/` | Images, gated SSH deploy — [deployment.md](deployment.md) |

## Notable decisions (full reasoning: [design-decisions.md](design-decisions.md))

- SSE streams turn **events** and chunks the reply — raw model tokens are
  internal monologue and are never streamed.
- Permanent provider errors **disable and fail over** rather than cool
  down (a retry timer hides config bugs) or raise (one bad provider must
  not end a turn three healthy ones could serve).
- The session id doubles as the graph thread id: memory is shared across
  sessions (it is the user's memory), the context window is per-session.
