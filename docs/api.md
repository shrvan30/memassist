# API

FastAPI service. One turn per request, streamed as events, with
suspendable human approvals.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Run one turn; opens the SSE stream |
| `POST /sessions/{sid}/approve` | Answer a pending interrupt (approve/deny) |
| `GET /pending` | What awaits approval |
| `POST /reset` | Clear the session's context window (memory survives) |
| `GET /sessions/{sid}` | Snapshot: core blocks, tier counts, context % |
| `GET /providers` | The provider panel: requests, cooling, disabled |
| `GET /healthz` | Status + active storage backend |

## The SSE design, defended

The stream carries turn **events** — tool calls, tool results, evictions,
security decisions — live, then the finished reply chunked as `token`
events. Raw model tokens are never streamed, for two reasons:

1. The user-facing reply is a `send_message` tool **argument**; raw
   content tokens are internal monologue (memory reasoning, security
   deliberation) that must never render.
2. Tool-call argument deltas stream differently across four providers —
   exactly the cross-provider fragility the flat-schema rule avoids.

The memory machinery becomes the visible show, which is the product.

## Sessions, locks, interrupts

- **Per-session non-blocking lock**: a second `/chat` during a running
  turn returns 409 immediately — told, not queued. A mismatched approval
  (stale or already resolved) also 409s, so two tabs cannot approve the
  wrong write.
- **Suspension across requests**: an approval suspends the turn via the
  graph checkpointer under the session id (which doubles as the graph
  thread id) and resumes on the later `/approve` — on Postgres this
  survives an API restart, verified with a SIGKILL mid-interrupt.
- **Memory is shared across sessions; the window is per-session.** It is
  the user's memory, not the session's.

## Startup

The embedding model warms at boot in a failure-isolated step (first
archival search: 16.4s -> 0.16s). One-time costs belong at edges you
control, never inside a user-visible request.
