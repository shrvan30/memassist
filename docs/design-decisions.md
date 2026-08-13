# Design decisions

Each decision with the alternative it rejected and the trade it accepted.
The recurring principle: **an instruction is not a guarantee** — any
property whose violation breaks the system is enforced in code; the
prompt only makes outcomes better.

## Implement MemGPT from scratch, not Letta
The memory engine was the point; adopting the reference framework would
have outsourced the well-understood core while every distinctive piece
(router, budgets, security, benchmark) sat outside its scope anyway.
Letta was read, never copied. Trade: more code owned. For a product
deadline the answer flips — build what is the thesis, buy what is
commodity to it.

## Hand-rolled loop first, LangGraph in phase two
Writing the turn cycle by hand is how the mechanics were understood;
frameworks adopted before understanding become load-bearing magic. The
migration happened when two features demanded a graph runtime — resumable
interrupts (checkpointing) and explicit conditional flow — and was scoped
hard: LangGraph got control flow only, the router stayed the sole LLM
door, `step()`'s API stayed byte-stable so the benchmark ran unmodified
through the swap. 290 lines became a 93-line adapter at an unchanged
score.

## Own failover router, not LiteLLM
The router is where the distinctive engineering lives (the zero-quota
taxonomy, persistent UTC budgets, lanes), and the SDK-retry conflict
proved two retry authorities in one stack fight — a wrapped router inside
a retrying framework reproduces that bug one layer up. At a company with
a platform team, a gateway is the right call; here the $0 constraint IS
the problem, and the router is the response to it.

## Deterministic benchmark, not LLM-as-judge
A judge's verdicts drift with its version and prompt, so a score change
stops naming the diff as its cause — and a merge criterion that can move
on its own is not a criterion. Where judgment is genuinely required
(T12), grading is still a pure function of the reply text; nondeterminism
is confined to generation. Accepted cost: substring brittleness, stated
in the docs and queued for v1.2 — a known, inspectable weakness over an
unknowable drifting one.

## Dual storage backends
SQLite+Chroma is the zero-setup quickstart; Postgres+pgvector is the
production path where memory, budgets, and checkpoints must share one
durable store. The drift risk is paid down structurally: one store
Protocol, one selection point (`assembly.build_stores`), one parametrized
suite and the full bench on both backends in CI. Support N backends only
if you gate all N continuously.

## SSE streams events, not model tokens
The reply is a `send_message` argument; raw tokens are internal monologue
and provider-inconsistent to stream. The turn's events — tools,
evictions, security decisions — are the honest live feed.

## Permanent provider errors disable-and-fail-over
Not cooldown (a retry timer hides config bugs — the primary once served
zero requests for weeks behind one), not raise (one bad provider must not
end a turn three healthy ones could serve). Disable loudly, keep serving.

## Regex privacy gate, not an LLM classifier
Asking a model whether something is sensitive means sending it the thing
first — the exact disclosure being prevented. Deterministic, offline,
testable; tuned for recall because the failure costs are asymmetric.

## Postgres on host port 15432
5432 is commonly owned by a native install or another stack with
`restart: always`; a quickstart that dies on a port collision before
anything runs is a broken promise. Containers use `postgres:5432`
internally and never notice.

## Safety-by-instruction was the original mistake
Four separate failures — the silent turn, the 219% window, the
provenance-tag leak, inferred-as-stated — all traced to trusting the
model to follow rules. Each fix moved the invariant into code (the
respond fallback, forced eviction, storage-layer stripping, guard-forced
provenance) and left the prompt as optimization. That correction, applied
until it became the first principle, is the project's real lesson.
