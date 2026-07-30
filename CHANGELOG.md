# Changelog

All notable changes to MemAssist. Distilled from the five phase reports in
[`BENCHMARKS.md`](BENCHMARKS.md); the score after each phase is the
deterministic benchmark, which is the project's regression gate.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-07-31

First release. Bench **115/115** on both storage backends, `pytest` 236 passed
(Postgres) / 216 passed + 20 skipped (SQLite), CI green across three jobs.

### Added

- **Durable turn state.** `assembly.build_checkpointer()` returns a
  `PostgresSaver` when the Postgres backend is selected, paired with the stores
  in the one module that knows which backend is running. A turn suspended on a
  human approval now survives the process that suspended it. Verified by
  killing the saver and the loop, rebuilding both, and finishing the turn.
- **T10 — the Mistral background consolidation lane** (`jobs/consolidate.py`),
  summarizing recall into archival through `chat_background()`, with a manual
  CLI (`--dry-run`, `--limit`) and a scheduled mode (`--every 6h`, also a
  `--profile jobs` compose service). New benchmark tier, **ceiling 110 → 115**.
- **`sensitive` flag on archival passages**, on both backends, defaulting to
  *detecting* rather than to `False` so a caller that forgets it fails closed.
  This was README defect #5 and the documented prerequisite for the background
  lane — pgvector carried the column but nothing ever wrote or read it, and
  Chroma had no equivalent.
- **`security/sensitivity.py`** — the deterministic, LLM-free privacy gate:
  credentials, tokens, private keys, card numbers (Luhn-checked), national
  identifiers, and content the user marked confidential.
- **Langfuse tracing** — one trace per turn, a span per graph node and per
  dispatched tool, provider tags from `served_by`, and named events for every
  security decision. Inert unless `LANGFUSE_PUBLIC_KEY` and
  `LANGFUSE_SECRET_KEY` are both set. Trace payloads are redacted through the
  same detector that gates the Mistral lane.
- **Unscored stress tier** (`make stress`): 100-turn session coherence, 50-fact
  retrieval precision, 20 messages under provider cooldowns, and a 10-page
  document recalled 20 turns later. Findings in `BENCHMARKS.md`.
- **GHCR image publishing on tag**, gated on both test jobs passing.
- `.claude/settings.json`, `CHANGELOG.md`, and a rewritten `README.md`.

### Fixed

- **Unbounded context growth when the model ignores the pressure warning.**
  Eviction fired only after the agent voluntarily called
  `archival_memory_insert`; the stress tier drove 100 turns to **219%** of the
  context limit with zero evictions and a 382-message queue. Paging is a safety
  property and cannot be contingent on the model's cooperation, so
  `Deps.hard_evict_fraction` (0.95) now forces the cut. The forced path leaves a
  different notice, because telling the model its context was summarized when
  nothing summarized it is a lie it then acts on. 219% → 95%.
  This is the second half of README defect #3, recorded as fixed in Phase 1.5.
- **`card-number` detection had never worked.** The Luhn gate stripped the
  digits from the match instead of the separators, so it length-checked the
  separators and rejected every card. It looked healthy because the only form
  under test also matched `aadhaar`; the unspaced form matched nothing and
  would have been sent to Mistral.
- **`pending_approval` was an instance attribute**, so a restarted process
  reported "nothing pending" over a graph still holding an interrupt. It is now
  a view over the checkpoint, like `messages` and `served_by` since Phase 2.
- **The pgvector test double zeroed out short queries.** Building the 384-d
  embedder by truncating the 512-d one to `[:384]` dropped every bucket past
  384, and `hashing_embedding("lease")` is non-zero only at index 463 — so the
  query was the zero vector, cosine distance was NaN, and the retrieval-order
  assertion was decided by physical row order. Passed alone, failed in a full
  run.
- **The benchmark's tier parser would have swallowed T10.** `cid[:2]`
  special-cased `T11`, so `T10a` filed itself under `T1`.
- Checkpointer connection pools are closed on API shutdown; an open pool keeps
  worker threads alive and turns a clean exit into a hang.

### Changed

- `AgentLoop` accepts `checkpointer` and `thread_id`; the API passes the session
  id, so a rebuilt loop addresses the same thread. Without it the persisted
  checkpoint would be durable but unaddressable.
- `AgentLoop.reset()` deletes the thread instead of rotating to a new id —
  `delete_thread` is on the base checkpointer interface, so it is one behaviour
  on both savers, and against Postgres it avoids stranding rows.
- CI waits for Postgres explicitly with the driver the tests use, rather than
  trusting the service healthcheck alone (defect #5, on probation).

---

## [0.4.0] — Phase 4: the production stack

Bench **110/110 on BOTH backends** — the phase's real claim.

### Added
- **Postgres + pgvector as a sibling backend** behind the same surface, chosen
  by config; setting `MEMASSIST_POSTGRES_DSN` is enough. Idempotent migration
  that re-embeds archival passages from stored text rather than copying vectors.
- **FastAPI service** with SSE turn streaming, session id == checkpointer
  thread id, and approve/deny endpoints that resume the graph.
- **Next.js + Tailwind UI**, including the approval flow.
- **Streamable HTTP transport** for the MCP memory server (`--http`).
- **docker-compose** stack with healthchecks, bge-small baked into an image
  *layer* rather than a volume, and CPU-only torch.
- CI: dual-backend Python matrix plus a node lint/typecheck/build job; the
  benchmark became a hard gate.

### Changed
- **Tool-schema economy**: a per-server `tools:` allowlist in the registry cut
  16 external schemas to 6 (filesystem 14 → 4). Every schema rides in every
  prompt for the life of a session.
- The provider budget ledger speaks both backends. An ephemeral SQLite file in
  a container would hand the router a fresh budget on every restart and
  cheerfully re-spend an exhausted free tier.

### Removed
- **Streamlit** (breaking), after `web/PARITY.md` was fully checked off.

---

## [0.3.0] — Phase 3: external tools and the security layer

Bench **110/110** (T1–T8 100, new T11 10). Ceiling 100 → 110.

### Added
- Real FastMCP memory server (`python -m memory_server`, 6 tools) and external
  MCP servers (`ddg-search`, `filesystem`) via `langchain-mcp-adapters`.
- **`security/sanitizer.py`** — untrusted results become marked-up *data*: 7
  injection patterns, marker-escape defusal on raw input, length capping, and
  a verbatim copy to recall for audit.
- **`security/guards.py`** — core memory closes for the turn once untrusted
  content is in it; archival writes forced to `source=external`;
  deny-by-default allowlist; path jail.
- Filesystem writes gated behind a LangGraph `interrupt()` with human approval.
- **T11 red-team corpus** (`security/injections/*.yaml`), read by both the
  benchmark tier and CI, so an attack case is written once.
- A checkpointer and an explicit state API (`reset`, `seed_context`) — the
  prerequisite for interrupts.

### Fixed
- The injection detector missed `[System note from the user]:` — the cheapest
  way to launder an instruction into something reading like the user's own
  words. Found by the corpus, which was written before the code was checked.

---

## [0.2.0] — Phase 2: the LangGraph refactor

Bench **100/100**, no tier moved — the point of the phase.

### Changed
- Control flow moved into `graph/`: eight nodes, a conditional pressure edge, a
  heartbeat cycle, and a recursion limit derived from `max_heartbeats`.
  `agent/loop.py` became a 93-line adapter with an unchanged `step()`, so the
  UI and the benchmark harness needed no changes.
- The cycle re-enters at `build_prompt`, not `call_llm`: a `core_memory_append`
  in round 1 has to reach the model in round 2.
- One canonical `GEMINI_API_KEY`; the documented `GOOGLE_API_KEY` is honoured
  with a deprecation warning.

### Fixed
- **CI had never once executed.** The committed workflow had a YAML parse error,
  so every run failed at startup and the badge was meaningless.

---

## [0.1.5] — Phase 1.5: the fix sprint

**58 → 100/100.** (The Phase 1 "79/100" harness was never in the repo, so a new
deterministic suite was written and Phase 1 re-measured on it. Compare 58 → 100.)

### Fixed
- **Gemini served zero requests for all of Phase 1.** Root cause was not the
  error handling: `providers.yaml` pinned `gemini-2.0-flash`, which has *zero*
  free-tier allowance and returns 429 `limit: 0` on every call. The router read
  that as a transient rate limit and cooled the priority-1 provider down in a
  loop, forever, silently. Now `gemini-2.5-flash-lite`, and
  `ProviderPermanentError` distinguishes "no allowance" and "wrong model" from
  backpressure — never cooled down, logged at ERROR, disabled for the process.
  *(T1 8→12)*
- **FIFO eviction after archival offload.** Phase 1 did the summarizing half of
  the MemGPT mechanic and never the paging half, so the queue only grew and the
  warning re-fired every turn. *(T4 10→18)*
- **bge-small-en-v1.5 (384-d) replaces the hashing embedder**, plus a one-time
  re-embed migration. The old one was lexical, so every paraphrase probe
  retrieved the wrong passage. *(T5 4→16)*
- Friendly provider-exhaustion copy instead of a raw traceback. *(T7 6→10)*
- Provenance tags (`stated` | `inferred`) on the human block and archival
  passages. *(T8 0→10)*
- `conversation_search_date` validates its bounds instead of silently returning
  "no matches", which gave the model no signal to correct its call. *(T3b, →100)*

---

## [0.1.0] — Phase 1: the MVP

### Added
- The three memory tiers (core / recall / archival) and the six memory tools,
  with flat schemas so all four provider tool-calling dialects accept them.
- The free-tier failover router: Gemini → Groq → OpenRouter → Mistral, with a
  persistent budget ledger and `served_by` on every reply.
- Context-pressure accounting and the memory-pressure warning.
- A Streamlit UI.

[1.0.0]: https://github.com/shrvan30/memassist/releases/tag/v1.0.0
