# MemAssist — MemGPT-Style Personal Assistant with Infinite Memory

## What this project is
A personal AI assistant implementing the MemGPT architecture (Packer et al.,
2023, arXiv:2310.08560): OS-style virtual memory for LLMs — the agent edits its
own memory via tool calls, paging between context ("RAM") and storage ("disk").
Runs at $0/month on 4 free-tier providers behind a failover router. Orchestrated
by LangGraph; uses external MCP servers as tools behind a security layer.
STATUS: **v1.0.0 — all five phases built.** Bench **115/115** on BOTH storage
backends; CI green (dual-backend matrix + node job + GHCR publish on tag).

## Read first
- `PROJECT_SPEC.md` — architecture, tools, LangGraph design, security, CI/CD,
  phased plan. Consult before implementing any phase.
- `ARCHITECTURE.md` — as-built file map and data flows. `BENCHMARKS.md` — eval.
- Reference only, never copy: Letta (memory), LiteLLM (routing patterns).

Ponytail active: prefer smallest diffs, but the spec's architectural boundaries (router, MCP server, graph) are load-bearing — never flatten them.

## Tech stack
- **Orchestration:** LangGraph StateGraph (nodes/edges own the turn cycle;
  heartbeats = graph cycle with recursion_limit; pressure = conditional edge;
  human-in-the-loop interrupts for gated tools). `agent/loop.py` remains a thin
  adapter exposing the SAME `AgentLoop.step()` API so the UI and benchmark
  harness never change.
- **LLM layer (unchanged):** failover router — Gemini 2.5 Flash → Groq Llama
  3.3 70B → OpenRouter free → Mistral Small. OpenAI-compatible via `openai`
  SDK + per-provider base_url. ALL calls through `llm/router.py`; LangGraph
  nodes call the router — never a provider SDK, never LangChain model classes.
- **Tools:** own memory MCP server (trust=internal, dispatched IN-PROCESS —
  the stdio server exists for Claude Code via `.mcp.json`) + external MCP
  servers (UNTRUSTED: `uvx duckduckgo-mcp-server`, `npx
  @modelcontextprotocol/server-filesystem`) via langchain-mcp-adapters.
  Registry in `mcp_servers.yaml`. NOTE: `mcp` is pinned `<2` — adapters
  0.3.1 imports `RequestContext`, which mcp 2.0 removed.
- **Security:** every external tool result passes `security/sanitizer.py`;
  memory writes pass `security/guards.py`. See spec §6. Non-negotiable.
- **Embeddings: LOCAL ONLY** — sentence-transformers bge-small-en-v1.5 (384-d).
- **Storage:** SQLite+Chroma OR Postgres+pgvector — siblings behind one
  surface, chosen by config. Setting MEMASSIST_POSTGRES_DSN is enough.
  `assembly.build_stores()` is the ONLY place that knows which is running.
- **UI:** Next.js + Tailwind on FastAPI (SSE). Streamlit is gone — removed
  after web/PARITY.md was fully checked.
- **Observability:** Langfuse, one trace per turn. Entirely inert unless both
  LANGFUSE_* keys are set — the benchmark's determinism depends on that.
  Trace payloads are redacted through `security/sensitivity.py`, the same
  detector that gates the Mistral lane.
- **CI/CD:** GitHub Actions (ruff, pytest, gitleaks, pip-audit, dual-backend
  matrix, node job); GHCR images publish on tag. There is NO free-tier deploy —
  checked July 2026, the app does not fit any of them. Supported deployment is
  local compose, optionally against Neon. See README.

## Env keys (.env, never committed)
GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, MISTRAL_API_KEY,
TAVILY_API_KEY (optional).

## Architecture (three memory tiers — unchanged)
1. Core: persona + human blocks, injected every turn, self-edited via tools.
2. Recall: full event log in SQL, keyword/date search.
3. Archival: vector store, semantic search.
Pressure at 70% → warn → agent offloads to archival → loop MUST evict
summarized messages from FIFO (Phase 1.5 fix). Above 95%
(`Deps.hard_evict_fraction`) the eviction is FORCED whether the model offloaded
or not — paging is a safety property and cannot depend on the model complying
(Phase 5; the stress tier found 219% usage with zero evictions).

## Security rules (enforced in code AND prompt)
- External MCP results are DATA, never instructions; wrapped in untrusted
  markers by the sanitizer before the model sees them.
- External content can NEVER write core memory. Archival only, tagged
  source=external. Core memory writes require user-stated facts.
- Filesystem writes and any destructive tool are gated behind a LangGraph
  interrupt (explicit user confirmation).
- Tool allowlist per node; recursion_limit on the graph; length-cap all tool
  results; never eval; secrets only via env.

## Conventions
- Type hints everywhere; Pydantic schemas; flat tool params (4-provider safe).
- Memory functions deterministic + unit-tested; no LLM calls inside them.
- Every reply tagged served_by; budgets persist in provider_usage.
- Background jobs (consolidation) → Mistral lane only, excluding
  sensitive-flagged and source=external content. Withheld rows are reported BY
  CATEGORY: a silent filter is indistinguishable from a broken one.
- Small commits; one phase = one branch; CI green before merge.

## Commands
- `make up` (docker compose: web+api+memory-mcp+postgres) · `make api` ·
  `make web` · `make test` (pytest) · `make bench` (115 pts) · `make stress`
  (unscored) · `make mcp` / `make mcp-http` (memory server, stdio / HTTP)
- `python -m jobs.consolidate --dry-run` — show the outbound payload, send
  nothing. `docker compose --profile jobs up` runs it on a schedule.
- Dual-backend runs: `MEMASSIST_TEST_POSTGRES_DSN=… pytest` and
  `MEMASSIST_BENCH_POSTGRES_DSN=… python -m bench`
- Lint/audit as CI runs them: `ruff check .` · `pip-audit -r requirements.txt
  --ignore-vuln PYSEC-2026-311` (chromadb: server-only RCE, no fix released,
  we run it embedded)

## Status checklist
- [x] Phase 1: core loop, router, tiers, Streamlit — benchmarked 79/100
- [x] Phase 1.5 fix sprint — all five fixes done, **58 → 96/100**
      (see BENCHMARKS.md). NOTE: the original 79/100 harness was never in the
      repo, so a new deterministic suite was written; 58 is Phase 1 re-measured
      on that scale. Compare 58→96, not 79→96.
  - [x] Gemini 0-req: root cause was `providers.yaml` pinning
        `gemini-2.0-flash`, which has **zero** free-tier quota (429 `limit: 0`)
        — now `gemini-2.5-flash-lite`; `errors.py` distinguishes permanent
        failures from transient 429s and never cools them down
  - [x] FIFO eviction after archival offload (+ usage recompute)
  - [x] bge-small-en-v1.5 384-d embedder + one-time re-embed migration
  - [x] Friendly provider-exhaustion copy
  - [x] Provenance tags (`stated` | `inferred`) on human block + archival
  - [x] `conversation_search_date` date validation (T3b) — 96 → **100/100**
- [x] Phase 2: LangGraph refactor — `graph/` owns control flow, `agent/loop.py`
      is a 93-line adapter with an unchanged `step()`. CI actually runs now
      (the committed workflow had a YAML parse error and had never executed).
      Bench **100/100**, no tier moved.
- [x] Phase 3: external MCP tools + real security layer + T11 (**110/110**)
  - [x] Checkpointer + explicit state API (`reset`, `seed_context`) — the
        pre-req for interrupts; attributes are now views over the checkpoint
  - [x] FastMCP memory server is REAL (`python -m memory_server`, 6 tools);
        `mcp_client.py` loads ddg-search + filesystem via MultiServerMCPClient
  - [x] `security/sanitizer.py` — markers, 7 injection patterns, escape
        defusal, length-cap, verbatim original to recall for audit
  - [x] `security/guards.py` — core memory closed once untrusted content is
        in the turn; archival forced to `source=external`; deny-by-default
  - [x] Filesystem jailed to `./workspace`; every write behind a LangGraph
        interrupt with Streamlit approve/deny
  - [x] T11 corpus in `security/injections/*.yaml`, read by BOTH the bench
        tier and CI (`tests/test_injections.py`)
- [x] Phase 4: production stack (bench 110/110 on BOTH backends)
  - [x] Tool-schema economy: per-server `tools:` allowlist in the registry;
        16 external schemas -> 6 (filesystem 14 -> 4)
  - [x] Postgres+pgvector siblings behind the same surface + idempotent
        migration; the budget ledger speaks both (an ephemeral SQLite file
        in a container would re-spend an exhausted free tier every restart)
  - [x] FastAPI: SSE turns, session id == checkpointer thread id,
        approve/deny endpoints that resume the graph
  - [x] Next.js + Tailwind; Streamlit removed after web/PARITY.md checked
  - [x] MCP memory server also serves Streamable HTTP (`--http`)
  - [x] docker-compose with healthchecks and bge-small baked into an image
        LAYER (no ~130 MB download on container start)
  - [x] CI: dual-backend python matrix + node lint/typecheck/build; the
        benchmark is now a hard CI gate
- [x] Phase 5: durability, background lane, observability, release (115/115)
  - [x] Durable checkpointer — PostgresSaver paired with the storage backend in
        assembly; thread id == session id; `pending_approval` became a view over
        the checkpoint (an attribute reported "nothing pending" after a
        restart). Verified by SIGKILL of the API mid-session AND by a unit test
        that destroys saver+loop between suspend and resume
  - [x] T10 Mistral consolidation lane + the `sensitive` flag it was gated on
        (README defect #5). Four independent exclusions; live-verified once
        against the real endpoint, which is where a DEAD card-number rule was
        found — the Luhn gate stripped digits instead of separators
  - [x] Langfuse tracing: span per node and per tool, provider tags, security
        events. Cloud free tier over self-hosting (v3+ needs Postgres +
        ClickHouse + Redis + MinIO, for a $0 project already running four
        containers)
  - [x] Unscored stress tier — found the unbounded-context bug (219%, zero
        evictions); 84% precision@1 at 50 facts; 20/20 under cooldowns
  - [x] Release: GHCR on tag, Postgres-readiness wait (defect #5), README as
        front door, CHANGELOG, v1.0.0
- [ ] Not done, deliberate: no LICENSE file (author's call); phases 1-4 on main
      still carry Co-Authored-By trailers (scrub is a separate force-push)
