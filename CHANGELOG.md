# Changelog

All notable changes to MemAssist. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — unreleased

### Changed
- The assistant now answers from what it retrieves instead of asking for it
  again. A search that returns relevant content is treated as an answer:
  it states the substance and says where it came from, and matches by meaning
  rather than by keyword, so a question about a "3 month goal" is answered by a
  plan the user described over three months without ever using that word. It
  asks the user only for things it could not retrieve.
- Goals, deadlines, projects and commitments are now saved as durable facts
  alongside identity and preferences — the short form to core memory, the
  detail to archival.
- Replies drawn from memory are attributed ("you told me on 12 June"), so a
  remembered fact is distinguishable from an assumed one.
- The system prompt carries today's date, which lets the assistant say whether
  a stored deadline has passed and reason from a date the user supplies
  ("it's mid-September — what did I miss?").

### Added
- Benchmark tier T12, memory utilization (10 points, ceiling 125). Unlike
  T1–T11 it needs a live provider, because it grades a decision the model
  makes; it is skipped when no key is present and the ceiling falls back to
  115, so CI stays offline and reproducible.

### Fixed
- A turn could end in silence. The assistant could spend every tool round
  searching and never send a message, leaving the user with an empty reply
  rather than a partial answer.

## [1.0.1] — 2026-08-01

### Changed
- Postgres is published on host port **15432** by default instead of 5432,
  which is commonly already in use by another database. Set
  `POSTGRES_HOST_PORT` to override. Services inside the compose network are
  unaffected and still connect to `postgres:5432`.
- `POSTGRES_PORT` renamed to `POSTGRES_HOST_PORT`, which says which side of the
  mapping it sets.
- Documentation rewritten: the README is organised around what the software
  does, and the architecture, specification and benchmark documents drop
  development history in favour of current behaviour.

### Added
- Healthcheck for the web service, which previously had none — `docker compose
  ps` could report the stack ready while the UI was still starting.

### Fixed
- The web healthcheck probes `127.0.0.1` rather than `localhost`. Next.js
  listens on IPv4 only while the container resolves `localhost` to IPv6 first,
  so a healthy service was reported as refused.

## [1.0.0] — 2026-07-31

First release. 115/115 on the benchmark on both storage backends.

### Added
- **Three-tier memory.** Core memory held in front of the model, a full recall
  log searchable by keyword and date range, and an archival store searched by
  meaning. Six tools let the agent read and write all three.
- **Provider failover.** Gemini, Groq, OpenRouter and Mistral tried in order,
  with per-day request and token budgets that persist across restarts,
  cooldowns for rate limits, and the serving provider recorded on every reply.
- **Turn cycle as a state machine.** Prompt construction, context-pressure
  check, model call, security gate, tool dispatch and sanitization, with a cap
  of five tool rounds per turn.
- **Context paging.** At 70% of the window the agent is asked to summarize old
  turns into archival memory and they are dropped from the window; above 95%
  they are dropped whether or not it summarized first.
- **Semantic retrieval** using a local bge-small model, so no text is sent
  elsewhere to be indexed.
- **Provenance on every stored fact:** `stated`, `inferred` or `external`.
- **MCP memory server** exposing the six tools to other clients over stdio or
  HTTP, plus external MCP tools: web search and a filesystem restricted to one
  directory, with writes suspended for human approval.
- **Security layer.** External content is wrapped and neutralized before the
  model sees it, cannot write to core memory, and is forced to
  `source=external` in archival. Path arguments are checked against the jail
  before any approval prompt. A corpus of injection and memory-poisoning cases
  is shared by the benchmark and the test suite.
- **Postgres + pgvector** as an alternative to SQLite + Chroma, selected by one
  environment variable, with an idempotent migration between them. Saved turn
  state moves with it, so a turn awaiting approval survives a restart.
- **HTTP API and web UI.** FastAPI streaming turn events over SSE, approve and
  deny endpoints, and a Next.js interface showing memory contents and the
  serving provider.
- **Background consolidation** summarizing old recall into archival through a
  single provider, behind a filter that withholds credentials, identifiers and
  anything sourced from the web, and reports what it withheld by category.
- **Tracing** of each turn to Langfuse, including per-node and per-tool spans
  and security decisions. Does nothing unless both keys are set.
- **Benchmark suite.** 115 points across ten tiers, offline and deterministic,
  run in CI against both storage backends, plus unscored load scenarios.
- **Container stack.** `docker compose up` starts web, API, memory server and
  Postgres, each waiting for its dependencies to report healthy. Images publish
  to GHCR on release tags.

### Fixed
- Context could grow without limit when the model ignored the pressure warning.
  Bounding the window no longer depends on the model cooperating.
- Card numbers were never detected by the privacy filter: the checksum step
  discarded the digits instead of the separators.
- A suspended approval was reported as absent after a restart, because the
  pending state was held in memory rather than read from saved turn state.
- Malformed dates in recall search returned no results instead of an error the
  model could correct.
- A zero-quota response from a provider was treated as a temporary rate limit,
  which put the first-choice provider on a permanent cooldown where it appeared
  busy but served nothing.

[1.0.1]: https://github.com/shrvan30/memassist/releases/tag/v1.0.1
[1.0.0]: https://github.com/shrvan30/memassist/releases/tag/v1.0.0
