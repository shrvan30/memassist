# MemAssist

[![CI](https://github.com/shrvan30/memassist/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/memassist/actions)
![Benchmark](https://img.shields.io/badge/benchmark-115%2F115_deterministic-blue)
![Release](https://img.shields.io/github/v/release/shrvan30/memassist)
![License](https://img.shields.io/badge/license-MIT-green)

A personal AI assistant that remembers you between conversations by
editing its own memory. It implements the MemGPT architecture (Packer et
al., 2023): the context window is treated as RAM, databases as disk, and
the model pages facts between them with tool calls. It runs at $0/month
on four free-tier LLM providers behind a failover router, and every
capability below is gated by a reproducible benchmark before it ships.

> **Demo:** ![demo](docs/assets/demo.gif) — restart the stack, ask
> "who's in my family?", and watch it answer from memory with a provider
> badge. *(2-minute walkthrough video: link pending)*

## What it can do

- **Remembers across restarts** in three tiers — core (always in
  context), recall (searchable log), archival (semantic search) — with
  provenance on every fact: `stated`, `inferred`, or `external`.
- **Survives a full context window** — summarizes old turns to archival
  at 70% usage; eviction is forced at 95% regardless of model
  cooperation.
- **Fails over across Gemini -> Groq -> OpenRouter -> Mistral
  mid-conversation**, with persistent daily budgets and a real error
  taxonomy (a 429 is not always a rate limit).
- **Never returns an empty reply** — a code-level liveness guarantee.
- **Uses external tools safely** — web search and a path-jailed
  filesystem via MCP; every write requires human approval through a
  suspended, resumable interrupt.
- **Blocks prompt injection and memory poisoning** — untrusted content
  is sanitized before the model reads it and can never write core
  memory; tested by an injection corpus written before the defenses.
- **Consolidates memory in the background** through a privacy gate that
  keeps sensitive content off the wire.
- **Runs as one command** — four health-gated services — and **scores
  115/115 on a deterministic benchmark, on both storage backends, in CI
  on every push** (plus a 10-point live tier).

## Quickstart (60 seconds)

```bash
git clone https://github.com/shrvan30/memassist && cd memassist
cp .env.example .env     # add at least one free provider key — see docs/configuration.md
docker compose up --build
# open http://localhost:3000 — tell it about yourself, restart, ask again.
```

Ports taken on your machine: web `3000`, api `8000`, memory server
`8090`, Postgres on **`15432`** (deliberately not 5432 — see
[docs/deployment.md](docs/deployment.md)). No Docker? The SQLite+Chroma
mode needs nothing installed — [docs/development.md](docs/development.md).

## Documentation

| Guide | What it answers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | The four services, the LangGraph turn cycle, one message's full path |
| [docs/memory.md](docs/memory.md) | The three tiers, provenance, paging and eviction, consolidation |
| [docs/failover-router.md](docs/failover-router.md) | The provider chain, the error taxonomy, budgets, lanes |
| [docs/security.md](docs/security.md) | The threat model: what the model reads, writes, and what leaves the machine |
| [docs/benchmarks.md](docs/benchmarks.md) | What 115/115 means, how to run it, the live tier, honest results |
| [docs/api.md](docs/api.md) | Endpoints, the SSE-events design, sessions and interrupts |
| [docs/configuration.md](docs/configuration.md) | Every environment variable, with the footguns called out |
| [docs/deployment.md](docs/deployment.md) | Compose, GHCR multi-arch images, the gated deploy pipeline, the free-tier reality |
| [docs/development.md](docs/development.md) | Local setup, the test doctrine, what CI runs, PR conventions |
| [docs/design-decisions.md](docs/design-decisions.md) | Why it is built this way — each decision with its trade-off |

## Who this is for

Me, daily — which is why the bugs got fixed. Anyone who wants a
reference implementation of MemGPT-style memory that runs end-to-end for
free. Anyone learning agent engineering — the benchmark, injection
corpus, and these docs are written to be studied, not just run.

## Roadmap (v1.2)

Robust live-tier grading · metadata-filtered retrieval + reranker ·
structured temporal facts · SIGKILL-safe bench cleanup · distributed
session lock for multi-replica API. Reasoning in
[docs/design-decisions.md](docs/design-decisions.md).

## Authorship

Built by Shravan Upadhye.
I designed the architecture, wrote the specifications, built the
benchmark and security corpus, and verified every phase against it.

MIT — see [LICENSE](LICENSE).
