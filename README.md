# MemAssist

[![CI](https://github.com/shrvan30/memassist/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/memassist/actions/workflows/ci.yml)
[![bench 115/115](https://img.shields.io/badge/bench-115%2F115-brightgreen)](BENCHMARKS.md)
[![backends](https://img.shields.io/badge/backends-sqlite%20%7C%20postgres-blue)](ARCHITECTURE.md)
[![license MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

A personal assistant that remembers you between conversations. It keeps a small
amount of information in front of the model at all times, writes the rest to a
database, and searches that database when it needs something back — editing its
own memory through tool calls rather than relying on a context window. It runs
on free-tier language models from four providers and switches between them
automatically when one is rate-limited or out of quota, so it costs nothing to
operate.

The memory design follows [MemGPT](https://arxiv.org/abs/2310.08560)
(Packer et al., 2023).

---

## What it can do

**Remembers facts across restarts.** Three tiers: core memory (always in front
of the model), recall (every message, searchable), archival (long-term, searched
by meaning). Stop the app, start it again, and it still knows.
*`memory_server/storage/`*

**Edits its own memory, and records where each fact came from.** Every stored
fact is tagged `stated` (you said it), `inferred` (the model concluded it), or
`external` (it came from the web). *`memory_server/memory_tools.py`,
`security/guards.py`*

**Searches past conversation by keyword or date range.** Malformed dates return
an error the model can correct rather than an empty result.
*`memory_server/storage/sqlite.py`*

**Finds memories by meaning, not matching words.** "Which medication makes him
unwell?" retrieves a note about a penicillin allergy. Embeddings are computed
locally, so no text is sent anywhere to be indexed.
*`memory_server/storage/embedder.py`, Chroma or pgvector*

**Keeps working when the conversation outgrows the context window.** At 70%
capacity it summarizes older turns into archival memory and drops them from the
window; above 95% they are dropped whether or not it summarized first.
*`agent/token_budget.py`, `graph/nodes.py`*

**Switches providers mid-conversation.** Gemini → Groq → OpenRouter → Mistral,
with per-day budget tracking that survives restarts. Each reply shows which
provider answered it. *`llm/router.py`, `llm/budgets.py`*

**Uses external tools, and asks before writing anything.** Web search and a
filesystem restricted to one directory. Any write suspends the turn until you
approve it in the UI. *`mcp_client.py`, `graph/nodes.py`*

**Treats web content as data, not instructions.** Text fetched from the internet
cannot write to core memory, and instruction-shaped spans in it are neutralized
before the model reads them. Checked against a corpus of injection and
memory-poisoning attempts. *`security/`, `security/injections/`*

**Summarizes old conversation in the background, behind a privacy filter.**
Credentials, card numbers, identifiers and anything sourced from the web are
withheld from the summarization request, and what was withheld is reported by
category. *`jobs/consolidate.py`, `security/sensitivity.py`*

**Exposes its memory over MCP.** The six memory tools run as a server other MCP
clients can connect to, over stdio or HTTP. *`memory_server/__main__.py`*

**Runs as one command.** `docker compose up` starts four services: web, API,
memory server, Postgres. *`docker-compose.yml`*

**Scores 115/115 on a benchmark you can run yourself.** `python -m bench` is
offline and deterministic — every model call is scripted — so the number is
reproducible on your machine. A further 10 points (T12) grade whether the
assistant answers from memory instead of asking you to repeat yourself; those
need a provider key and are skipped without one. *`bench/`*

---

## Quickstart

```bash
git clone https://github.com/shrvan30/memassist.git
cd memassist
cp .env.example .env          # paste one free API key — Gemini or Groq is enough
docker compose up --build
```

Open <http://localhost:3000>. Tell it something about yourself, restart the
stack, and ask it again.

Free API keys, no credit card: [Gemini](https://aistudio.google.com/apikey) ·
[Groq](https://console.groq.com/keys) ·
[OpenRouter](https://openrouter.ai/keys) ·
[Mistral](https://console.mistral.ai/api-keys)

Ports used: 3000 (web), 8000 (API), 8090 (memory server), and 15432 for
Postgres — not 5432, which is often already taken by another database on a
developer machine. Override with `POSTGRES_HOST_PORT`.

Without Docker: `pip install -e ".[dev]"`, then `make api` and `make web` —
SQLite and Chroma, no database needed.

---

## Configuration

Every setting is an environment variable with a working default;
[`.env.example`](.env.example) lists them all. The two that matter most:

- `MEMASSIST_POSTGRES_DSN` — set it and both the storage layer and the saved
  turn state move to Postgres + pgvector. Unset, the app uses SQLite + Chroma
  and requires no setup.
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — set both to record traces.
  Unset, tracing does nothing at all.

## Architecture

A LangGraph state machine runs each turn: build the prompt, check context
pressure, call the model through the router, check the requested tool calls
against the security rules, run them, sanitize anything external, and repeat up
to five times before replying. Storage is either SQLite + Chroma or Postgres +
pgvector, chosen by one environment variable, behind a single interface.
Details in [ARCHITECTURE.md](ARCHITECTURE.md); the design it was built against
is [PROJECT_SPEC.md](PROJECT_SPEC.md).

## Benchmarks

115/115 on both storage backends, measured by an offline deterministic suite
that also runs in CI — 125/125 including the live tier, which needs a provider
key. Tiers, method and stress-test findings: [BENCHMARKS.md](BENCHMARKS.md).

## Deployment

Local `docker compose` is the supported deployment. As of July 2026 no free
application-hosting tier fits this app: Render's free instance is 512 MB and
0.1 CPU, Fly and Koyeb have closed their free tiers to new accounts, and
Hugging Face charges for Docker Spaces. The database does have a free option —
[Neon](https://neon.com/pricing) includes pgvector — so pointing the local
stack at Neon keeps memory when the machine is off.

Images are published to GHCR on each release tag:
`ghcr.io/shrvan30/memassist-api` and `ghcr.io/shrvan30/memassist-web`.

## Author

Built by Shravan Upadhye. Claude Code served as the implementation agent; I
designed the architecture, wrote the specifications, built the benchmark and
security corpus, and verified every phase against it.

## License

MIT — see [LICENSE](LICENSE).
