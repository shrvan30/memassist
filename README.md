# MemAssist

**A personal AI assistant that remembers you — running on $0/month.**

[![CI](https://github.com/shrvan30/memassist/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/memassist/actions/workflows/ci.yml)
[![bench 115/115](https://img.shields.io/badge/bench-115%2F115-brightgreen)](BENCHMARKS.md)
[![backends](https://img.shields.io/badge/backends-sqlite%20%7C%20postgres-blue)](ARCHITECTURE.md)
[![cost](https://img.shields.io/badge/cost-%240%2Fmonth-success)](#deployment)

An implementation of the **MemGPT architecture** ([Packer et al., 2023](https://arxiv.org/abs/2310.08560)):
OS-style virtual memory for language models. The agent edits its own memory
through tool calls, paging facts between its context window ("RAM") and
persistent storage ("disk") — so it remembers you across sessions without ever
growing a context window it cannot afford.

```
You:  "I'm Shravan, I work on ML infra and I hate long emails."
      → agent calls core_memory_append(block="human", …)
      → restart the app, tell it nothing, ask "what do you know about me?"
      → it still knows. That's the whole thesis.
```

It runs on four free-tier providers behind a failover router, orchestrated by
LangGraph, with a security layer that treats every external tool result as
hostile until proven otherwise.

> **Demo**
>
> <!-- TODO: replace with docs/demo.gif — a ~30s capture showing a fact stated,
>      the memory inspector updating, a NEW session recalling it, and an
>      approval prompt suspending a filesystem write. -->
>
> ![demo placeholder](https://img.shields.io/badge/demo-GIF%20coming-lightgrey)

---

## 60-second quickstart

```bash
git clone https://github.com/shrvan30/memassist.git
cd memassist
cp .env.example .env          # paste ONE free API key — Gemini or Groq is enough
docker compose up --build     # web + api + memory-mcp + postgres
```

Open <http://localhost:3000>. Tell it something about yourself, restart the
stack, and ask it again.

Free keys, no credit card:
[Gemini](https://aistudio.google.com/apikey) ·
[Groq](https://console.groq.com/keys) ·
[OpenRouter](https://openrouter.ai/keys) ·
[Mistral](https://console.mistral.ai/api-keys)

Every service waits on its dependency being **healthy**, not merely started, so
a fresh `up` cannot race Postgres. The first build takes a few minutes: the
image bakes the embedding model into a layer, so container *start* never waits
on a download.

<details>
<summary>Without Docker</summary>

```bash
pip install -e ".[dev]"
make api      # FastAPI on :8000  (SQLite + Chroma, zero setup)
make web      # Next.js on :3000
```

No `MEMASSIST_POSTGRES_DSN` means SQLite + Chroma, which needs no database at
all. `python` not on PATH? Every make target takes `PY=/path/to/python`.
</details>

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
  Next.js  ──SSE──► │  FastAPI       session id == thread id  │
  :3000             └────────────────────┬────────────────────┘
                                         ▼
        ┌──── LangGraph: the turn cycle (graph/) ───────────────────────┐
        │                                                               │
        │  build_prompt → pressure_check ──(≥70%)──► inject_warning ─┐   │
        │       ▲             │ (under)                              │   │
        │       │             ▼                                      ▼   │
        │       │          call_llm ◄────────────────────────────────────│
        │       │             │                                          │
        │       │   ┌─ tool calls? ── no ──► respond ──► END             │
        │       │   ▼ yes                                                │
        │       │  security_gate ──► dispatch_tools ──► sanitize_results │
        │       │  deny by default,    memory │ external      mark       │
        │       │  core-memory lockout,                    untrusted     │
        │       │  path jail, interrupt                     as DATA      │
        │       └───────── heartbeat < 5 and not done? ──────────────────┘
        └───────────────────────────────┬───────────────────────────────┘
                                        ▼
   ┌────────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
   │ llm/router.py      │   │  Memory tiers        │   │ External MCP    │
   │ the ONLY place a   │   │                      │   │   UNTRUSTED     │
   │ provider is called │   │ Core    always in    │   │  ddg-search     │
   │                    │   │         context      │   │  filesystem     │
   │ Gemini  ─┐         │   │ Recall  full log,    │   │  (jailed, and   │
   │ Groq    ─┤ failover│   │         SQL search   │   │   human-gated)  │
   │ OpenRtr ─┤ + budget│   │ Archival vectors,    │   └─────────────────┘
   │ Mistral ─┘ ledger  │   │         semantic     │
   └─────────┬──────────┘   └──────────┬───────────┘
             │ background lane         │
             │ (Mistral only)          ▼
             ▼                   SQLite + Chroma ──OR── Postgres + pgvector
    jobs/consolidate.py          (zero setup)           (one DSN — stores,
    recall → archival,                                   budget ledger AND
    behind the privacy gate                              the checkpointer)
```

Three ideas carry the design.

**1. The agent manages its own memory.** Six tools: append/replace core memory,
search recall by keyword or date, insert/search archival by meaning. When the
context window crosses 70% the agent is *told*, and it summarizes old turns into
archival. Above 95% the paging happens whether it cooperates or not — bounding
the window is a safety property, not a request.

**2. Every LLM call goes through one router.** Providers are tried in priority
order with cooldowns, a persistent daily budget ledger, and a failure taxonomy
that distinguishes "slow down" from "you have no allowance here" — a distinction
that cost this project the whole of Phase 1 to learn. Every reply is stamped
with `served_by`.

**3. External content is data, never instructions.** Results from untrusted MCP
servers are pattern-neutralized, length-capped and wrapped in markers before the
model sees them; the verbatim original goes to the audit log. Once untrusted
content enters a turn, core memory is closed for the rest of it. Filesystem
writes are jailed to `./workspace` and suspend the turn for human approval.

Full as-built map: **[ARCHITECTURE.md](ARCHITECTURE.md)** ·
design rationale: **[PROJECT_SPEC.md](PROJECT_SPEC.md)**

---

## What's verified

The benchmark is deterministic and offline — every provider call is a scripted
fake and every check gets a fresh store — so a score delta is attributable to a
source change and nothing else.

```bash
make test     # pytest: no keys, no network
make bench    # 115 points, the CI regression gate
make stress   # unscored: long sessions, 50 facts, provider cooldowns
```

| Tier | | Pts |
|---|---|---|
| T1 | Router & provider health | 12 |
| T2 | Core memory | 12 |
| T3 | Recall memory | 12 |
| T4 | Archival + context pressure | 18 |
| T5 | Semantic retrieval quality | 16 |
| T6 | Tool dispatch safety | 10 |
| T7 | Resilience & degradation | 10 |
| T8 | Provenance | 10 |
| T10 | Background consolidation & the privacy gate | 5 |
| T11 | Prompt injection & memory poisoning | 10 |
| | **Total** | **115** |

**115/115 on both storage backends.** CI runs the whole suite twice — once on
SQLite + Chroma, once against a real pgvector service — because two
implementations behind one surface only stay equivalent if something checks.

The stress tier is unscored on purpose: it measures behaviour under load, and
load-shaped numbers drift with the machine. It is also where the last real bug
before v1.0.0 was found — 100 turns reaching **219%** of the context limit with
zero evictions. [Findings and the fix](BENCHMARKS.md).

---

## Deployment

**The supported deployment is local `docker compose`.** That is a finding, not a
shrug — the free tiers were checked in July 2026, before this was written:

| Platform | Status for this app |
|---|---|
| [Render](https://render.com/docs/free) free web service | **512 MB RAM, 0.1 CPU.** The API carries CPU-only torch and a local embedding model; it does not fit, and would be unusable at 0.1 CPU if it did. Free Postgres also **expires 30 days after creation** |
| [Fly.io](https://fly.io/docs/about/pricing/) | Free allowances **retired for new accounts** — a 2 VM-hour / 7-day trial, then paid |
| [Koyeb](https://www.koyeb.com/docs/faqs/pricing) | Free Starter tier **closed to new signups** after the Mistral acquisition; the instance was 512 MB / 0.1 vCPU regardless |
| [Hugging Face Spaces](https://huggingface.co/docs/hub/en/spaces-overview) | 16 GB RAM would fit comfortably, but **Docker Spaces now require a paid plan**; only Static Spaces are free |
| [Neon](https://neon.com/pricing) | **Genuinely free and permanent** — 0.5 GB, pgvector included, no card, no expiry |

The database has a real free option; the *application* does not. Rather than
recommend a deploy that would OOM on the first request, the honest $0
configuration is **run the stack locally and point it at Neon**, so your memory
outlives the laptop:

```bash
# In .env — nothing else changes.
MEMASSIST_POSTGRES_DSN=postgresql://user:pass@ep-xxx.neon.tech/memassist?sslmode=require
docker compose up --build
```

Setting the DSN is the entire switch. `assembly.build_stores()` follows it, the
budget ledger follows the stores, and so does the graph checkpointer — which
means suspended approvals become durable too.

For a public host, the smallest thing that actually works is a single VM with
~2 GB of RAM running this same compose file.
[Oracle Cloud Always Free](https://www.oracle.com/cloud/free/) is the only
always-free VM big enough, with real caveats: a card is required for
verification, ARM capacity is frequently unavailable, and the free A1 allowance
was cut to 2 OCPU / 12 GB in June 2026. **That path is not tested here**, and
this README will not claim it is.

Images publish to GHCR on every tag, after both test jobs pass:

```
ghcr.io/shrvan30/memassist-api:1.0.0
ghcr.io/shrvan30/memassist-web:1.0.0
```

`NEXT_PUBLIC_API_BASE` is baked into the web bundle at **build** time, so
serving the API from another origin means rebuilding the web image rather than
setting an environment variable.

---

## Configuration

Defaults live in [`config.py`](config.py); everything is env-overridable and
`.env.example` documents the lot. The ones that change behaviour:

| Variable | Default | Effect |
|---|---|---|
| `MEMASSIST_POSTGRES_DSN` | unset | Set it and the storage layer *and* the checkpointer switch to Postgres + pgvector |
| `MEMASSIST_CONTEXT_LIMIT` | `32000` | Planning cap; the real limit is `min(this, active provider window)` |
| `MEMASSIST_PRESSURE_THRESHOLD` | `0.7` | Where the agent is warned to offload |
| `MEMASSIST_MAX_HEARTBEATS` | `5` | Chained tool rounds per turn; each is a real request |
| `MEMASSIST_EXTERNAL_TOOLS` | `1` | `0` runs with memory tools only, no subprocesses |
| `LANGFUSE_PUBLIC_KEY` / `_SECRET_KEY` | unset | Both set enables tracing; otherwise it is entirely inert |

### Observability

With Langfuse keys set, each turn is one trace: a span per graph node and per
dispatched tool, tagged with the serving provider, plus named events for
sanitizer hits, guard denials and approval outcomes. Trace payloads pass through
the *same* redaction the background lane uses — a credential that may not go to
Mistral may not sit in a hosted dashboard either.

### The background lane

`jobs/consolidate.py` summarizes old recall into archival through Mistral, the
only provider on the `background` lane.

```bash
python -m jobs.consolidate --dry-run    # print the payload, send nothing
python -m jobs.consolidate              # one pass
docker compose --profile jobs up        # scheduled, opt-in
```

Opt-in because Mistral's free tier trains on prompts. Four independent filters
decide what may leave: conversation messages only (so verbatim external tool
results are excluded *structurally*, not by pattern-matching), no untrusted
markers, nothing the sensitivity detector flags, and no internal audit rows.
Withheld content is counted and reported **by category** — a silent filter is
indistinguishable from a broken one, which is exactly how a dead card-number
rule survived until the live run caught it.

---

## Repo layout

```
agent/            thin adapter + prompt/budget helpers
graph/            the turn cycle — nodes, edges, state
llm/              the failover router, budgets, error taxonomy
memory_server/    the six tools + both storage backends
security/         sanitizer, guards, sensitivity, injection corpus
api/              FastAPI (SSE)                web/     Next.js + Tailwind
jobs/             background consolidation     bench/   115 pts + stress tier
observability.py  Langfuse tracing (inert by default)
tests/            the pytest suite             workspace/  the filesystem jail
```

---

## Documentation

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | As-built file map, data flows, every deliberate deviation |
| [PROJECT_SPEC.md](PROJECT_SPEC.md) | The design and the phased plan it was built against |
| [BENCHMARKS.md](BENCHMARKS.md) | Every scored run, what each fix moved, the stress findings |
| [CHANGELOG.md](CHANGELOG.md) | Phase by phase |

---

## Author

Built by Shravan Upadhye. Claude Code served as the implementation agent; I
designed the architecture, wrote the specifications, built the benchmark and
security corpus, and verified every phase against it.

## References

- Packer et al., *MemGPT: Towards LLMs as Operating Systems* — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- [Letta](https://github.com/letta-ai/letta) (the MemGPT successor) and [LiteLLM](https://github.com/BerriAI/litellm) — read for reference, never copied
- [Model Context Protocol](https://modelcontextprotocol.io/)
