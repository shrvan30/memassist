# Configuration

Everything is set via `.env` (copy from `.env.example`). Values must be
**unquoted**: docker compose strips quotes but `docker run --env-file`
does not — a quoted key arrives two characters too long and 401s.

## Provider keys (at least one required — all free)

| Variable | Where to get it |
|---|---|
| `GEMINI_API_KEY` | aistudio.google.com (`GOOGLE_API_KEY` is a deprecated alias — honored, logs a warning) |
| `GROQ_API_KEY` | console.groq.com |
| `OPENROUTER_API_KEY` | openrouter.ai |
| `MISTRAL_API_KEY` | console.mistral.ai (also powers the background consolidation lane) |

## Model behavior

| Variable | Default | Meaning |
|---|---|---|
| `MEMASSIST_TEMPERATURE` | 0.3 | Fixed across providers for a stable voice through failover |
| `MEMASSIST_PRESSURE_THRESHOLD` | 0.7 | Context fraction at which the memory-pressure warning fires |
| `MEMASSIST_REQUEST_TIMEOUT` | 30.0 | Per-request cap; interacts with retries and the 120s turn ceiling |
| `MEMASSIST_PROVIDERS_YAML` | ./llm/providers.yaml | Provider chain config |

## Storage

| Variable | Meaning |
|---|---|
| `MEMASSIST_DB_PATH`, `MEMASSIST_CHROMA_PATH` | SQLite + Chroma locations (defaults just work) |
| `MEMASSIST_POSTGRES_DSN` | Setting it switches **four things as one unit**: core/recall store, archival store, graph checkpointer, budget ledger. Host-side DSN uses port `15432`; services inside compose use `postgres:5432`. |

## Compose

`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` (change the
password before exposing beyond localhost) · `POSTGRES_HOST_PORT`
(default **15432**, deliberately not 5432 — see
[deployment.md](deployment.md)) · `API_PORT`, `WEB_PORT` ·
`NEXT_PUBLIC_API_BASE` — **baked into the web bundle at build time**;
cannot change at runtime · `MEMASSIST_EXTERNAL_TOOLS=0` in the api image
(it ships no uvx/npx).

## Observability (optional — fully inert when unset)

`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`. With
keys unset: no client, no threads, no network — the benchmark's
determinism depends on this. Every trace payload passes the same
sensitivity filter that gates the Mistral lane.

## Background consolidation

`MEMASSIST_CONSOLIDATE_EVERY` (e.g. `6h`; non-positive rejected — a zero
would busy-loop a 2-RPM API) · `MEMASSIST_CONSOLIDATE_LIMIT`. Manual:
`python -m jobs.consolidate --dry-run`.

## Benchmark

`MEMASSIST_BENCH_POSTGRES_DSN` — the bench's **own** DSN, deliberately
not aliased to the app's (the suite issues destructive DDL) ·
`MEMASSIST_BENCH_LIVE=1` + `MEMASSIST_BENCH_T12_PROVIDER` — the live
tier's double gate.
