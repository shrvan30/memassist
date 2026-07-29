# BENCHMARKS — MemAssist

## How to run
```
make bench              # deterministic suite, 100 points
make bench LIVE=1       # + real-provider smoke (reported, never scored)
python -m bench --json out.json
```

## Scoring model
The suite is **deterministic and offline**: every provider call is a scripted
fake, and every check gets a fresh temp SQLite + Chroma directory. The score is
therefore reproducible, so a delta between two runs is attributable to a source
change and nothing else. Live provider calls are a separate, unscored smoke
section — they would otherwise make the number depend on free-tier weather.

Tiers map one-to-one onto the Phase 1.5 fix sprint, so each fix moves exactly
one tier:

| Tier | Capability | Pts | Sprint fix |
|---|---|---|---|
| T1 | Router & provider health | 12 | fix 1 — Gemini 0-req |
| T2 | Core memory | 12 | — |
| T3 | Recall memory | 12 | — |
| T4 | Archival + context pressure | 18 | fix 2 — FIFO eviction (T4c) |
| T5 | Semantic retrieval quality | 16 | fix 3 — bge-small |
| T6 | Tool dispatch safety | 10 | — |
| T7 | Resilience & degradation | 10 | fix 4 — friendly exhaustion (T7c) |
| T8 | Provenance | 10 | fix 5 — source tags |

> **This scale is NOT the Phase 1 "79/100".** That harness and its BENCHMARKS.md
> were not in the repo when the Phase 1.5 sprint began (only the score was
> quoted, in CLAUDE.md and PROJECT_SPEC.md §11), so it could not be re-run or
> diffed against. This suite was written from scratch against the spec and is
> deliberately harder on the five known Phase 1.5 gaps — it scores them at zero
> rather than partially crediting them. **Compare runs within this table only.**

---

## Run log

### Baseline — Phase 1 as-built (pre-sprint)
**58 / 100** · `pytest`: 56 passed

| Tier | Score |
|---|---|
| T1 Router & provider health | 8/12 |
| T2 Core memory | 12/12 |
| T3 Recall memory | 8/12 |
| T4 Archival + context pressure | 10/18 |
| T5 Semantic retrieval quality | 4/16 |
| T6 Tool dispatch safety | 10/10 |
| T7 Resilience & degradation | 6/10 |
| T8 Provenance | 0/10 |

Failing at baseline:
- **T1c (0/4)** — a `limit: 0` zero-quota 429 classifies as a plain
  `RateLimitError`, identical to a transient rate limit. This *is* the Gemini
  0-req bug: the provider is cooled down for 60s in a loop and silently never
  serves. A 404 also falls through unclassified.
- **T4c (0/8)** — no FIFO eviction. After an archival offload the in-context
  queue only grows (20 → 26 messages) and pressure never clears.
- **T5a/b/d (0/12)** — the hashing-bag-of-tokens embedder is lexical, so
  paraphrase probes that share no content words with their target retrieve the
  wrong passage. T5c passes only because "born" survives tokenization.
- **T7c (0/4)** — `AllProvidersExhausted` propagates raw to the caller; the UI
  would show a stack trace.
- **T8a/b (0/10)** — no provenance anywhere: archival metadata hardcodes
  `source="agent"`, and core-memory writes carry no source at all.

**Pre-existing failure, outside sprint scope:**
- **T3b (0/4)** — `conversation_search_date` accepts malformed dates
  (`"not-a-date"`) and silently returns "no matches" instead of an error, so the
  model gets no signal to correct its call. `SQLiteStore.search_messages_by_date`
  never raises `ValueError`, so the handler's error branch is dead code. Not one
  of the five sprint fixes; left failing deliberately so the baseline is honest.

---

### Fix 1 — Gemini 0-req root cause
**62 / 100 (+4)** · T1 8→12 · `pytest`: 56 passed

The root cause was **not** in `errors.py`. Verified against the live endpoint
with bare curl, outside the app:

| model | result |
|---|---|
| `gemini-2.0-flash` (was pinned) | 429 · `limit: 0` |
| `gemini-2.0-flash-001` | 429 · `limit: 0` |
| `gemini-2.5-flash` | 404 (listed by `/models`, but chat 404s) |
| `gemini-2.5-flash-lite` | **200, standard flat `tool_calls`** |
| `gemini-flash-latest` | 200, but 3.x — needs `thought_signature` |

The key was always valid (`/models` → 200). `providers.yaml` pinned
`gemini-2.0-flash`, which has **zero** free-tier allowance on this project, so
every call 429'd, the router applied a rolling 60s cooldown, and the
priority-1 provider served nothing for all of Phase 1 while looking merely
"busy". Treating that 429 as transient is what made it invisible.

Changes:
- `llm/providers.yaml` — repointed to `gemini-2.5-flash-lite` (verified to emit
  standard flat tool calls, so the 4-provider tool format is unaffected).
- `llm/errors.py` — new `ProviderPermanentError`; a 429 carrying `limit: 0`, a
  404, or a model-not-found marker now classifies as permanent, never as a
  rate limit. `ProviderAuthError` (401/403) subclasses it.
- `llm/router.py` — permanent errors are **never cooled down**: the provider is
  disabled for the process, logged at ERROR, and reported in `provider_status()`
  and in `AllProvidersExhausted` attempts.

Deviation from the brief, deliberate: the spec says these should "raise
loudly, no cooldown". Raising immediately would let one misconfigured provider
kill a turn the remaining three could serve, so the router instead disables it,
logs at ERROR, and keeps failing over — loud and uncooled, but not fatal. The
error type itself is distinct and does propagate if nothing else can serve.

**Done-condition met** — `make bench LIVE=1`:
```
gemini  model=gemini-2.5-flash-lite  available=True reason=ok
LIVE CALL -> served_by=gemini model=gemini-2.5-flash-lite content='ok'
```

---

### Fix 2 — FIFO eviction after archival offload
**70 / 100 (+8)** · T4 10→18 · `pytest`: 56 passed

Phase 1 did the summarizing half of the MemGPT mechanic but never the paging
half: the agent offloaded to archival and freed nothing, so the queue only grew
(20 → 26 messages across one pressure turn) and the warning re-fired every turn.

Changes (`agent/loop.py`):
- A successful `archival_memory_insert` while under pressure now sets an
  `offloaded` flag; after the tool round the loop calls `_evict_offloaded()`.
- `_evict_offloaded()` drops the oldest half of the FIFO, leaves an
  `[CONTEXT EVICTED]` marker at the head, and records an `eviction` event to
  recall for audit.
- `_safe_cut()` advances the cut past orphaned `tool` messages — evicting an
  assistant message while keeping its tool replies produces a transcript every
  OpenAI-compatible provider rejects.
- `_recompute_usage()` re-estimates from the retained queue, because the
  provider's `prompt_tokens` describes the *pre*-eviction prompt. The estimate
  is superseded by the real count on the next call.

**Done-condition met** — queue 20 → 15 and context pressure clears in the same
turn (`pressure_after=False`).

---

### Fix 3 — bge-small embedder swap + re-embed migration
**82 / 100 (+12)** · T5 4→16 · `pytest`: 56 passed

The Phase 1 embedder was a signed-hashing bag of tokens: purely lexical, so a
query only matched a passage it shared literal words with. Every paraphrase
probe retrieved the wrong passage — "Which medication makes him unwell?"
returned the passage about a mortgage.

Changes:
- `memory_server/storage/embedder.py` (new) — local `BAAI/bge-small-en-v1.5`,
  384-d, L2-normalized, lazily loaded and cached per process. No API key, no
  network after the one-time ~130 MB model download.
- `memory_server/storage/chroma.py` — defaults to `embedder.embed`.
  `hashing_embedding` is retained as an injectable zero-dependency fallback.
- `memory_server/storage/migrate_embeddings.py` (new) — one-time re-embed.

Migration ran against the live store (`data/` backed up to `data.bak` first):
```
Re-embedding 1 passage(s): 512-d -> 384-d
Done - 1 passage(s) now at 384-d.
Already 384-d (1 passages) - nothing to do.     # idempotent on re-run
```
A Chroma collection's vector width is fixed at creation, so the migration
rebuilds the collection from the stored passage text — no LLM involved, no
content regenerated.

**Done-condition met** — all four T5 paraphrase probes now retrieve their target
as top-1, each sharing no content words with it:
```
'Which medication makes him unwell?' -> 'He is allergic to penicillin...'
'What vehicle does he own?'          -> 'He drives a red Toyota Corolla...'
'Where was his child delivered?'     -> "The user's daughter Mira was born..."
'What dish does she enjoy most?'     -> 'Her favourite meal is grilled salmon...'
```

---

### Fix 4 — friendly provider-exhaustion copy
**86 / 100 (+4)** · T7 6→10 · `pytest`: 56 passed

`AllProvidersExhausted` propagated raw out of `AgentLoop.step()`, so a
free-tier stall surfaced in the UI as a stack trace.

Changes (`agent/loop.py`): the `router.chat` call is wrapped; on exhaustion the
loop logs the provider detail at WARNING, records it to recall, and returns
`PROVIDERS_EXHAUSTED_MESSAGE` as a normal reply. Anything already delivered
earlier in the turn still stands.

The loop imports only the exception type from `llm.errors` — it is part of the
`LLMRouter` contract the loop already depends on, so the "no provider details in
the loop" boundary holds.

**Done-condition met** — T7c: `"I've hit the free-tier limit on every
language-model provider I can reach…"`, with no exception name or traceback in
the user-facing text.

---

### Fix 5 — provenance tags (stated | inferred)
**96 / 100 (+10)** · T8 0→10 · `pytest`: 56 passed

Nothing recorded where a fact came from: archival hardcoded `source="agent"`
and core-memory writes carried no provenance at all.

Changes:
- `memory_server/schemas.py` — flat `source` enum (`stated` | `inferred`) on
  `core_memory_append` and `archival_memory_insert`, in both the Pydantic model
  and the JSON schema the model sees.
- `memory_server/memory_tools.py` — human-block lines are stored as
  `"Job: Data engineer [stated]"`; archival passes `source` through to metadata.
  The persona block is deliberately untagged: it is the agent's own identity,
  where "did the user say this?" has no meaning.
- `agent/prompts.py` — new PROVENANCE section instructing when to use each value.

Default is **`inferred`** everywhere. A fact is only ever labelled `stated` by an
explicit claim, so a mislabelled write under-claims rather than putting words in
the user's mouth. Phase 3 adds `external` for untrusted MCP content (§6.3).

A live end-to-end turn caught a bug the deterministic suite could not: the model
copied the rendered format and typed `[stated]` into `content` as well, yielding
`"Name: Shravan [stated] [inferred]"`. Fixed on both sides — the prompt now says
not to type the tag, and `_PROVENANCE_TAG_RE` strips one if the model does,
since the `source` argument is the only authority.

**Done-condition met** — live turn against `gemini-2.5-flash-lite`:
```
Name: Shravan [stated]
Job: Data engineer [stated]
Location: Pune [stated]
```

---

## Summary — Phase 1.5 fix sprint

| Step | Score | Δ | Tier moved |
|---|---|---|---|
| Baseline (Phase 1 as-built) | 58 | — | — |
| Fix 1 — Gemini 0-req root cause | 62 | +4 | T1 8→12 |
| Fix 2 — FIFO eviction | 70 | +8 | T4 10→18 |
| Fix 3 — bge-small + re-embed | 82 | +12 | T5 4→16 |
| Fix 4 — friendly exhaustion copy | 86 | +4 | T7 6→10 |
| Fix 5 — provenance tags | 96 | +10 | T8 0→10 |

**58 → 96 (+38).** `pytest`: 56 passed at every step. Each fix moved exactly one
tier, and no fix regressed another.

Remaining 4 points are the pre-existing **T3b** date-validation gap described
under the baseline — outside the five-fix scope, deliberately left failing.

Harness note: T4c was recalibrated during this fix. The original version let the
scripted `prompt_tokens` decide whether pressure cleared, which handed out
partial credit for no real work. It now reports `prompt_tokens=0` on the second
call, forcing the loop to estimate from the queue it actually holds. Re-verified
against the pre-fix code (`_evict_offloaded` stubbed to a no-op): still 0/8, so
the 58 baseline is unchanged.
