# Benchmarks

## Result

**125 / 125 on both storage backends** — SQLite + Chroma and Postgres + pgvector
score identically, which is the point of running both. `pytest`: 251 passed
(Postgres) / 231 passed + 20 skipped (SQLite, where the Postgres tests skip
rather than silently pass).

Of that, **115 points are offline and deterministic** and 10 (T12) require a
live provider. T12 is opt-in: a plain run scores out of 115 and spends nothing.

```bash
make bench                            # the scored offline suite, out of 115
MEMASSIST_BENCH_LIVE=1 python -m bench   # adds T12, out of 125 — spends quota
make bench LIVE=1                     # adds a provider smoke test, not scored
make stress                           # unscored load scenarios
python -m bench --json out.json
```

## What each tier measures

| Tier | Measures | Pts |
|---|---|---|
| T1 | Provider failover: transient 429 cools down and moves on, zero-quota and 404 disable permanently, budgets skip proactively | 12 |
| T2 | Core memory: append, replace, character limits, and survival across a store reopen | 12 |
| T3 | Recall search: keyword matching, date ranges, and a correctable error on a malformed date | 12 |
| T4 | Archival writes and context pressure: the warning fires, the offload happens, the queue actually shrinks | 18 |
| T5 | Semantic retrieval: paraphrased questions that share no words with their target passage | 16 |
| T6 | Tool dispatch safety: unknown tool names, bad arguments, and the heartbeat cap | 10 |
| T7 | Degradation: archival unavailable, invalid input, and every provider exhausted | 10 |
| T8 | Provenance: `stated` vs `inferred` recorded correctly on both core and archival writes | 10 |
| T10 | Background consolidation: the privacy filter holds, the lane stays on Mistral, an all-sensitive window spends zero requests | 5 |
| T11 | Prompt injection and memory poisoning: web content cannot write core memory, instructions are neutralized, filesystem writes are gated | 10 |
| T12 | Memory utilization — see below. **Live, not part of the deterministic suite** | 10 |
| | **Total** | **125** |

T9 does not exist; the numbering left room for a tier that was never needed.

## T12 — memory utilization (live)

Whether the agent *answers* from what it retrieved instead of asking the user to
repeat it. One line per case:

| Case | Setup and probe | Pass | Pts |
|---|---|---|---|
| T12a | A three-month plan the user never called a "goal"; asked "what is my 3 month goal?" | States the plan, attributed. Asking the user scores zero | 4 |
| T12b | "VidRAG due 30 August"; asked "it's mid-September, what did I miss?" | Names the deadline and says it has passed, relative to the date the user gave | 3 |
| T12c | A stated goal, probed in words the user never used | Answers with provenance and no counter-question | 3 |

**This tier is not deterministic, and cannot be.** T1–T11 drive scripted fake
routers; a scripted T12 reply would grade the harness's own fixtures rather than
the agent, because what is being measured is a decision the model makes. So T12
issues real requests against one pinned provider
(`MEMASSIST_BENCH_T12_PROVIDER`, default `openrouter`) at temperature 0. The
grading is deterministic — given a reply, the verdict is a pure function of its
text — but the reply is not.

The provider is pinned rather than allowed to fail over, since failover would
change which model is being graded partway through a run. Provider exhaustion is
retried with a pause: a 429 means the free tier is busy, not that the agent
failed to use its memory.

**T12 is registered only when `MEMASSIST_BENCH_LIVE=1` AND the pinned provider
has a key.** Either one missing and the tier does not exist, the ceiling is 115,
and the run makes no network call — so CI, which holds no keys, stays offline,
reproducible and green.

The environment variable is the newer half of that condition, and it exists
because a key alone was too easy to satisfy. Anyone with a working `.env` spent
real quota on every routine `python -m bench`: three turns of up to five
heartbeats each. Quota burned that way then fails T12 for reasons that have
nothing to do with the agent — the reply becomes "I've hit the free-tier limit",
which scores zero exactly like a wrong answer would. The tier was costing
something and measuring nothing. Running the release gate is now a deliberate
act:

```bash
MEMASSIST_BENCH_LIVE=1 MEMASSIST_BENCH_T12_PROVIDER=gemini python -m bench
```

Pick the provider with quota to spare rather than the strongest model on paper.
Providers differ more in their free-tier ceilings than in their ability to pass
this tier: a chain member with a low tokens-per-minute limit fails T12 on
arithmetic, since three ~3,000-token turns in quick succession exceed it
regardless of how good the model is.

Because the score depends on the pinned model's instruction-following, T12 is a
measurement of the whole system rather than of the memory layer alone. Treat a
T12 regression as a prompt or model question first.

## Running against Postgres

```bash
MEMASSIST_BENCH_POSTGRES_DSN=postgresql://memassist:memassist@localhost:15432/memassist \
  python -m bench
```

The header names the backend, so a run that says `[sqlite+chroma]` when you
expected Postgres has not picked up the DSN.

**That variable is deliberately not `MEMASSIST_POSTGRES_DSN`, and the two are
not aliased.** Every check creates a throwaway schema and drops it, so the
suite issues destructive DDL continuously; if it read the application's DSN it
would do that to whatever database the app is configured for. Setting the app's
variable therefore has no effect on the benchmark — which used to look exactly
like a broken benchmark, so a run in that state now prints a warning saying so.

Schemas are dropped in the same `finally` that removes each check's temp
directory, so a failing check cleans up as reliably as a passing one. Runs from
before that existed left one schema behind per check:

```bash
MEMASSIST_BENCH_POSTGRES_DSN=... python -m bench --cleanup-orphan-schemas
# prints the count, asks for confirmation; --yes skips the prompt
```

It only drops names matching `bench_` plus twelve hex digits — exactly what the
harness generates. A SQL `LIKE 'bench_%'` filter alone would not be safe, since
`_` is itself a single-character wildcard and would also match a schema someone
had called `benchmarks`.

## Method

T1–T11 are deterministic and offline. Every provider call is a scripted fake
and every check gets a fresh temporary store, so the number reproduces on any
machine and a change in it is attributable to a change in the source. They run
in CI against both backends and fail the build below 115. T12 is the exception
and is described above.

`LIVE=1` adds one real request per configured provider. Those results are
printed but never scored, because free-tier availability varies by the hour and
would otherwise make the number depend on the weather.

**The scale was re-baselined once.** An earlier harness reported 79/100 but was
never committed, so it could not be re-run or diffed. The current suite was
written from scratch and is deliberately harder on the gaps that harness missed.
Scores from before the re-baseline are not comparable to scores after it; only
runs within the current table can be compared.

## Stress tier

Unscored, run with `make stress`. These measure behaviour under load, and
load-shaped numbers move with the machine — scoring them would undermine the
reproducibility the main suite depends on.

**100-turn session.** Context stays bounded at 95% of the limit, three evictions
fire, and all ten facts stated in the first ten turns are still retrievable
ninety turns later.

This scenario found the one substantive bug of the last release. Eviction ran
only after the model chose to summarize, so a model that ignored the pressure
warning grew the window without limit: 100 turns reached **219%** of the context
limit with zero evictions. Bounding the window is now independent of the model's
cooperation, and the forced path says the messages were dropped without being
summarized rather than claiming a summary that was never written.

**50-fact retrieval.** 84% precision@1, 94% within the top 3, roughly 40 ms per
query. Every failure is the probe landing on a neighbouring fact in the same
subject area — a question about medication retrieving a note about blood type.
Since the agent retrieves five passages by default, it still sees the right one
in 94% of cases.

**20 messages under provider failure.** With the first-choice provider
unavailable for the opening stretch and the second dying across an overlapping
window, all 20 turns were answered and the recall log holds both halves of every
exchange. The failover is invisible except in the per-reply provider badge.

**Ten-page document, recalled 20 turns later.** One specific fact buried in the
middle of uniform filler, retrieved at rank 1 by all three probes including a
bare identifier string.

## Notes on two checks

**T4c measures its own budget.** It originally hardcoded a token budget, which
silently went stale when the system prompt grew and made the check unfailable in
one direction. It now measures the prompt floor and derives the budget, and was
re-verified by stubbing eviction to a no-op: the check drops to zero, so it is
still measuring the mechanic rather than the arithmetic.

**T11 shares its corpus with the test suite.** `security/injections/*.yaml` is
read by both the benchmark and `tests/test_injections.py`, so an attack case is
written once and both harnesses pick it up. Cases have been added after live
findings, including a payload whose only distinguishing feature is a forged
closing marker.
