# Benchmarks

The suite is a regression gate: run it before and after a change, and any
movement in the number is attributable to that change. Nothing merges
below the deterministic floor.

## Running it

```bash
python -m bench                                   # 115/115 — offline, no keys, deterministic
MEMASSIST_BENCH_POSTGRES_DSN=postgresql://... python -m bench    # same suite on Postgres
MEMASSIST_BENCH_LIVE=1 MEMASSIST_BENCH_T12_PROVIDER=gemini python -m bench   # + live tier (125 ceiling)
python -m bench --cleanup-orphan-schemas          # drop leaked bench_* schemas
```

## What the numbers mean

**125 total. 115 offline and deterministic:** every model call is a
scripted FakeClient — including scripted failures; the "gemini auth
error" banner at startup is a scripted 401 from tier T1, proven under
`--network none` — and every check gets a fresh temporary store. CI gates
115 on **both** storage backends on every push, holding no keys. The
badge reads 115 because a badge should show what the pipeline measured.

**T12 (10 pts) is live and opt-in** (key AND `MEMASSIST_BENCH_LIVE=1` —
measurement is never ambient when it costs quota). It measures what
scripted tiers structurally cannot: whether the agent **answers from what
it retrieved** instead of asking the user to repeat it. One pinned
provider, temperature 0, grading as a pure function of the reply text; a
429 mid-tier is retried with a pause, not scored — "the free tier is
busy" and "the agent failed to use its memory" are different facts. Live
results are provider-sensitive and recorded honestly in this file's
repo-history: a weak model scored 3/10 where a stronger one scored 10/10,
identically across runs — model behavior, not harness flake.

## The tiers

T1 failover · T2 core memory · T3 recall search · T4 archival + pressure
· T5 semantic retrieval · T6 dispatch safety · T7 degradation · T8
provenance · T10 consolidation privacy · T11 injection defense · T12
memory utilization (live). T9 does not exist — tier ids are append-only
identifiers, and renumbering would invalidate every historical score.

## The rules this suite taught the project

1. **Test the mechanic, not the arithmetic.** T4c once hardcoded a token
   budget that silently went stale and became unfailable in one
   direction; it now measures the prompt floor and derives its budget.
2. **Prove a check can fail.** Stub eviction to a no-op and watch T4c
   drop to 0/8. A test you have never seen fail proves nothing.
3. **A score without a committed harness is a claim.** An early 79/100
   was never committed, so it could not be re-run or diffed; the rebuilt
   suite re-baselined at 58 and earned the ceiling with attributable
   per-fix deltas. Pre- and post-rebaseline scores are incomparable.

## The stress tier (unscored, by design)

Load-shaped numbers move with the machine, so the 100-turn session,
50-fact retrieval precision (84% P@1 / 94% top-3 / ~40ms), rapid-fire
cooldown behavior, and long-document recall are reported as findings —
and the 100-turn scenario found the unbounded-context bug (219% of the
window, zero evictions) that all 115 scored points missed.

## Postgres warning

The bench reads `MEMASSIST_BENCH_POSTGRES_DSN` — deliberately not the
app's DSN, because the suite issues destructive DDL continuously (one
`bench_<12-hex>` schema per check, dropped in the same `finally` as the
temp dir). Cleanup regex-matches that exact shape rather than SQL
`LIKE 'bench_%'`, because underscore is itself a single-character
wildcard.
