# Development

Local setup, the test doctrine, what CI runs, and how changes ship.

## Setup — two modes

**Zero-install (SQLite + Chroma):**
```bash
pip install -r requirements.txt
cp .env.example .env          # one provider key, unquoted
uvicorn api.main:app --reload # api on 8000
cd web && npm install && npm run dev   # web on 3000
```

**Postgres mode:** `docker compose up -d postgres`, set
`MEMASSIST_POSTGRES_DSN` (host port 15432) — storage, checkpointer, and
budgets switch as one unit.

## Running the checks

```bash
ruff check .
python -m pytest -q      # ~250 tests, no keys, no network
python -m bench          # 115/115 deterministic — the merge floor
```

## The test doctrine

**Construct the condition under test rather than inherit it from the
shell.** Three incidents taught it: config resolvers are tested through
the function with the env var explicitly deleted AND set (never against
import-time constants); DSN-absence tests need `monkeypatch.delenv` PLUS
neutralizing config's cached module value; and any `.env` line can blind
an absence-test — assume one exists. The seams that keep everything
offline: `client_factory` (scripted provider transports, including
scripted 401/429/zero-quota), `now_fn` (injected time — cooldowns and
midnight rollover tested by moving a clock, never sleeping), FakeRouter
scenarios for the graph, temp stores per test, and the hashing embedder
as the deterministic test double (the real model appears only where
semantic quality is the thing measured). Storage tests are parametrized
across both backends: "differences are dialect, not behaviour" is
enforced, not asserted. Every test carries a timeout — a hang must become
a named two-minute failure, never a 14-minute mystery on a remote runner.

## What CI runs (every push and PR)

ruff -> pytest + the deterministic benchmark on **both** backends (a
matrix leg each, so a divergence names its backend) -> gitleaks secret
scan -> pip-audit (one scoped, commented ignore) -> web lint + production
build. Failures are republished as annotations, so diagnosis never needs
runner-log access. The HF model cache is keyed by the dependency lockfile
hash with `HF_HOME` pinned — the incident that motivated it was 706s vs
12s for the same tree.

## How changes ship

Branch per change -> PR with a written report (what changed, what was
verified, what was NOT verified) -> CI green -> **merge commit** (never
squash; the history is part of the record) -> delete the branch. The
deterministic bench is the floor: nothing merges below 115/115 on both
backends. Release: annotated `v*` tag on the gated commit -> CI publishes
multi-arch images -> GitHub Release from the CHANGELOG.
