"""MemAssist benchmark harness — deterministic, offline, 100 points.

Run with ``make bench`` (or ``python -m bench``). Every check uses scripted
fakes instead of live providers, so a score delta between two runs is
attributable to a source change and nothing else. ``--live`` additionally runs
a real-provider smoke test; it is reported separately and does NOT affect the
score, so the number stays reproducible in CI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from llm import errors  # noqa: E402
from llm.budgets import BudgetLedger  # noqa: E402
from llm.router import ChatResult, ProviderConfig, Router, ToolCall, Usage  # noqa: E402
from memory_server.memory_tools import MemoryTools  # noqa: E402
from memory_server.storage.chroma import ArchivalStore  # noqa: E402
from memory_server.storage.sqlite import SQLiteStore  # noqa: E402

# --- registry -------------------------------------------------------------
CHECKS: list[tuple[str, str, int, object]] = []
TIER_NAMES = {
    "T1": "Router & provider health",
    "T2": "Core memory",
    "T3": "Recall memory",
    "T4": "Archival + context pressure",
    "T5": "Semantic retrieval quality",
    "T6": "Tool dispatch safety",
    "T7": "Resilience & degradation",
    "T8": "Provenance",
}


def check(cid: str, points: int):
    def deco(fn):
        CHECKS.append((cid[:2], cid, points, fn))
        return fn

    return deco


# --- shared fakes ---------------------------------------------------------
class FakeHTTPError(Exception):
    """Stands in for an openai SDK error (it exposes .status_code / .code)."""

    def __init__(self, status_code: int, message: str = "", code: str = "") -> None:
        self.status_code = status_code
        self.code = code
        super().__init__(message)


# Verbatim shape of the real Gemini free-tier refusal.
ZERO_QUOTA_BODY = (
    "You exceeded your current quota, please check your plan and billing details. "
    "* Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests, limit: 0, model: gemini-2.0-flash"
)
TRANSIENT_429_BODY = "Rate limit reached for model. Please retry shortly."


class FakeClient:
    """A provider transport that replays a scripted list of outcomes."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def raw_reply(model: str, content: str = "ok"):
    """Minimal duck-typed stand-in for an OpenAI chat-completion response."""

    class _Fn:
        name = "send_message"
        arguments = '{"text":"ok"}'

    class _TC:
        id = "call_1"
        function = _Fn()

    class _Msg:
        def __init__(self):
            self.content = content
            self.tool_calls = [_TC()]

    class _Choice:
        def __init__(self):
            self.message = _Msg()
            self.finish_reason = "tool_calls"

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Raw:
        def __init__(self):
            self.choices = [_Choice()]
            self.usage = _Usage()
            self.model = model

    return _Raw()


def build_router(tmp: Path, scripts: dict[str, list]) -> tuple[Router, dict]:
    cfgs = [
        ProviderConfig(
            name="gemini", priority=1, model="m-gemini",
            base_url="http://x", api_key_env="GEMINI_API_KEY", context_window=100_000,
        ),
        ProviderConfig(
            name="groq", priority=2, model="m-groq",
            base_url="http://x", api_key_env="GROQ_API_KEY", context_window=50_000,
        ),
    ]
    clients = {name: FakeClient(script) for name, script in scripts.items()}
    ledger = BudgetLedger(str(tmp / "bench.db"))
    router = Router(
        cfgs,
        ledger,
        client_factory=lambda cfg, key: clients[cfg.name],
        api_keys={"gemini": "k1", "groq": "k2"},
        sleep_fn=lambda s: None,
        server_retries=0,
    )
    return router, clients


class FakeRouter:
    """Scripted router for loop-level checks (mirrors tests/test_loop.py)."""

    def __init__(self, scripted, default=None, window=100_000):
        self.scripted = list(scripted)
        self.default = default
        self.window = window
        self.calls = 0

    def chat(self, messages, tools=None, *, tool_choice="auto", lane="interactive"):
        self.calls += 1
        if self.scripted:
            item = self.scripted.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self.default is not None:
            return self.default()
        raise AssertionError("FakeRouter ran out of scripted responses")

    def context_window(self, provider):
        return self.window

    def min_context_window(self):
        return self.window


def result(content=None, tool_calls=None, served_by="gemini", prompt_tokens=100):
    return ChatResult(
        served_by=served_by,
        model=f"{served_by}-model",
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5),
    )


def tc(name, arguments, cid="call_1"):
    return ToolCall(id=cid, name=name, arguments=arguments)


def make_memory(tmp: Path, with_archival: bool = True) -> MemoryTools:
    store = SQLiteStore(
        str(tmp / "mem.db"),
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=2000,
    )
    archival = ArchivalStore(str(tmp / "chroma")) if with_archival else None
    return MemoryTools(store, archival, session_id="bench", page_size=3, archival_top_k=5)


def make_loop(mem, router, **kw):
    params = dict(planning_context_limit=1000, pressure_threshold=0.7, max_heartbeats=5)
    params.update(kw)
    return AgentLoop(router, mem, tools=[], **params)


# =========================================================================
# T1 — Router & provider health (12)
# =========================================================================
@check("T1a", 4)
def t1a_transient_429_fails_over(tmp):
    """A real transient 429 cools the provider down and the chain fails over."""
    router, _ = build_router(
        tmp,
        {
            "gemini": [FakeHTTPError(429, TRANSIENT_429_BODY)],
            "groq": [raw_reply("m-groq")],
        },
    )
    res = router.chat([{"role": "user", "content": "hi"}])
    cooling = router.ledger.is_cooling("gemini")
    return res.served_by == "groq" and cooling, f"served_by={res.served_by} cooling={cooling}"


@check("T1b", 4)
def t1b_auth_error_skips_without_cooldown(tmp):
    """401 is a misconfiguration, not a rate limit: skip, never cool down."""
    router, _ = build_router(
        tmp,
        {"gemini": [FakeHTTPError(401, "bad key")], "groq": [raw_reply("m-groq")]},
    )
    res = router.chat([{"role": "user", "content": "hi"}])
    cooling = router.ledger.is_cooling("gemini")
    return res.served_by == "groq" and not cooling, f"served_by={res.served_by} cooling={cooling}"


@check("T1c", 4)
def t1c_permanent_quota_is_not_a_transient_429(tmp):
    """`limit: 0` means the model has NO free-tier allowance — permanent.

    Classifying it as a plain RateLimitError would cool the provider down for
    60s forever so it silently never serves. A 404 (model retired/unavailable)
    is likewise permanent. Both must be distinguishable from a transient 429.
    """
    zero = FakeHTTPError(429, ZERO_QUOTA_BODY)
    transient = FakeHTTPError(429, TRANSIENT_429_BODY)
    notes = []

    def classify(exc):
        try:
            return errors.classify_provider_error("gemini", exc)
        except Exception as raised:
            return raised

    zero_cls = classify(zero)
    transient_cls = classify(transient)
    four04 = classify(FakeHTTPError(404, "model not found"))

    transient_ok = isinstance(transient_cls, errors.RateLimitError)
    zero_ok = not (
        type(zero_cls) is errors.RateLimitError  # noqa: E721 - exact type on purpose
    )
    notes.append(f"transient={type(transient_cls).__name__}")
    notes.append(f"zero_quota={type(zero_cls).__name__}")
    notes.append(f"404={type(four04).__name__}")
    # A 404 must not be silently swallowed into a cooldown-and-continue either.
    four04_ok = not isinstance(four04, (errors.RateLimitError, errors.QuotaExceededError))
    return (transient_ok and zero_ok and four04_ok), " ".join(notes)


# =========================================================================
# T2 — Core memory (12)
# =========================================================================
@check("T2a", 4)
def t2a_append_and_replace(tmp):
    mem = make_memory(tmp)
    mem.core_memory_append("human", "Name: Alice.")
    mem.core_memory_replace("human", "Alice", "Alicia")
    rendered = mem.render_core_memory()
    return "Alicia" in rendered and "<human>" in rendered, rendered[:60].replace("\n", " ")


@check("T2b", 4)
def t2b_char_limit_returns_error_string(tmp):
    mem = make_memory(tmp)
    out = mem.core_memory_append("human", "x" * 5000)
    return out.startswith("Error:"), out[:60]


@check("T2c", 4)
def t2c_persists_across_restart(tmp):
    mem = make_memory(tmp)
    mem.core_memory_append("human", "Name: Bob.")
    mem.store.close()
    reopened = SQLiteStore(
        str(tmp / "mem.db"),
        default_persona=config.DEFAULT_PERSONA,
        default_human=config.DEFAULT_HUMAN,
        core_block_char_limit=2000,
    )
    blocks = reopened.get_core_blocks()
    reopened.close()
    return "Bob" in blocks.get("human", ""), blocks.get("human", "")[:60]


# =========================================================================
# T3 — Recall memory (12)
# =========================================================================
@check("T3a", 4)
def t3a_keyword_search_with_provenance(tmp):
    mem = make_memory(tmp)
    mem.record_event("assistant", "message", "Your flight to Lisbon is booked.", served_by="groq")
    rows, total = mem.store.search_messages("Lisbon")
    return total >= 1 and any(r["served_by"] == "groq" for r in rows), f"total={total}"


@check("T3b", 4)
def t3b_date_search_and_bad_input(tmp):
    mem = make_memory(tmp)
    mem.record_event("user", "message", "hello there")
    good = mem.conversation_search_date("2000-01-01", "2100-01-01")
    bad = mem.conversation_search_date("not-a-date", "also-bad")
    return "hello there" in good and bad.startswith("Error:"), bad[:50]


@check("T3c", 4)
def t3c_pagination(tmp):
    mem = make_memory(tmp)
    for i in range(7):
        mem.record_event("user", "message", f"note number {i}")
    p0 = mem.conversation_search("note", page=0)
    p1 = mem.conversation_search("note", page=1)
    return ("page 0" in p0 and "page 1" in p1 and p0 != p1), "paged"


# =========================================================================
# T4 — Archival + context pressure (18)
# =========================================================================
@check("T4a", 5)
def t4a_archival_insert_and_search(tmp):
    mem = make_memory(tmp)
    mem.archival_memory_insert("The user signed the lease on 3 March.")
    out = mem.archival_memory_search("lease")
    return "lease" in out and mem.archival.count() == 1, out[:60].replace("\n", " ")


@check("T4b", 5)
def t4b_pressure_warning_injected(tmp):
    mem = make_memory(tmp)
    router = FakeRouter([result(tool_calls=[tc("send_message", '{"text":"ok"}')], prompt_tokens=50)])
    loop = make_loop(mem, router)
    loop.last_input_tokens, loop.last_limit = 800, 1000  # 80% > 70%
    loop.step("carry on")
    injected = [
        m for m in loop.messages
        if isinstance(m.get("content"), str) and "[MEMORY PRESSURE]" in m["content"]
    ]
    return len(injected) == 1, f"injected={len(injected)}"


@check("T4c", 8)
def t4c_fifo_eviction_after_offload(tmp):
    """Offloading to archival must FREE context.

    Under pressure the agent summarizes into archival — and the loop must then
    EVICT those summarized messages from the in-context FIFO and recompute
    usage. Without eviction the queue only ever grows and the warning fires
    forever.
    """
    mem = make_memory(tmp)
    router = FakeRouter(
        [
            result(
                tool_calls=[tc(
                    "archival_memory_insert",
                    '{"content":"Summary of the earlier conversation.",'
                    '"request_heartbeat":true}',
                )],
                prompt_tokens=2400,
            ),
            # prompt_tokens=0 forces the loop to estimate usage from the queue it
            # ACTUALLY still holds, so a stale pre-eviction count cannot fake the
            # recovery. Un-evicted this estimates ~2370 tokens; evicted, ~1580,
            # against a 1960 threshold — so only real eviction clears pressure.
            # (The ~616-token system prompt is a floor eviction cannot touch, so
            # the evictable messages have to dominate it for a clean signal.)
            result(tool_calls=[tc("send_message", '{"text":"done"}', cid="call_2")], prompt_tokens=0),
        ]
    )
    loop = make_loop(mem, router, planning_context_limit=2800)
    # 20 stale turns already in context, over the threshold.
    filler = "recap of an earlier exchange, " * 10
    for i in range(10):
        loop.messages.append({"role": "user", "content": f"old user message {i}: {filler}"})
        loop.messages.append({"role": "assistant", "content": f"old reply {i}: {filler}"})
    before = len(loop.messages)
    loop.last_input_tokens, loop.last_limit = 2400, 2800

    loop.step("please summarize and continue")

    after = len(loop.messages)
    shrank = after < before
    relieved = not loop.under_pressure()
    notes = f"queue {before}->{after} pressure_after={loop.under_pressure()}"
    if shrank and relieved:
        return True, notes
    if shrank or relieved:
        return 0.5, notes  # partial credit
    return False, notes


# =========================================================================
# T5 — Semantic retrieval quality (16)
# =========================================================================
T5_CORPUS = [
    "The user's daughter Mira was born in Pune in 2019.",
    "He drives a red Toyota Corolla with a dented rear bumper.",
    "Her favourite meal is grilled salmon with lemon and dill.",
    "The mortgage on the house was refinanced at 4.2 percent.",
    "He is allergic to penicillin and breaks out in hives.",
]
# Paraphrase probes: deliberately share NO content words with their target, so
# a lexical/hashing embedder cannot score them. (query, expected substring)
T5_PROBES = [
    ("Which medication makes him unwell?", "penicillin"),
    ("What vehicle does he own?", "Toyota"),
    ("Where was his child delivered?", "Mira"),
    ("What dish does she enjoy most?", "salmon"),
]


def _t5_probe(tmp, idx):
    mem = make_memory(tmp)
    for passage in T5_CORPUS:
        mem.archival_memory_insert(passage)
    query, expected = T5_PROBES[idx]
    items, _ = mem.archival.search(query, top_k=1)
    top = items[0]["content"] if items else ""
    return (expected in top), f"{query!r} -> {top[:44]!r}"


@check("T5a", 4)
def t5a(tmp):
    return _t5_probe(tmp, 0)


@check("T5b", 4)
def t5b(tmp):
    return _t5_probe(tmp, 1)


@check("T5c", 4)
def t5c(tmp):
    return _t5_probe(tmp, 2)


@check("T5d", 4)
def t5d(tmp):
    return _t5_probe(tmp, 3)


# =========================================================================
# T6 — Tool dispatch safety (10)
# =========================================================================
@check("T6a", 3)
def t6a_unknown_tool(tmp):
    mem = make_memory(tmp)
    out = mem.dispatch("rm_rf_everything", {})
    return out.startswith("Error:"), out[:50]


@check("T6b", 3)
def t6b_invalid_args_never_raise(tmp):
    mem = make_memory(tmp)
    out = mem.dispatch("core_memory_append", {"wrong": "shape"})
    return out.startswith("Error:"), out[:50]


@check("T6c", 4)
def t6c_heartbeat_cap(tmp):
    mem = make_memory(tmp)
    router = FakeRouter(
        [],
        default=lambda: result(tool_calls=[tc("core_memory_append", '{"block":"human","content":"x"}')]),
    )
    loop = make_loop(mem, router, max_heartbeats=3)
    loop.step("hello")
    return router.calls == 3, f"calls={router.calls}"


# =========================================================================
# T7 — Resilience & degradation (10)
# =========================================================================
@check("T7a", 3)
def t7a_archival_unavailable(tmp):
    mem = make_memory(tmp, with_archival=False)
    out = mem.archival_memory_insert("something")
    return out.startswith("Error:") and "archival" in out, out[:50]


@check("T7b", 3)
def t7b_tool_failure_is_text_not_crash(tmp):
    mem = make_memory(tmp)
    out = mem.dispatch("core_memory_append", {"block": "nonexistent_block", "content": "x"})
    return out.startswith("Error:"), out[:50]


@check("T7c", 4)
def t7c_friendly_exhaustion_message(tmp):
    """When every provider is down the user must get plain language, not a stack trace."""
    mem = make_memory(tmp)
    router = FakeRouter([errors.AllProvidersExhausted({"gemini": "rate_limited", "groq": "quota_exceeded"})])
    loop = make_loop(mem, router)
    try:
        outputs = loop.step("are you there?")
    except errors.AllProvidersExhausted:
        return False, "raw AllProvidersExhausted propagated to caller"
    except Exception as exc:
        return False, f"unexpected {type(exc).__name__}"
    if not outputs:
        return False, "no message returned to the user"
    text = " ".join(outputs).lower()
    leaked = "allprovidersexhausted" in text or "traceback" in text
    friendly = any(w in text for w in ("try again", "shortly", "later", "capacity", "quota", "limit"))
    return (friendly and not leaked), f"{outputs[0][:60]!r}"


# =========================================================================
# T8 — Provenance (10)
# =========================================================================
@check("T8a", 5)
def t8a_archival_metadata_source(tmp):
    """Archival passages must record whether a fact was stated or inferred."""
    mem = make_memory(tmp)
    try:
        mem.dispatch(
            "archival_memory_insert",
            {"content": "The user prefers window seats.", "source": "stated"},
        )
    except Exception as exc:
        return False, f"raised {type(exc).__name__}"
    items, _ = mem.archival.search("window seats", top_k=1)
    if not items:
        return False, "passage not retrievable"
    src = items[0].get("source")
    return src in ("stated", "inferred"), f"source={src!r}"


@check("T8b", 5)
def t8b_human_block_lines_tagged(tmp):
    """Human-block lines carry a provenance marker."""
    mem = make_memory(tmp)
    try:
        mem.dispatch(
            "core_memory_append",
            {"block": "human", "content": "Works as a nurse.", "source": "stated"},
        )
    except Exception as exc:
        return False, f"raised {type(exc).__name__}"
    human = mem.core_blocks().get("human", "")
    tagged = "stated" in human.lower()
    prompt_rule = "source" in _system_prompt_text().lower()
    if tagged and prompt_rule:
        return True, human.strip()[:60]
    if tagged or prompt_rule:
        return 0.5, f"tagged={tagged} prompt_rule={prompt_rule}"
    return False, f"tagged={tagged} prompt_rule={prompt_rule}"


def _system_prompt_text() -> str:
    from agent.prompts import BASE_SYSTEM_PROMPT

    return BASE_SYSTEM_PROMPT


# --- live smoke (not scored) ---------------------------------------------
def run_live_smoke() -> list[str]:
    """One real request per configured provider. Reported, never scored."""
    from assembly import build_router as build_real_router

    lines = []
    router = build_real_router()
    for status in router.provider_status():
        lines.append(
            f"  {status['name']:<11} model={status['model']:<28} "
            f"available={status['available']} reason={status['reason']}"
        )
    try:
        res = router.chat([{"role": "user", "content": "Reply with the single word: ok"}])
        lines.append(f"  LIVE CALL -> served_by={res.served_by} model={res.model} "
                     f"content={(res.content or '').strip()[:40]!r}")
    except Exception as exc:
        lines.append(f"  LIVE CALL -> FAILED: {type(exc).__name__}: {str(exc)[:180]}")
    return lines


# --- runner ---------------------------------------------------------------
def run(live: bool = False, json_out: str | None = None) -> int:
    results = []
    for tier, cid, points, fn in CHECKS:
        tmp = Path(tempfile.mkdtemp(prefix=f"bench_{cid}_"))
        try:
            outcome, note = fn(tmp)
        except Exception as exc:  # a crash scores zero, never aborts the run
            outcome, note = False, f"EXCEPTION {type(exc).__name__}: {str(exc)[:90]}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        earned = points * (1.0 if outcome is True else (0.5 if outcome == 0.5 else 0.0))
        results.append({"tier": tier, "id": cid, "points": points, "earned": earned, "note": note})

    total = sum(r["earned"] for r in results)
    possible = sum(r["points"] for r in results)

    print("\nMemAssist benchmark — deterministic suite\n" + "=" * 62)
    for tier in sorted(TIER_NAMES):
        rows = [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        got, cap = sum(r["earned"] for r in rows), sum(r["points"] for r in rows)
        print(f"\n{tier} {TIER_NAMES[tier]} — {got:g}/{cap}")
        for r in rows:
            mark = "PASS" if r["earned"] == r["points"] else ("PART" if r["earned"] else "FAIL")
            print(f"  [{mark}] {r['id']:<5} {r['earned']:g}/{r['points']}  {r['note']}")

    print("\n" + "=" * 62)
    print(f"TOTAL: {total:g} / {possible}")

    if live:
        print("\nLive provider smoke (not scored)\n" + "-" * 62)
        for line in run_live_smoke():
            print(line)

    if json_out:
        Path(json_out).write_text(
            json.dumps({"total": total, "possible": possible, "checks": results}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="bench")
    ap.add_argument("--live", action="store_true", help="also run a real-provider smoke test")
    ap.add_argument("--json", dest="json_out", default=None, help="write results to a JSON file")
    args = ap.parse_args()
    raise SystemExit(run(live=args.live, json_out=args.json_out))
