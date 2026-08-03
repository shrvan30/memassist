"""Tests for the free-tier failover router — all offline via fake clients."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from llm import errors
from llm.budgets import BudgetLedger
from llm.router import (
    REQUEST_TIMEOUT_SECONDS,
    OpenAIChatClient,
    ProviderConfig,
    Router,
    ToolCall,
    _request_timeout,
    load_providers,
    resolve_api_key,
)


# --- helpers --------------------------------------------------------------
def make_cfg(name: str, priority: int, **over) -> ProviderConfig:
    base = dict(
        model=f"{name}-model",
        base_url=f"https://{name}.example",
        api_key_env=f"{name.upper()}_API_KEY",
        context_window=100_000,
        lanes=("interactive",),
    )
    base.update(over)
    return ProviderConfig(name=name, priority=priority, **base)


def fake_response(content="hello", tool_calls=None, model="m", total=15, finish="stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=total)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


class FakeAPIError(Exception):
    """Mimics an openai APIStatusError (has a .status_code)."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class ToolUseFailedError(Exception):
    """Mimics Groq/Llama's transient 400 'tool_use_failed'."""

    def __init__(self) -> None:
        self.status_code = 400
        self.code = "tool_use_failed"
        super().__init__("Failed to call a function. See failed_generation. tool_use_failed")


class FakeClient:
    """Yields scripted behaviors (responses or exceptions), then repeats default."""

    def __init__(self, behaviors=(), default=None) -> None:
        self.behaviors = list(behaviors)
        self.default = default
        self.calls = 0
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        item = self.behaviors.pop(0) if self.behaviors else self.default
        if isinstance(item, Exception):
            raise item
        if item is None:
            raise AssertionError("FakeClient had no behavior/default for this call")
        return item


def build_router(providers, clients, *, api_keys=None, ledger=None) -> Router:
    ledger = ledger or BudgetLedger(":memory:")
    keys = api_keys if api_keys is not None else {c.name: f"key-{c.name}" for c in providers}

    def factory(cfg, api_key):
        return clients[cfg.name]

    return Router(
        providers,
        ledger,
        client_factory=factory,
        api_keys=keys,
        sleep_fn=lambda s: None,
        rng=random.Random(0),
    )


MSG = [{"role": "user", "content": "hi"}]


# --- happy path -----------------------------------------------------------
def test_primary_success_records_usage_and_tags():
    client = FakeClient([fake_response(content="hi", total=20)])
    r = build_router([make_cfg("gemini", 1)], {"gemini": client})

    res = r.chat(MSG)

    assert res.served_by == "gemini"
    assert res.content == "hi"
    assert res.usage.total_tokens == 20
    assert r.ledger.get_usage("gemini") == (1, 20)
    assert client.last_kwargs["temperature"] == r.temperature


def test_usage_accumulates_across_calls():
    client = FakeClient([fake_response(total=10), fake_response(total=15)])
    r = build_router([make_cfg("gemini", 1)], {"gemini": client})
    r.chat(MSG)
    r.chat(MSG)
    assert r.ledger.get_usage("gemini") == (2, 25)


# --- failover -------------------------------------------------------------
def test_failover_on_rate_limit():
    providers = [make_cfg("gemini", 1), make_cfg("groq", 2)]
    clients = {
        "gemini": FakeClient([FakeAPIError(429)]),
        "groq": FakeClient([fake_response(content="from groq", total=12)]),
    }
    r = build_router(providers, clients)

    res = r.chat(MSG)

    assert res.served_by == "groq"
    assert r.ledger.is_cooling("gemini")  # 429 -> 60s cooldown
    assert r.ledger.cooldown_remaining("gemini") > 59
    assert r.ledger.get_usage("gemini") == (0, 0)  # failed call is not counted
    assert r.ledger.get_usage("groq") == (1, 12)


def test_quota_402_cools_until_midnight():
    ledger = BudgetLedger(":memory:", now_fn=lambda: datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc))
    providers = [make_cfg("gemini", 1), make_cfg("groq", 2)]
    clients = {
        "gemini": FakeClient([FakeAPIError(402)]),
        "groq": FakeClient([fake_response(content="groq")]),
    }
    r = build_router(providers, clients, ledger=ledger)

    res = r.chat(MSG)

    assert res.served_by == "groq"
    assert r.ledger.cooldown_remaining("gemini") == 12 * 3600  # noon -> midnight


def test_server_error_retries_once_then_fails_over():
    providers = [make_cfg("gemini", 1), make_cfg("groq", 2)]
    gemini = FakeClient(default=FakeAPIError(500))  # 500 on every call
    groq = FakeClient([fake_response(content="groq")])
    r = build_router(providers, {"gemini": gemini, "groq": groq})

    res = r.chat(MSG)

    assert res.served_by == "groq"
    assert gemini.calls == 2  # initial call + one retry (server_retries=1)


def test_bad_request_400_propagates_without_failover():
    providers = [make_cfg("gemini", 1), make_cfg("groq", 2)]
    groq = FakeClient([fake_response()])
    r = build_router(providers, {"gemini": FakeClient([FakeAPIError(400)]), "groq": groq})

    with pytest.raises(FakeAPIError):  # a request bug must surface, not fail over
        r.chat(MSG)
    assert groq.calls == 0


def test_transient_tool_use_failure_fails_over():
    # Groq's 400 tool_use_failed is a flaky model failure, not a request bug —
    # the router must fail over (after one retry), not crash the turn.
    providers = [make_cfg("groq", 1), make_cfg("mistral", 2)]
    groq = FakeClient(default=ToolUseFailedError())
    mistral = FakeClient([fake_response(content="from mistral")])
    r = build_router(providers, {"groq": groq, "mistral": mistral})

    res = r.chat(MSG)

    assert res.served_by == "mistral"
    assert groq.calls == 2  # initial + one retry, then fail over


def test_all_providers_exhausted_raises():
    r = build_router([make_cfg("gemini", 1)], {"gemini": FakeClient([FakeAPIError(429)])})
    with pytest.raises(errors.AllProvidersExhausted) as excinfo:
        r.chat(MSG)
    assert excinfo.value.attempts.get("gemini") == "rate_limited"


# --- proactive skipping ---------------------------------------------------
def test_proactive_skip_when_cooling():
    ledger = BudgetLedger(":memory:")
    ledger.cooldown_for_seconds("gemini", 60)
    gemini = FakeClient([fake_response(content="should not run")])
    groq = FakeClient([fake_response(content="groq")])
    r = build_router([make_cfg("gemini", 1), make_cfg("groq", 2)], {"gemini": gemini, "groq": groq}, ledger=ledger)

    res = r.chat(MSG)

    assert res.served_by == "groq"
    assert gemini.calls == 0  # skipped without spending a request


def test_proactive_skip_on_daily_request_budget():
    ledger = BudgetLedger(":memory:")
    ledger.record_request("gemini")
    ledger.record_request("gemini")  # now at the limit of 2
    providers = [make_cfg("gemini", 1, daily_request_limit=2), make_cfg("groq", 2)]
    gemini = FakeClient([fake_response()])
    groq = FakeClient([fake_response(content="groq")])
    r = build_router(providers, {"gemini": gemini, "groq": groq}, ledger=ledger)

    res = r.chat(MSG)

    assert res.served_by == "groq"
    assert gemini.calls == 0


def test_missing_api_key_skips_provider():
    gemini = FakeClient([fake_response()])
    groq = FakeClient([fake_response(content="groq")])
    r = build_router(
        [make_cfg("gemini", 1), make_cfg("groq", 2)],
        {"gemini": gemini, "groq": groq},
        api_keys={"gemini": None, "groq": "k"},
    )

    res = r.chat(MSG)

    assert res.served_by == "groq"
    assert gemini.calls == 0


# --- api key resolution ---------------------------------------------------
def test_gemini_key_prefers_canonical_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "new")
    monkeypatch.setenv("GOOGLE_API_KEY", "old")
    assert resolve_api_key("GEMINI_API_KEY") == "new"


def test_legacy_google_key_still_works_and_warns(monkeypatch, caplog):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "old")
    with caplog.at_level("WARNING"):
        assert resolve_api_key("GEMINI_API_KEY") == "old"
    assert "GOOGLE_API_KEY is deprecated" in caplog.text


def test_unknown_env_has_no_legacy_fallback(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "old")
    assert resolve_api_key("GROQ_API_KEY") is None


# --- tool calls + lanes ---------------------------------------------------
def test_tool_calls_are_normalized_and_tools_forwarded():
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="core_memory_append",
            arguments='{"block":"human","content":"Name: Zed"}',
        ),
    )
    client = FakeClient([fake_response(content=None, tool_calls=[tc])])
    r = build_router([make_cfg("gemini", 1)], {"gemini": client})

    tools = [{"type": "function", "function": {"name": "core_memory_append"}}]
    res = r.chat(MSG, tools=tools)

    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "core_memory_append"
    assert call.parsed_arguments() == {"block": "human", "content": "Name: Zed"}
    assert client.last_kwargs["tools"] == tools
    assert client.last_kwargs["tool_choice"] == "auto"


def test_no_tools_are_omitted_from_call():
    client = FakeClient([fake_response()])
    r = build_router([make_cfg("gemini", 1)], {"gemini": client})
    r.chat(MSG)  # tools=None
    assert "tools" not in client.last_kwargs
    assert "tool_choice" not in client.last_kwargs


def test_background_lane_routes_to_mistral_only():
    providers = [
        make_cfg("gemini", 1, lanes=("interactive",)),
        make_cfg("mistral", 4, lanes=("interactive", "background")),
    ]
    gemini = FakeClient([fake_response(content="gem")])
    mistral = FakeClient([fake_response(content="from mistral", total=7)])
    r = build_router(providers, {"gemini": gemini, "mistral": mistral})

    res = r.chat_background(MSG)

    assert res.served_by == "mistral"
    assert gemini.calls == 0
    assert r.ledger.get_usage("mistral") == (1, 7)


# --- introspection + config loading --------------------------------------
def test_provider_status_reports_availability():
    ledger = BudgetLedger(":memory:")
    ledger.cooldown_for_seconds("gemini", 30)
    r = build_router(
        [make_cfg("gemini", 1), make_cfg("groq", 2)],
        {"gemini": FakeClient(), "groq": FakeClient()},
        ledger=ledger,
    )
    status = {s["name"]: s for s in r.provider_status()}
    assert status["gemini"]["available"] is False
    assert status["gemini"]["reason"] == "cooling_down"
    assert status["groq"]["available"] is True


def test_transport_leaves_all_retrying_to_the_router(monkeypatch):
    """The SDK must not retry underneath us, and must not hang for ten minutes.

    Its own 429 retries — two, honouring ``retry-after`` — turned a failover
    that should take milliseconds into ~17s of inline sleeping per rate-limited
    provider, because the router never got the error until the SDK gave up.

    Constructing the client is offline: ``openai.OpenAI()`` opens no connection,
    so this needs no key and no ``llm`` marker. The class is imported at module
    scope, before conftest's autouse guard rebinds the name.
    """
    client = OpenAIChatClient("https://example.invalid/v1", "sk-not-a-real-key")
    assert client._client.max_retries == 0
    assert client._client.timeout == REQUEST_TIMEOUT_SECONDS

    # The default is asserted through the function, not the constant: the
    # constant reflects whatever MEMASSIST_REQUEST_TIMEOUT was set to in the
    # process running the suite, and a test that passes only on an unset
    # environment is the same defect as reading a DSN out of the ambient shell.
    monkeypatch.delenv("MEMASSIST_REQUEST_TIMEOUT", raising=False)
    assert _request_timeout() == 30.0
    monkeypatch.setenv("MEMASSIST_REQUEST_TIMEOUT", "90")
    assert _request_timeout() == 90.0


def test_load_providers_from_default_yaml():
    providers = load_providers()  # ships with the repo
    assert [p.name for p in providers] == ["gemini", "groq", "openrouter", "mistral"]
    assert providers[0].priority == 1
    assert providers[0].context_window == 1_048_576
    mistral = next(p for p in providers if p.name == "mistral")
    assert "background" in mistral.lanes
