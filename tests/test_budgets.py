"""Tests for the persistent per-provider usage ledger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from llm.budgets import BudgetLedger


class Clock:
    """Controllable UTC clock for deterministic time-based tests."""

    def __init__(self, dt: datetime) -> None:
        self.dt = dt

    def __call__(self) -> datetime:
        return self.dt

    def advance(self, seconds: float) -> None:
        self.dt = self.dt + timedelta(seconds=seconds)


def _noon() -> datetime:
    return datetime(2026, 7, 23, 12, 0, 0, tzinfo=timezone.utc)


def test_record_and_get_usage():
    led = BudgetLedger(":memory:")
    assert led.get_usage("gemini") == (0, 0)
    led.record_request("gemini", tokens=100)
    led.record_request("gemini", tokens=50)
    assert led.get_usage("gemini") == (2, 150)


def test_cooldown_for_seconds_expires():
    clock = Clock(_noon())
    led = BudgetLedger(":memory:", now_fn=clock)
    led.cooldown_for_seconds("groq", 60)
    assert led.is_cooling("groq")
    assert led.cooldown_remaining("groq") == 60.0
    clock.advance(61)
    assert not led.is_cooling("groq")
    assert led.cooldown_remaining("groq") == 0.0


def test_cooldown_until_utc_midnight():
    clock = Clock(_noon())  # 12:00 UTC -> next midnight is 12h away
    led = BudgetLedger(":memory:", now_fn=clock)
    led.cooldown_until_utc_midnight("gemini")
    assert led.is_cooling("gemini")
    assert led.cooldown_remaining("gemini") == 12 * 3600


def test_daily_usage_resets_on_new_utc_day():
    clock = Clock(_noon())
    led = BudgetLedger(":memory:", now_fn=clock)
    led.record_request("gemini", tokens=100)
    assert led.get_usage("gemini") == (1, 100)
    clock.advance(24 * 3600)  # next day, same time
    assert led.get_usage("gemini") == (0, 0)


def test_midnight_cooldown_clears_after_rollover():
    clock = Clock(_noon())
    led = BudgetLedger(":memory:", now_fn=clock)
    led.cooldown_until_utc_midnight("gemini")
    assert led.is_cooling("gemini")
    clock.advance(24 * 3600)
    assert not led.is_cooling("gemini")


def test_snapshot():
    led = BudgetLedger(":memory:")
    led.record_request("gemini", tokens=10)
    led.record_request("groq", tokens=5)
    snap = led.snapshot()
    assert snap["gemini"] == {"requests": 1, "tokens": 10, "cooldown_remaining": 0.0}
    assert snap["groq"]["tokens"] == 5
