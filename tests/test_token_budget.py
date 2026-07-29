"""Tests for the pure token-budget helpers."""

from __future__ import annotations

from agent.token_budget import (
    approx_tokens,
    format_usage,
    is_under_pressure,
    usage_fraction,
)


def test_usage_fraction():
    assert usage_fraction(50, 100) == 0.5
    assert usage_fraction(0, 100) == 0.0
    assert usage_fraction(10, 0) == 0.0  # no divide-by-zero


def test_is_under_pressure():
    assert is_under_pressure(70, 100, 0.7) is True
    assert is_under_pressure(69, 100, 0.7) is False
    assert is_under_pressure(100, 100, 0.7) is True


def test_format_usage():
    s = format_usage(1234, 32000)
    assert "1,234" in s and "32,000" in s and "%" in s


def test_approx_tokens():
    assert approx_tokens("a" * 40) == 10
    assert approx_tokens("") == 1  # floors at 1
