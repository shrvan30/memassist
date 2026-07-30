"""Pure token-budget math for the memory-pressure signal.

No API or LLM calls live here so the logic is deterministic and unit-tested. The
agent loop feeds in the *actual* input-token count from ``response.usage`` each
turn; these helpers just interpret it against the (virtual) context budget.
"""

from __future__ import annotations


def approx_tokens(text: str) -> int:
    """Very rough character-based token estimate (~4 chars/token).

    Used only for a pre-first-response sidebar preview; real counts come from
    the provider's response usage.
    """
    return max(1, len(text) // 4)


def usage_fraction(input_tokens: int, limit: int) -> float:
    if limit <= 0:
        return 0.0
    return input_tokens / limit


def is_under_pressure(input_tokens: int, limit: int, threshold: float) -> bool:
    """True when context usage has crossed the pressure threshold."""
    return usage_fraction(input_tokens, limit) >= threshold


def format_usage(input_tokens: int, limit: int) -> str:
    pct = round(usage_fraction(input_tokens, limit) * 100)
    return f"{input_tokens:,} / {limit:,} tokens ({pct}%)"
