"""The privacy gate's detector (security/sensitivity.py).

This is the only thing standing between the recall log and a provider that
trains on prompts, so it gets CI coverage rather than only a __main__ check.
"""

from __future__ import annotations

import pytest

from security import sensitivity


def test_module_self_check_passes():
    """Runs the module's own corpus, so the two never drift apart."""
    sensitivity.demo()


@pytest.mark.parametrize(
    "text, category",
    [
        ("my key is sk-abcdefghijklmnopqrstuvwx", "api-key"),
        ("AIzaSyD-1234567890abcdefghijklmnopqrst", "google-api-key"),
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer-token"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private-key"),
        ("AKIAIOSFODNN7EXAMPLE", "aws-access-key"),
        ("password: hunter2", "credential-assignment"),
        ("ssn 123-45-6789", "ssn"),
        ("please keep this confidential", "user-marked"),
    ],
)
def test_each_category_fires(text, category):
    assert category in sensitivity.classify(text)


@pytest.mark.parametrize(
    "text",
    [
        "4111111111111111",              # unspaced — the form that leaked
        "my visa is 4539578763621486",
        "card 4111 1111 1111 1111",
        "4111-1111-1111-1111",
    ],
)
def test_card_numbers_are_caught_in_every_written_form(text):
    """Regression: the Luhn gate stripped the digits instead of the separators,
    so it rejected every card. The spaced form still matched `aadhaar`, which
    hid it — the unspaced form matched nothing and would have gone outbound."""
    assert "card-number" in sensitivity.classify(text)


@pytest.mark.parametrize(
    "text",
    [
        "order number 1234567890123456",   # 16 digits, fails Luhn
        "call me on 9876543210",
        "I forgot my password again",      # no value attached
        "we rotate the API key quarterly",
        "The user's daughter Mira was born in Pune in 2019.",
        "",
    ],
)
def test_ordinary_content_is_not_flagged(text):
    """False positives cost a skipped message; the gate must still let normal
    conversation through or consolidation would never send anything."""
    assert sensitivity.classify(text) == []
