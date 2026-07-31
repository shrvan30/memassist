"""Deterministic sensitivity detection — the privacy gate on the Mistral lane.

Why this exists (README defect #5, spec §11 P5): the background consolidation
lane routes to Mistral, whose free tier trains on prompts. Everything else in
this codebase stays local or goes to a provider the user chose for *that turn*;
a batch job is different, because it sends content the user is not watching go
out. So the rule is not "summarize the conversation" but "summarize the part of
the conversation that is safe to hand to a model that learns from it".

Three properties this deliberately has:

- **Deterministic, no LLM.** Same rule as every other memory function
  (CLAUDE.md): asking a model whether something is sensitive means sending it
  the thing first, which is the exact disclosure being prevented.
- **Recall over precision.** A false positive costs one skipped message in a
  batch job nobody is waiting on. A false negative is a credential in someone
  else's training set. The asymmetry is not close, so borderline patterns are
  included.
- **Substring-anchored, not semantic.** It catches secrets and identifiers,
  which have shape. It does NOT try to judge whether prose is "private" — that
  is not a thing a regex knows, and pretending otherwise would be the dangerous
  kind of reassurance.

Used by ``jobs/consolidate.py`` to filter the outbound payload, and by the
archival stores to stamp the ``sensitive`` column at insert time.
"""

from __future__ import annotations

import re

# (name, pattern). Names appear in the job's audit log, so they are the
# explanation of WHY something was withheld.
SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Provider keys and generic long tokens. sk-/gsk_/AIza cover the four
    # providers this project itself uses, which is the likeliest thing a user
    # ever pastes into it.
    ("api-key", re.compile(r"\b(?:sk|gsk|pk|rk)[-_][A-Za-z0-9_-]{16,}", re.I)),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}")),
    ("bearer-token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}", re.I)),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+")),
    ("private-key", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # A named secret with a value attached: `password: hunter2`, `api_key=...`.
    # The value is required, so prose *about* passwords is not flagged.
    (
        "credential-assignment",
        re.compile(
            r"\b(?:pass(?:word|wd|phrase)|secret|api[_-]?key|access[_-]?token|"
            r"auth[_-]?token|credential)s?\b\s*[:=]\s*\S+",
            re.I,
        ),
    ),
    # Identifiers. Card numbers are Luhn-checked below rather than by regex.
    ("card-number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("aadhaar", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")),
    # The user asking for it directly always wins over any heuristic here.
    ("user-marked", re.compile(r"\b(?:do not (?:share|send|store)|confidential|"
                               r"private note|between us|off the record)\b", re.I)),
)

_NON_DIGITS = re.compile(r"\D")


def _luhn_ok(digits: str) -> bool:
    """Standard checksum. Without it every 13-19 digit run — order numbers,
    timestamps concatenated, phone strings — reads as a card number."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def classify(text: str) -> list[str]:
    """Every sensitivity category present in ``text``. Empty means safe to send."""
    if not text:
        return []
    hits: list[str] = []
    for name, pattern in SENSITIVE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        if name == "card-number":
            # Luhn-gate this one only; the rest are shape-unique enough.
            # Strip the NON-digits. Stripping the digits leaves the separators,
            # whose length never reaches 13, so the gate rejected every card and
            # the rule was dead — masked in testing because the spaced form
            # `4111 1111 1111 1111` also matches `aadhaar`, while the unspaced
            # form matched nothing at all and would have gone outbound.
            digits = _NON_DIGITS.sub("", match.group(0))
            if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
                continue
        hits.append(name)
    return hits


def is_sensitive(text: str) -> bool:
    return bool(classify(text))


def demo() -> None:
    """Runnable check: the categories that must fire, and the prose that must not."""
    must_flag = [
        "my key is sk-abcdefghijklmnopqrstuvwx",
        "AIzaSyD-1234567890abcdefghijklmnopqrst",
        "password: hunter2",
        "API_KEY=xoxb-not-a-real-token",
        "card 4111 1111 1111 1111",
        # Unspaced too: the spaced form also matches `aadhaar`, so testing only
        # that one let a dead card-number rule look alive.
        "card 4111111111111111",
        "my visa is 4539578763621486",
        "ssn 123-45-6789",
        "-----BEGIN RSA PRIVATE KEY-----",
        "AKIAIOSFODNN7EXAMPLE",
        "keep this confidential please",
        "Aadhaar 1234 5678 9012",
    ]
    must_not_flag = [
        "I forgot my password again",           # no value attached
        "We discussed the API key rotation policy",
        "order number 1234567890123456",        # 16 digits, fails Luhn
        "The user's daughter Mira was born in Pune in 2019.",
        "He drives a red Toyota Corolla.",
        "",
    ]
    for t in must_flag:
        assert is_sensitive(t), f"MISSED: {t!r}"
    for t in must_not_flag:
        assert not is_sensitive(t), f"FALSE POSITIVE: {t!r} -> {classify(t)}"
    print(f"sensitivity: {len(must_flag)} flagged, {len(must_not_flag)} clean - OK")


if __name__ == "__main__":
    demo()
