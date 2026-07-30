"""Router exceptions + classification of provider-call failures.

The classifier maps SDK/transport exceptions to router-level categories that
drive the failover algorithm:

- 429  -> RateLimitError       (short cooldown, then next provider)
- 402  -> QuotaExceededError   (cooldown until UTC midnight, then next provider)
- 5xx / connection / timeout -> ProviderServerError (retry once, then next)
- 401/403/404/model-not-found/zero-quota -> ProviderPermanentError
  (NEVER cooled down — a retry timer would hide a config bug; see below)
- other (e.g. 400) -> propagated unchanged, so request bugs surface instead of
  being silently masked by a failover to every provider.

Why zero-quota 429s are permanent: some providers (e.g. Google) return HTTP 429
both for "you are going too fast" and for "this model has no free-tier allowance
on your project" (``limit: 0``). Treating the second as a transient rate limit
puts the provider on a rolling 60s cooldown forever, so it serves zero requests
and never once reports why. The two must be distinguished here, at classification
time.
"""

from __future__ import annotations


class RouterError(Exception):
    """Base class for router-level errors."""


class ProviderConfigError(RouterError):
    """Configuration problem (e.g. no providers for a lane)."""


class AllProvidersExhausted(RouterError):
    """Every provider in the chain was unavailable or failed this turn."""

    def __init__(self, attempts: dict[str, str]) -> None:
        self.attempts = dict(attempts)
        detail = ", ".join(f"{name}={reason}" for name, reason in self.attempts.items())
        super().__init__(
            f"All providers exhausted or unavailable ({detail or 'no providers tried'})."
        )


class ProviderError(Exception):
    """A classified failure from a single provider call."""

    def __init__(self, provider: str, message: str, *, status_code: int | None = None) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"[{provider}] {message}")


class RateLimitError(ProviderError):
    """HTTP 429 — rate/RPM limit hit."""


class QuotaExceededError(ProviderError):
    """HTTP 402 — daily/quota budget exhausted."""


class ProviderServerError(ProviderError):
    """HTTP 5xx or a transport error — retryable, then fail over."""


class ProviderPermanentError(ProviderError):
    """The provider cannot serve until a human changes configuration.

    Deliberately NOT cooled down. A cooldown implies "try again later", which is
    false here and hides the real fault behind a retry timer.
    """


class ProviderAuthError(ProviderPermanentError):
    """HTTP 401/403 — bad or missing credentials; skip this provider."""


_TRANSPORT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "APIConnectionTimeoutError",
    "InternalServerError",
}

# Transient 400s that mean "this provider's model flubbed the tool call", not
# "the request is malformed" — safe to fail over.
_TRANSIENT_400_MARKERS = ("tool_use_failed", "failed to call a function")

# A 429 carrying one of these means the *allowance itself* is zero, not that we
# are going too fast. Google phrases it "limit: 0" in the quota-metric lines.
_ZERO_QUOTA_MARKERS = ("limit: 0", "limit:0", "quota_limit_value: 0")

# The model slug is wrong, retired, or not enabled for this key. Retrying and
# failing over are both pointless until providers.yaml is corrected.
_MODEL_NOT_FOUND_MARKERS = (
    "model_not_found",
    "model not found",
    "is not found",
    "does not exist",
    "unsupported model",
    "unknown model",
)


def _mentions(exc: Exception, markers: tuple[str, ...]) -> bool:
    code = str(getattr(exc, "code", "") or "")
    haystack = f"{code} {exc}".lower()
    return any(marker in haystack for marker in markers)


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _TRANSPORT_ERROR_NAMES


def classify_provider_error(provider: str, exc: Exception) -> ProviderError:
    """Return a router ``ProviderError`` for a failover-able exception.

    Re-raises the original exception for cases that must NOT trigger failover
    (unexpected 4xx like 400/422 — usually a bug in our request). Callers use it
    as ``perr = classify_provider_error(...)`` inside an ``except`` block: if it
    returns, dispatch on ``perr``'s type; if it raises, the error propagates.
    """
    status = getattr(exc, "status_code", None)
    if status == 429:
        # "limit: 0" -> the free tier grants this model nothing. Permanent.
        if _mentions(exc, _ZERO_QUOTA_MARKERS):
            return ProviderPermanentError(
                provider,
                "zero free-tier quota for this model (HTTP 429, limit: 0) — "
                "change the model slug in providers.yaml or enable billing",
                status_code=429,
            )
        return RateLimitError(provider, "rate limited (HTTP 429)", status_code=429)
    if status == 402:
        return QuotaExceededError(provider, "quota exceeded (HTTP 402)", status_code=402)
    if status in (401, 403):
        return ProviderAuthError(provider, f"auth error (HTTP {status})", status_code=status)
    if status == 404 or _mentions(exc, _MODEL_NOT_FOUND_MARKERS):
        return ProviderPermanentError(
            provider,
            f"model '{getattr(exc, 'model', '') or 'configured slug'}' unavailable "
            f"(HTTP {status}) — check providers.yaml",
            status_code=status,
        )
    if isinstance(status, int) and status >= 500:
        return ProviderServerError(provider, f"server error (HTTP {status})", status_code=status)
    if isinstance(status, int):
        # Some providers surface a *transient* model-side failure as a 400 — e.g.
        # Groq/Llama "tool_use_failed" when the model emits a malformed tool call.
        # The request itself is valid (it works on other providers), so fail over
        # instead of treating it as a fatal request bug.
        code = str(getattr(exc, "code", "") or "")
        haystack = f"{code} {exc}".lower()
        if any(marker in haystack for marker in _TRANSIENT_400_MARKERS):
            return ProviderServerError(
                provider, f"transient tool-call failure ({code or 'HTTP 400'})", status_code=status
            )
        # Any other 4xx: a malformed request that would fail on every provider.
        raise exc
    # No HTTP status → connection/timeout are worth retrying/failing over; the
    # rest (programming errors) propagate.
    if _is_transport_error(exc):
        return ProviderServerError(provider, f"transport error: {type(exc).__name__}")
    raise exc
