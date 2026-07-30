"""Langfuse tracing — one trace per turn (spec §11 P5).

Everything this project does to a turn is already recorded somewhere: the
router stamps ``served_by``, the guards log their refusals, the sanitizer counts
its hits, the recall log keeps the lot. What was missing is a place to see one
turn end to end — which node was slow, which provider served it, whether the
security layer fired — without reading a database by hand.

**Why Langfuse Cloud's free tier, not self-hosted.** Self-hosting Langfuse v3+
means Postgres *and* ClickHouse *and* Redis *and* MinIO. This project's whole
premise is $0/month on free tiers, and it already asks the reader to run four
containers; adding four more to watch the first four is the tail wagging the
dog. The cloud free tier costs nothing, needs no infrastructure, and the
trade-off it asks for — conversation content leaving the machine — is one this
codebase is unusually well equipped to handle, because the T10 privacy gate
already exists. ``mask`` below routes every trace payload through the SAME
detector, so a credential that may not go to Mistral may not go to Langfuse
either. Anyone who prefers self-hosting sets ``LANGFUSE_HOST`` and changes
nothing else.

**Disabled is the default and must stay cheap.** With no keys set, every
function here is a no-op: no client, no threads, no network, no import cost
beyond this module. The benchmark's determinism and the test suite's
offline-ness are not negotiable, and neither may depend on remembering to turn
tracing off.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any

from security.sensitivity import classify

_log = logging.getLogger(__name__)

# Langfuse's own env var names — no MEMASSIST_ prefix, so an existing Langfuse
# setup works here unchanged.
PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
HOST_ENV = "LANGFUSE_HOST"

_REDACTED = "[redacted: {}]"

_client: Any = None
_resolved = False


def _mask(*, data: Any, **_kwargs: Any) -> Any:
    """Redact anything the privacy gate would withhold, at any nesting depth.

    Same detector as ``jobs/consolidate.py``. Two destinations, one rule: if a
    string is not safe to hand to a model that trains on it, it is not safe to
    park in a hosted dashboard either.
    """
    if isinstance(data, str):
        categories = classify(data)
        return _REDACTED.format(",".join(categories)) if categories else data
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask(data=v) for v in data]
    return data


def _resolve() -> Any:
    """Build the client once. A failure disables tracing; it never raises.

    Observability that can break the thing it observes is a liability, so every
    failure mode here — missing package, bad keys, unreachable host — degrades
    to "no traces" rather than to "no assistant".
    """
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    if not (os.getenv(PUBLIC_KEY_ENV) and os.getenv(SECRET_KEY_ENV)):
        return None
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=os.getenv(PUBLIC_KEY_ENV),
            secret_key=os.getenv(SECRET_KEY_ENV),
            host=os.getenv(HOST_ENV) or None,
            mask=_mask,
        )
        _log.info("Langfuse tracing enabled (host=%s)", os.getenv(HOST_ENV) or "cloud")
    except Exception as exc:  # noqa: BLE001 - tracing must never break a turn
        _log.warning("Langfuse tracing unavailable, continuing without it: %s", exc)
        _client = None
    return _client


def enabled() -> bool:
    return _resolve() is not None


def reset_for_tests() -> None:
    """Forget the resolved client so a test can toggle the env vars."""
    global _client, _resolved
    _client, _resolved = None, False


@contextmanager
def span(name: str, *, as_type: str = "span", **kwargs: Any):
    """A traced span, or a no-op context manager when tracing is off."""
    client = _resolve()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as s:
            yield s
    except Exception as exc:  # noqa: BLE001
        _log.debug("Langfuse span %r failed: %s", name, exc)
        yield None


def event(name: str, **kwargs: Any) -> None:
    """A point-in-time event on the current span (a guard denial, an interrupt)."""
    client = _resolve()
    if client is None:
        return
    try:
        client.create_event(name=name, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.debug("Langfuse event %r failed: %s", name, exc)


def update_trace(span: Any, *, session_id: str | None = None, tags: Any = None) -> None:
    """Stamp trace-level fields on the turn's root span.

    Set as OpenTelemetry attributes deliberately. Langfuse v4 has no
    ``update_current_trace``, and ``span.update(session_id=...)`` accepts the
    keyword into ``**kwargs`` and then silently drops it — so the obvious call
    raises nothing, writes nothing, and looks like it worked. The attribute
    names in ``LangfuseOtelSpanAttributes`` are the actual contract.
    """
    if span is None or _resolve() is None:
        return
    try:
        import json

        from langfuse import LangfuseOtelSpanAttributes as attrs

        otel = span._otel_span
        if session_id:
            otel.set_attribute(attrs.TRACE_SESSION_ID, session_id)
        if tags:
            otel.set_attribute(attrs.TRACE_TAGS, json.dumps(list(tags)))
    except Exception as exc:  # noqa: BLE001
        _log.debug("Langfuse trace update failed: %s", exc)


def flush() -> None:
    client = _resolve()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # noqa: BLE001
            _log.debug("Langfuse flush failed: %s", exc)


def shutdown() -> None:
    client = _resolve()
    if client is not None:
        try:
            client.shutdown()
        except Exception as exc:  # noqa: BLE001
            _log.debug("Langfuse shutdown failed: %s", exc)


# --- what a turn reports --------------------------------------------------
def summarize_turn(state: dict) -> dict:
    """The security- and routing-relevant facts of one turn, for trace metadata.

    Deliberately counts and names rather than content: the trace should let you
    see THAT the sanitizer fired and on what tool, then send you to the recall
    log for the bytes. The recall log is local; the trace is not.
    """
    return {
        "served_by": state.get("served_by"),
        "heartbeats": state.get("heartbeat_count", 0),
        "input_tokens": state.get("input_tokens", 0),
        "limit": state.get("limit", 0),
        "saw_untrusted": bool(state.get("saw_untrusted")),
        "injection_flags": list(state.get("injection_flags") or []),
        "blocked_tools": list(state.get("blocked_tools") or []),
        "interrupted": bool(state.get("gated_action")),
    }


def traced_node(fn, name: str):
    """Wrap a graph node so each one becomes a span. Free when tracing is off.

    ``nullcontext`` rather than :func:`span` on the disabled path: this runs on
    every node of every heartbeat of every turn, including in the benchmark, so
    the off-switch has to be a branch and not a generator.
    """

    def wrapped(state, *args, **kwargs):
        if _resolve() is None:
            return fn(state, *args, **kwargs)
        with span(name, as_type="chain") as s:
            out = fn(state, *args, **kwargs)
            if s is not None and isinstance(out, dict):
                try:
                    s.update(output=_node_output(name, out))
                except Exception as exc:  # noqa: BLE001
                    _log.debug("Langfuse node output failed: %s", exc)
            return out

    wrapped.__name__ = getattr(fn, "__name__", name)
    return wrapped


# Per-node output projections. The full state carries the entire message
# history, which would make every span enormous and largely identical.
_NODE_KEYS = {
    "build_prompt": ("context_pct",),
    "pressure_check": ("needs_warning", "context_pct"),
    "call_llm": ("served_by", "input_tokens", "limit"),
    "security_gate": ("blocked_tools", "saw_untrusted"),
    "dispatch_tools": ("heartbeat_count", "done"),
    "sanitize_results": ("saw_untrusted", "injection_flags"),
    "respond": ("done",),
}


def _node_output(name: str, out: dict) -> dict:
    keys = _NODE_KEYS.get(name)
    if keys is None:
        return {"keys": sorted(out)}
    return {k: out[k] for k in keys if k in out}


__all__ = [
    "enabled",
    "event",
    "flush",
    "nullcontext",
    "reset_for_tests",
    "shutdown",
    "span",
    "summarize_turn",
    "traced_node",
    "update_trace",
]
