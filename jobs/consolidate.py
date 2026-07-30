"""T10 — the Mistral consolidation lane: recall -> archival summarization.

Old conversation sits in the recall log forever, searchable only by keyword. The
useful residue of it — the facts — belongs in archival, where semantic search can
find them. That summarization is batch work with no user waiting on it, so it
runs on the background lane: ``router.chat_background()``, which goes to Mistral
and nowhere else (``lanes: [background]`` in providers.yaml). Mistral's 2 RPM is
irrelevant to a job that runs nightly.

**The privacy gate is the point of this module, not a feature of it.** Mistral's
free tier trains on prompts, so this job sends content the user is not watching
go out, to a model that keeps it. README defect #5 called the ``sensitive`` flag
a hard prerequisite and said not to wire this lane until it existed. It exists
now, and the filter here is what makes it mean something.

Four independent exclusions, each of which alone would be enough for its class,
kept together because they fail differently:

1. **Only ``event_type='message'``.** Structural, and the strongest: every
   external tool result is recorded as ``tool_result`` (verbatim, for audit —
   spec §6.2), so this single filter excludes all of it by construction rather
   than by pattern-matching. Security that depends on a regex noticing is
   weaker than security that depends on a row never being selected.
2. **No untrusted markers.** Catches external content that was *echoed* into an
   assistant message, which rule 1 cannot see.
3. **Nothing sensitive** (``security.sensitivity``): credentials, identifiers,
   and anything the user marked confidential.
4. **No ``system_event`` rows.** Guard denials and eviction notices are internal
   audit, not conversation, and they quote the arguments they refused.

Withheld rows are counted and reported by category — a silent filter is one you
cannot tell from a broken one.

Usage:
    python -m jobs.consolidate --dry-run     # show what WOULD be sent, send nothing
    python -m jobs.consolidate               # one pass
    python -m jobs.consolidate --every 6h    # scheduled, in the foreground
"""

from __future__ import annotations

import argparse
import logging
import re
import threading
from dataclasses import dataclass, field

from security.sanitizer import CLOSE_MARKER, OPEN_MARKER
from security.sensitivity import classify

_log = logging.getLogger(__name__)

EVENT_MESSAGE = "message"
SOURCE_CONSOLIDATION = "consolidation"

# Withholding reasons, as reported in the result. Stable strings: the bench tier
# and the tests assert on them.
WITHHELD_NON_MESSAGE = "non-message event"
WITHHELD_SYSTEM_EVENT = "system event"
WITHHELD_EXTERNAL = "external content"
WITHHELD_SENSITIVE = "sensitive"

_SUMMARY_PROMPT = (
    "You are a memory consolidation job for a personal assistant. Below is a "
    "transcript of older conversation. Write a compact list of durable FACTS "
    "about the user worth remembering long-term — one per line, no preamble, no "
    "commentary, no speculation. Omit small talk, omit anything you are unsure "
    "of, and never invent detail that is not stated. If there is nothing worth "
    "keeping, reply with exactly: NOTHING"
)
_NOTHING = "NOTHING"

# A short, cheap guard against the model echoing the transcript back wholesale.
_MAX_SUMMARY_CHARS = 4000


@dataclass
class ConsolidationResult:
    """What one pass did. Returned by :func:`consolidate`, printed by the CLI."""

    considered: int = 0
    sent: int = 0
    withheld: dict[str, int] = field(default_factory=dict)
    served_by: str | None = None
    summary: str = ""
    passage_id: str | None = None
    skipped_reason: str | None = None

    @property
    def wrote_anything(self) -> bool:
        return self.passage_id is not None

    def describe(self) -> str:
        parts = [f"considered={self.considered}", f"sent={self.sent}"]
        for reason, n in sorted(self.withheld.items()):
            parts.append(f"withheld[{reason}]={n}")
        if self.served_by:
            parts.append(f"served_by={self.served_by}")
        if self.skipped_reason:
            parts.append(f"skipped={self.skipped_reason}")
        if self.passage_id:
            parts.append(f"passage={self.passage_id}")
        return " ".join(parts)


# --- the filter -----------------------------------------------------------
def withhold_reason(row: dict) -> str | None:
    """Why this recall row must not go outbound, or None if it may.

    Deny-by-default in shape: every branch returns a reason, and only falling
    off the end permits the row.
    """
    if (row.get("event_type") or "") != EVENT_MESSAGE:
        return WITHHELD_NON_MESSAGE
    if (row.get("role") or "") == "system_event":
        return WITHHELD_SYSTEM_EVENT
    content = row.get("content") or ""
    if OPEN_MARKER in content or CLOSE_MARKER in content:
        return WITHHELD_EXTERNAL
    categories = classify(content)
    if categories:
        return f"{WITHHELD_SENSITIVE}:{','.join(categories)}"
    return None


def select_payload(rows) -> tuple[list[dict], dict[str, int]]:
    """Split recall rows into (sendable, {reason: count})."""
    sendable: list[dict] = []
    withheld: dict[str, int] = {}
    for row in rows:
        reason = withhold_reason(row)
        if reason is None:
            sendable.append(row)
        else:
            withheld[reason] = withheld.get(reason, 0) + 1
    return sendable, withheld


def render_transcript(rows) -> str:
    """Oldest-first, role-prefixed. Recall pages newest-first, so this reverses."""
    return "\n".join(
        f"{row.get('role', '?')}: {(row.get('content') or '').strip()}"
        for row in reversed(rows)
    )


# --- one pass -------------------------------------------------------------
def consolidate(
    memory,
    router,
    *,
    limit: int = 100,
    dry_run: bool = False,
) -> ConsolidationResult:
    """Summarize recent recall into one archival passage, via the Mistral lane.

    ``memory`` is a ``MemoryTools``; ``router`` anything with
    ``chat_background``. Never raises into a scheduler: a provider failure is a
    skipped pass, reported, not a crashed job.
    """
    # event_types=() deliberately fetches EVERYTHING, including the rows that
    # will be withheld. Filtering in SQL would be marginally cheaper and would
    # make the job unable to say what it held back — and a filter that cannot
    # report is indistinguishable from a filter that is broken.
    rows, _ = memory.store.search_messages("", page=0, page_size=limit, event_types=())
    result = ConsolidationResult(considered=len(rows))

    sendable, withheld = select_payload(rows)
    result.sent = len(sendable)
    result.withheld = withheld

    if not sendable:
        result.skipped_reason = "nothing sendable"
        return result

    transcript = render_transcript(sendable)
    # Belt-and-braces on the assembled payload, not just its parts: the check
    # that matters is the one on the bytes actually leaving the process.
    leaked = classify(transcript)
    if leaked or OPEN_MARKER in transcript:
        # Unreachable via select_payload; kept because "unreachable" is a claim
        # about today's code, and this is the last line before the network.
        result.skipped_reason = f"payload check failed: {leaked or 'external marker'}"
        _log.error("Consolidation payload failed the outbound check — sending nothing.")
        return result

    if dry_run:
        result.skipped_reason = "dry run"
        result.summary = transcript
        return result

    try:
        reply = router.chat_background(
            [
                {"role": "system", "content": _SUMMARY_PROMPT},
                {"role": "user", "content": transcript},
            ]
        )
    except Exception as exc:  # a nightly job must not die on free-tier weather
        _log.warning("Consolidation could not reach the background lane: %s", exc)
        result.skipped_reason = f"provider unavailable ({type(exc).__name__})"
        return result

    result.served_by = reply.served_by
    summary = (reply.content or "").strip()[:_MAX_SUMMARY_CHARS]
    result.summary = summary

    if not summary or summary.upper().startswith(_NOTHING):
        result.skipped_reason = "model found nothing worth keeping"
        return result

    if memory.archival is None:
        result.skipped_reason = "archival memory unavailable"
        return result

    result.passage_id = memory.archival.insert(summary, source=SOURCE_CONSOLIDATION)
    memory.record_event(
        "system_event",
        "consolidation",
        f"Consolidated {result.sent} message(s) into archival: {result.describe()}",
        served_by=reply.served_by,
    )
    return result


# --- scheduled ------------------------------------------------------------
_DURATION = re.compile(r"^(\d+)([smhd])$", re.I)
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_every(text: str) -> int:
    """'6h' -> 21600. Plain digits are seconds."""
    text = text.strip()
    if text.isdigit():
        return int(text)
    match = _DURATION.match(text)
    if not match:
        raise ValueError(f"Cannot parse interval {text!r} — use e.g. 900, 30m, 6h, 1d.")
    return int(match.group(1)) * _UNITS[match.group(2).lower()]


def run_scheduled(
    build,
    every_seconds: int,
    *,
    stop: threading.Event | None = None,
    limit: int = 100,
) -> None:
    """Run a pass every ``every_seconds`` until ``stop`` is set.

    ``build`` returns a fresh ``(memory, router)`` per pass rather than closing
    over one: a long-lived job outlives connections, and rebuilding is cheap
    next to the model call it precedes.

    ponytail: a sleep loop, not a cron library. There is one job on one
    interval; APScheduler would be a dependency to express `while True`.
    """
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            memory, router = build()
            result = consolidate(memory, router, limit=limit)
            _log.info("Consolidation pass: %s", result.describe())
        except Exception:  # never let one bad pass end the schedule
            _log.exception("Consolidation pass failed")
        stop.wait(every_seconds)


def start_background(build, every_seconds: int, limit: int = 100) -> threading.Event:
    """Start :func:`run_scheduled` on a daemon thread. Returns its stop event."""
    stop = threading.Event()
    threading.Thread(
        target=run_scheduled,
        args=(build, every_seconds),
        kwargs={"stop": stop, "limit": limit},
        name="consolidation",
        daemon=True,
    ).start()
    return stop


# --- CLI ------------------------------------------------------------------
def _build_default():
    import assembly

    return assembly.build_memory(), assembly.build_router()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="jobs.consolidate",
        description="Summarize recall into archival via the Mistral background lane.",
    )
    ap.add_argument("--limit", type=int, default=100, help="recall rows to consider")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the payload that WOULD be sent and send nothing",
    )
    ap.add_argument(
        "--every",
        default=None,
        help="run on a schedule instead of once (e.g. 30m, 6h, 1d)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.every:
        run_scheduled(_build_default, parse_every(args.every), limit=args.limit)
        return 0

    memory, router = _build_default()
    result = consolidate(memory, router, limit=args.limit, dry_run=args.dry_run)
    print(result.describe())
    if args.dry_run:
        print("\n--- payload that would be sent to the background lane ---")
        print(result.summary or "(nothing)")
    elif result.summary:
        print("\n--- summary ---")
        print(result.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
