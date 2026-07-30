"""Session registry: one AgentLoop per session id (spec §11 P4).

A session id maps to exactly one ``AgentLoop``, and each loop owns a
checkpointer thread — so "session" at the HTTP layer and "thread" in the graph
are the same thing, and a suspended turn is still suspended when the next
request arrives.

Memory (core / recall / archival) is shared: it is the *user's* memory, not the
session's. Only the in-context window is per-session, which is exactly the
MemGPT split.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Iterator

import assembly
import config
from agent.loop import AgentLoop

_log = logging.getLogger(__name__)


class EventRecorder:
    """Wraps a MemoryInterface and mirrors every event to live subscribers.

    ``record_event`` is already called for every tool call, tool result,
    eviction and security decision, so it is the turn's audit stream — which
    makes it the cheapest possible source of live UI events. No graph changes,
    no new plumbing.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._subscribers: list[Queue] = []
        self._lock = threading.Lock()

    # -- MemoryInterface passthrough --------------------------------------
    def render_core_memory(self) -> str:
        return self._inner.render_core_memory()

    def memory_stats(self) -> dict:
        return self._inner.memory_stats()

    def dispatch(self, name: str, arguments: dict) -> str:
        return self._inner.dispatch(name, arguments)

    def record_event(
        self, role: str, event_type: str, content: str, served_by: str | None = None
    ) -> None:
        self._inner.record_event(role, event_type, content, served_by=served_by)
        self._publish(
            {"type": "event", "role": role, "event": event_type,
             "content": content[:2000], "served_by": served_by}
        )

    def __getattr__(self, item):  # store / archival / core_blocks / …
        return getattr(self._inner, item)

    # -- fan-out -----------------------------------------------------------
    def subscribe(self) -> Queue:
        q: Queue = Queue()
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _publish(self, payload: dict) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            q.put(payload)


@dataclass
class Session:
    session_id: str
    loop: AgentLoop
    memory: EventRecorder
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        """Everything the memory-inspector panel shows, in one read."""
        stats = self.loop.memory.memory_stats()
        blocks = self.loop.memory.core_blocks()
        return {
            "session_id": self.session_id,
            "core": {"persona": blocks.get("persona", ""), "human": blocks.get("human", "")},
            "tiers": {
                "recall_messages": stats.get("recall_messages", 0),
                "recall_events": stats.get("recall_events", 0),
                "archival_passages": stats.get("archival_passages", 0),
                "context_messages": len(self.loop.messages),
            },
            "context": {
                "input_tokens": self.loop.last_input_tokens,
                "limit": self.loop.last_limit,
                "usage": self.loop.usage_string,
                "pct": (
                    self.loop.last_input_tokens / self.loop.last_limit
                    if self.loop.last_limit
                    else 0.0
                ),
                "under_pressure": self.loop.under_pressure(),
            },
            "served_by": self.loop.served_by,
            "pending_approval": self.loop.pending_approval,
            "archival_available": getattr(self.loop.memory, "archival", None) is not None,
        }


class SessionRegistry:
    """Process-local session store.

    The *turn state* behind a session is durable on the Postgres backend (the
    checkpointer is keyed by session id, so a restarted process rebuilds a loop
    onto the same thread and a suspended approval is still suspended). This
    dict is only the cache of live ``AgentLoop`` objects in front of it.

    ponytail: still an in-process dict, so a second API replica rebuilds its own
    loops rather than sharing these. That is now merely wasteful instead of
    wrong — the state they would both read is in the database.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._external = None
        self._router = None

    def startup(self) -> None:
        # Subprocess MCP servers and the provider ledger are per-process, not
        # per-session: started once, shared by every loop.
        self._external = assembly.build_tools()
        self._router = assembly.build_router()

    def shutdown(self) -> None:
        if self._external is not None:
            self._external.close()
        assembly.close_checkpointers()

    def get(self, session_id: str) -> Session:
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing:
                return existing
            memory = EventRecorder(assembly.build_memory(session_id=session_id))
            loop = assembly.build_loop(
                memory=memory,
                router=self._router,
                external=self._external,
                session_id=session_id,
            )
            session = Session(session_id=session_id, loop=loop, memory=memory)
            self._sessions[session_id] = session
            _log.info("Session %s created", session_id)
            return session

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions)


def run_turn(session: Session, work, chunk_size: int = 24) -> Iterator[dict]:
    """Run ``work()`` on a worker thread, yielding SSE payloads as it goes.

    Live events come from the memory recorder; the reply text is chunked into
    ``token`` events afterwards. See api/main.py for why that is the honest
    streaming boundary here.
    """
    queue = session.memory.subscribe()
    outcome: dict = {}

    def target():
        try:
            outcome["replies"] = work()
        except Exception as exc:  # surfaced as an SSE error, never a 500 mid-stream
            _log.exception("Turn failed")
            outcome["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            queue.put(None)  # sentinel: the turn is over

    worker = threading.Thread(target=target, name=f"turn-{session.session_id}", daemon=True)
    worker.start()

    while True:
        try:
            item = queue.get(timeout=120)
        except Empty:
            outcome.setdefault("error", "Timed out waiting for the assistant.")
            break
        if item is None:
            break
        yield item
    worker.join(timeout=5)
    session.memory.unsubscribe(queue)

    if "error" in outcome:
        yield {"type": "error", "message": outcome["error"]}
        return

    served_by = session.loop.served_by
    for reply in outcome.get("replies") or []:
        for i in range(0, len(reply), chunk_size):
            yield {"type": "token", "text": reply[i : i + chunk_size]}
        yield {"type": "message", "text": reply, "served_by": served_by}

    pending = session.loop.pending_approval
    if pending:
        yield {"type": "approval_required", "request": pending}
    yield {"type": "state", **session.snapshot()}
    yield {"type": "done"}


registry = SessionRegistry()
DEFAULT_SESSION = config.API_DEFAULT_SESSION
