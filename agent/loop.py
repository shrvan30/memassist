"""Thin adapter over the LangGraph turn cycle (spec §4.1).

``AgentLoop`` used to *be* the turn cycle; the cycle now lives in ``graph/``.
What survives here is the public surface everything else already depends on —
``step()``, ``messages``, ``served_by``, ``last_input_tokens``, ``last_limit``,
``under_pressure()`` — so the Streamlit app, the tests, and the benchmark
harness read exactly as before.

Turn state lives in a LangGraph **checkpointer**, not in attributes: an
interrupt suspends the graph mid-turn and resumes it after a human decision,
which only works if the state it resumes into was persisted. The read
properties below — including :attr:`pending_approval` — are views over the
current checkpoint, and the two mutators (:meth:`reset`, :meth:`seed_context`)
are the only way to change it; assigning to ``loop.messages`` would otherwise
write to a throwaway copy and silently do nothing.

*Which* checkpointer is an assembly decision, paired with the storage backend,
and the loop is deliberately ignorant of it. On the Postgres backend the
persistence is real: a turn suspended on an approval survives the process that
suspended it, provided the caller supplies a stable ``thread_id`` so the
rebuilt loop addresses the same thread.
"""

from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import observability
from graph import build_graph, recursion_limit
from graph.nodes import EVENT_MESSAGE
from graph.state import Deps, LLMRouter, MemoryInterface

from .token_budget import format_usage, is_under_pressure


class AgentLoop:
    def __init__(
        self,
        router: LLMRouter,
        memory: MemoryInterface,
        tools: Sequence[dict],
        *,
        planning_context_limit: int,
        pressure_threshold: float,
        max_heartbeats: int,
        external=None,
        checkpointer=None,
        thread_id: str | None = None,
    ) -> None:
        self.router = router
        self.memory = memory
        self.tools = list(tools)
        self.planning_context_limit = planning_context_limit
        self.pressure_threshold = pressure_threshold
        self.max_heartbeats = max_heartbeats
        self.external = external

        self._deps = Deps(
            router=router,
            memory=memory,
            tools=self.tools,
            planning_context_limit=planning_context_limit,
            pressure_threshold=pressure_threshold,
            max_heartbeats=max_heartbeats,
            external=external,
        )
        # The checkpointer is chosen in assembly, paired with the storage
        # backend (Postgres backend -> PostgresSaver). The default keeps turn
        # state in-process, which is right for the benchmark and the tests:
        # they want a clean slate per run, not durability.
        self._checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self._graph = build_graph(self._deps, checkpointer=self._checkpointer)
        self._recursion_limit = recursion_limit(max_heartbeats)
        # A STABLE thread id is what makes durability work: after a restart the
        # rebuilt loop has to address the same thread the dead process wrote,
        # or the persisted checkpoint is unreachable and durability buys
        # nothing. Callers with a durable identity (an API session) pass theirs.
        self._thread_id = thread_id or uuid4().hex
        self._config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": self._recursion_limit,
        }

    # -- state: read -------------------------------------------------------
    def _snapshot(self) -> dict:
        return self._graph.get_state(self._config).values or {}

    @property
    def messages(self) -> list[dict]:
        """The in-context FIFO. A view — use ``seed_context`` to change it."""
        return self._snapshot().get("messages", [])

    @property
    def last_input_tokens(self) -> int:
        return self._snapshot().get("input_tokens", 0)

    @property
    def last_limit(self) -> int:
        return self._snapshot().get("limit", self._deps.limit_for(None))

    @property
    def served_by(self) -> str | None:
        return self._snapshot().get("served_by")

    @property
    def usage_string(self) -> str:
        return format_usage(self.last_input_tokens, self.last_limit)

    def under_pressure(self) -> bool:
        return is_under_pressure(
            self.last_input_tokens, self.last_limit, self.pressure_threshold
        )

    # -- state: write ------------------------------------------------------
    @property
    def thread_id(self) -> str:
        """The checkpointer thread this loop reads and writes."""
        return self._thread_id

    def reset(self) -> None:
        """Clear the in-context window. Saved memory is untouched.

        Deletes the thread's checkpoints rather than blanking the state: it
        drops the accumulated history too, so a long session cannot grow
        without bound, and it clears a pending interrupt (which a state update
        would leave scheduled). ``delete_thread`` is on the base checkpointer
        interface, so this is one behaviour on both savers — and with a durable
        one it matters more, since orphaned threads would otherwise be rows
        nobody ever reads again.
        """
        self._checkpointer.delete_thread(self._thread_id)

    def seed_context(
        self,
        messages: Sequence[dict] | None = None,
        input_tokens: int | None = None,
        limit: int | None = None,
    ) -> None:
        """Force turn state directly, without running a turn.

        For tests and the benchmark, which need to start a turn from a
        specific context — a preloaded FIFO, or usage already over the
        pressure threshold.
        """
        values: dict = {}
        if messages is not None:
            values["messages"] = list(messages)
        if input_tokens is not None:
            values["input_tokens"] = input_tokens
        if limit is not None:
            values["limit"] = limit
        if values:
            self._graph.update_state(self._config, values)

    # -- the turn ----------------------------------------------------------
    def step(self, user_text: str) -> list[str]:
        """Run one user turn. Returns the message(s) the agent sent to the user.

        If the agent asks for a gated action (a filesystem write), the turn
        SUSPENDS: this returns whatever was delivered so far and
        :attr:`pending_approval` becomes non-None. Call :meth:`resume` to finish.
        """
        self.memory.record_event("user", EVENT_MESSAGE, user_text)
        state = self._snapshot()

        return self._invoke(
            {
                "messages": [
                    *state.get("messages", []),
                    {"role": "user", "content": user_text},
                ],
                # Carried across turns; defaulted here for the first one.
                "input_tokens": state.get("input_tokens", 0),
                "limit": state.get("limit", self._deps.limit_for(None)),
                # Per-turn, always reset.
                "heartbeat_count": 0,
                "pending_tool_calls": [],
                "final_reply": [],
                "last_text": "",
                "done": False,
                "saw_untrusted": False,
                "untrusted_results": [],
                "injection_flags": [],
                "blocked_tools": [],
                "tool_decisions": {},
            }
        )

    # -- human-in-the-loop -------------------------------------------------
    @property
    def pending_approval(self) -> dict | None:
        """The gated action waiting on a human, or None (spec §4.3, §6.3).

        A view over the checkpoint, not an attribute. A suspended turn is only
        durable if the *fact that it is suspended* is durable too: after a
        restart the rebuilt loop is a new object with no memory of having asked
        anything, and an attribute here would report "nothing pending" over a
        graph that is still holding an interrupt.
        """
        interrupts = self._graph.get_state(self._config).interrupts
        return interrupts[0].value if interrupts else None

    def resume(self, approved: bool) -> list[str]:
        """Answer a pending approval and run the rest of the turn."""
        if self.pending_approval is None:
            return []
        return self._invoke(Command(resume=bool(approved)))

    def _invoke(self, payload) -> list[str]:
        # Everything already delivered this turn stands even if the turn then
        # suspends; the interrupt itself is read back off the checkpoint.
        with observability.span(
            "turn", as_type="agent", input=_trace_input(payload)
        ) as turn:
            final = self._graph.invoke(payload, self._config)
            summary = observability.summarize_turn(final)
            if turn is not None:
                turn.update(output={"replies": len(final.get("final_reply") or [])},
                            metadata=summary)
            observability.update_trace(
                turn,
                session_id=self._thread_id,
                # Tags are what make a trace findable: "show me every turn where
                # the guards fired" is the question this exists to answer.
                tags=_trace_tags(summary),
            )
            return final.get("final_reply", [])


def _trace_input(payload) -> dict:
    """Only the new user text, never the accumulated FIFO — the window is
    re-sent every heartbeat and would dominate the trace with duplicates."""
    if not isinstance(payload, dict):
        return {"resume": True}
    messages = payload.get("messages") or []
    last = messages[-1] if messages else {}
    return {"user": last.get("content", ""), "context_messages": len(messages)}


def _trace_tags(summary: dict) -> list[str]:
    tags = [f"provider:{summary.get('served_by') or 'none'}"]
    if summary.get("saw_untrusted"):
        tags.append("untrusted-content")
    if summary.get("injection_flags"):
        tags.append("injection-flagged")
    if summary.get("blocked_tools"):
        tags.append("guard-denied")
    if summary.get("interrupted"):
        tags.append("interrupted")
    return tags
