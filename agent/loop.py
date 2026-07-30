"""Thin adapter over the LangGraph turn cycle (spec §4.1).

``AgentLoop`` used to *be* the turn cycle; the cycle now lives in ``graph/``.
What survives here is the public surface everything else already depends on —
``step()``, ``messages``, ``served_by``, ``last_input_tokens``, ``last_limit``,
``under_pressure()`` — so the Streamlit app, the tests, and the benchmark
harness read exactly as before.

Turn state lives in a LangGraph **checkpointer**, not in attributes: an
interrupt suspends the graph mid-turn and resumes it after a human decision,
which only works if the state it resumes into was persisted. The read
properties below are views over the current checkpoint, and the two mutators
(:meth:`reset`, :meth:`seed_context`) are the only way to change it — assigning
to ``loop.messages`` would otherwise write to a throwaway copy and silently do
nothing.
"""

from __future__ import annotations

from typing import Sequence
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

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
        # ponytail: InMemorySaver keeps every checkpoint for the life of the
        # thread. reset() rotates the thread id, which is what bounds it in
        # practice; swap for SqliteSaver when turn state has to outlive the
        # process (Phase 4 already puts a database underneath).
        self._graph = build_graph(self._deps, checkpointer=InMemorySaver())
        self._recursion_limit = recursion_limit(max_heartbeats)
        self._pending_approval: dict | None = None
        self._new_thread()

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
    def _new_thread(self) -> None:
        self._config = {
            "configurable": {"thread_id": uuid4().hex},
            "recursion_limit": self._recursion_limit,
        }

    def reset(self) -> None:
        """Clear the in-context window. Saved memory is untouched.

        A fresh thread rather than a blanking update: it drops the accumulated
        checkpoints too, so a long session cannot grow without bound.
        """
        self._pending_approval = None
        self._new_thread()

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
        """The gated action waiting on a human, or None (spec §4.3, §6.3)."""
        return self._pending_approval

    def resume(self, approved: bool) -> list[str]:
        """Answer a pending approval and run the rest of the turn."""
        if self._pending_approval is None:
            return []
        self._pending_approval = None
        return self._invoke(Command(resume=bool(approved)))

    def _invoke(self, payload) -> list[str]:
        final = self._graph.invoke(payload, self._config)
        # A turn that hit interrupt() comes back carrying the request instead of
        # having finished. Everything already delivered this turn still stands.
        pending = final.get("__interrupt__")
        self._pending_approval = pending[0].value if pending else None
        return final.get("final_reply", [])
