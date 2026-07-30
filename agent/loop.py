"""Thin adapter over the LangGraph turn cycle (spec §4.1).

``AgentLoop`` used to *be* the turn cycle; the cycle now lives in ``graph/``.
What survives here is the public surface everything else already depends on —
``step()``, the mutable ``messages`` FIFO, ``served_by``, ``last_input_tokens``,
``last_limit``, ``under_pressure()`` — so the Streamlit app, the tests, and the
benchmark harness run unmodified.

The attributes are the source of truth between turns: ``step()`` seeds graph
state from them and writes the result back, which is why callers can still
mutate ``loop.messages`` or force ``loop.last_input_tokens`` before a turn.
"""

from __future__ import annotations

from typing import Sequence

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
    ) -> None:
        self.router = router
        self.memory = memory
        self.tools = list(tools)
        self.planning_context_limit = planning_context_limit
        self.pressure_threshold = pressure_threshold
        self.max_heartbeats = max_heartbeats

        self._deps = Deps(
            router=router,
            memory=memory,
            tools=self.tools,
            planning_context_limit=planning_context_limit,
            pressure_threshold=pressure_threshold,
            max_heartbeats=max_heartbeats,
        )
        self._graph = build_graph(self._deps)
        self._config = {"recursion_limit": recursion_limit(max_heartbeats)}

        self.messages: list[dict] = []  # in-context FIFO queue (OpenAI format)
        self.last_input_tokens = 0
        self.served_by: str | None = None
        self.last_limit = self._deps.limit_for(None)

    # -- public API --------------------------------------------------------
    @property
    def usage_string(self) -> str:
        return format_usage(self.last_input_tokens, self.last_limit)

    def under_pressure(self) -> bool:
        return is_under_pressure(
            self.last_input_tokens, self.last_limit, self.pressure_threshold
        )

    def step(self, user_text: str) -> list[str]:
        """Run one user turn. Returns the message(s) the agent sent to the user."""
        self.memory.record_event("user", EVENT_MESSAGE, user_text)
        self.messages.append({"role": "user", "content": user_text})

        final = self._graph.invoke(
            {
                "messages": self.messages,
                "input_tokens": self.last_input_tokens,
                "limit": self.last_limit,
                "heartbeat_count": 0,
                "pending_tool_calls": [],
                "final_reply": [],
                "last_text": "",
                "served_by": self.served_by,
                "done": False,
            },
            self._config,
        )

        self.messages = final["messages"]
        self.last_input_tokens = final["input_tokens"]
        self.last_limit = final["limit"]
        self.served_by = final.get("served_by")
        return final.get("final_reply", [])
