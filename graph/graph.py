"""StateGraph wiring for the turn cycle (spec §4.3).

```
START → build_prompt → pressure_check ─(≥threshold)→ inject_warning ─┐
             ▲              │(under)                                 │
             │              ▼                                        │
             │           call_llm ◄───────────────────────────────---┘
             │              │
             │    ┌─ tool calls? ─── no ──► respond → END
             │    ▼ yes
             │  security_gate → dispatch_tools → sanitize_results
             │                                        │
             └──── heartbeat < cap and not done? ──────┘
                                 else → respond → END
```

The cycle re-enters at ``build_prompt`` rather than ``call_llm`` so the model
sees fresh core memory and fresh stats on every heartbeat; ``pressure_check``
short-circuits after the first pass, so the warning is still injected once.
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AgentState, Deps

# Supersteps per heartbeat round (6) plus slack for the warning and respond.
# The counter in `dispatch_tools` is the real limit; this is the backstop that
# catches a routing bug the counter cannot see.
_STEPS_PER_ROUND = 8
_RECURSION_SLACK = 8


def recursion_limit(max_heartbeats: int) -> int:
    return _STEPS_PER_ROUND * max(1, max_heartbeats) + _RECURSION_SLACK


# --- conditional edges ----------------------------------------------------
def route_pressure(state: AgentState) -> str:
    return "inject_warning" if state.get("needs_warning") else "call_llm"


def route_after_llm(state: AgentState) -> str:
    return "security_gate" if state.get("pending_tool_calls") else "respond"


def route_heartbeat(state: AgentState, max_heartbeats: int) -> str:
    if state.get("done"):
        return "respond"
    return "respond" if state.get("heartbeat_count", 0) >= max_heartbeats else "build_prompt"


# --- wiring ---------------------------------------------------------------
def build_graph(deps: Deps, checkpointer=None):
    """Compile the turn graph. Nodes are bound to ``deps`` at build time.

    A checkpointer makes state survive between ``invoke`` calls, which is what
    lets the FIFO accumulate across turns — and what ``interrupt()`` needs to
    suspend a turn and resume it after a human decision (spec §4.3).
    """
    g = StateGraph(AgentState)
    for name in (
        "build_prompt",
        "pressure_check",
        "inject_warning",
        "call_llm",
        "security_gate",
        "dispatch_tools",
        "sanitize_results",
        "respond",
    ):
        g.add_node(name, partial(getattr(nodes, name), deps=deps))

    g.add_edge(START, "build_prompt")
    g.add_edge("build_prompt", "pressure_check")
    g.add_conditional_edges(
        "pressure_check", route_pressure, ["inject_warning", "call_llm"]
    )
    g.add_edge("inject_warning", "call_llm")
    g.add_conditional_edges("call_llm", route_after_llm, ["security_gate", "respond"])
    g.add_edge("security_gate", "dispatch_tools")
    g.add_edge("dispatch_tools", "sanitize_results")
    g.add_conditional_edges(
        "sanitize_results",
        partial(route_heartbeat, max_heartbeats=deps.max_heartbeats),
        ["build_prompt", "respond"],
    )
    g.add_edge("respond", END)
    return g.compile(checkpointer=checkpointer)
