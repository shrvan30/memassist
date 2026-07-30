"""Graph state and the dependency bundle the nodes run against (spec §4.2).

``AgentState`` is what flows between nodes: one turn's worth of working memory.
``Deps`` is what does NOT flow — the router, the memory interface, and the
turn-cycle limits are wired once at build time and bound into each node, so
nothing unserializable ever ends up in state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypedDict, runtime_checkable


@runtime_checkable
class MemoryInterface(Protocol):
    def render_core_memory(self) -> str: ...
    def memory_stats(self) -> dict: ...
    def dispatch(self, name: str, arguments: dict) -> str: ...
    def record_event(
        self, role: str, event_type: str, content: str, served_by: str | None = None
    ) -> None: ...


class LLMRouter(Protocol):
    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        *,
        tool_choice: str = "auto",
        lane: str = "interactive",
    ) -> Any: ...

    def context_window(self, provider: str) -> int: ...
    def min_context_window(self) -> int: ...


class AgentState(TypedDict, total=False):
    """One turn in flight. Nodes return partial dicts; the default reducer
    overwrites, so any node that grows a list returns the whole list."""

    # -- the FIFO and the prompt built from it ----------------------------
    messages: list[dict]        # in-context queue, OpenAI chat format
    system: str                 # system prompt rendered for the next call
    core_render: str            # core memory as injected this turn

    # -- context accounting -----------------------------------------------
    input_tokens: int           # prompt tokens of the last call (or an estimate)
    limit: int                  # min(active provider window, planning cap)
    context_pct: float          # input_tokens / limit
    needs_warning: bool         # pressure_check's verdict, read by the edge

    # -- the heartbeat cycle ----------------------------------------------
    heartbeat_count: int        # completed tool rounds this turn (cap 5)
    pending_tool_calls: list    # router ToolCall objects awaiting dispatch
    done: bool                  # turn is finished; route to respond

    # -- outputs and provenance -------------------------------------------
    served_by: str | None       # provider that answered the last call
    last_text: str              # assistant prose, surfaced only as a fallback
    final_reply: list[str]      # what step() returns to the caller
    gated_action: dict | None   # tool call awaiting human approval

    # -- security ----------------------------------------------------------
    untrusted_results: list     # results from trust=untrusted servers, awaiting
                                # sanitize_results (spec §6.1)
    saw_untrusted: bool         # an untrusted result entered context this turn,
                                # so core memory is closed for the rest of it


class ExternalToolset(Protocol):
    """The slice of ``mcp_client.ExternalTools`` the graph depends on."""

    def names(self) -> frozenset[str]: ...
    def trust_of(self, tool_name: str) -> str: ...
    def server_of(self, tool_name: str) -> str | None: ...
    def call(self, tool_name: str, arguments: dict) -> str: ...


@dataclass(frozen=True)
class Deps:
    """Everything the nodes need that is not turn state."""

    router: LLMRouter
    memory: MemoryInterface
    tools: list[dict]
    planning_context_limit: int
    pressure_threshold: float
    max_heartbeats: int
    external: ExternalToolset | None = None

    def limit_for(self, provider: str | None) -> int:
        """Context budget against the ACTIVE provider's window.

        Windows differ by an order of magnitude across the chain, so the
        pre-first-response default is the smallest one in it.
        """
        window = (
            self.router.context_window(provider)
            if provider
            else self.router.min_context_window()
        )
        cap = self.planning_context_limit
        return min(window, cap) if cap and cap > 0 else window
