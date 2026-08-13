"""LangGraph orchestration for the MemGPT turn cycle (docs/architecture.md).

The graph owns CONTROL FLOW only. Providers stay behind ``llm/router.py``,
storage stays behind the memory interface, and the UI is untouched:
``agent/loop.py`` is a thin adapter whose public API does not change.
"""

from .graph import build_graph, recursion_limit
from .state import AgentState, Deps

__all__ = ["AgentState", "Deps", "build_graph", "recursion_limit"]
