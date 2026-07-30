"""The nodes of the turn cycle (spec §4.3).

Each node takes ``(state, deps)`` and returns a partial ``AgentState``. The
behaviour is the Phase 1.5 ``AgentLoop`` verbatim — only the seams between the
steps are new, so the benchmark score is a real regression gate.

``security_gate`` and ``sanitize_results`` are pass-through stubs: Phase 2 puts
the seams in the graph, Phase 3 fills them in (spec §6).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent.prompts import eviction_notice, memory_pressure_warning, render_system_prompt
from agent.token_budget import approx_tokens, format_usage, is_under_pressure, usage_fraction
from llm.errors import AllProvidersExhausted

from .state import AgentState, Deps

EVENT_MESSAGE = "message"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_PRESSURE_WARNING = "pressure_warning"
EVENT_EVICTION = "eviction"
EVENT_INTERNAL = "internal"

# Below this many messages there is nothing worth paging out, so an offload
# leaves the queue alone.
MIN_MESSAGES_BEFORE_EVICTION = 8

# Shown when every provider in the chain is rate-limited, out of quota, or
# misconfigured. The user gets plain language; the detail goes to the log and to
# the provider-status panel, where it is actually actionable.
PROVIDERS_EXHAUSTED_MESSAGE = (
    "I've hit the free-tier limit on every language-model provider I can reach, "
    "so I can't think of a reply right now. Your message is saved in my memory — "
    "please try again in a minute, or check the provider panel for details."
)

_log = logging.getLogger(__name__)


# --- nodes ----------------------------------------------------------------
def build_prompt(state: AgentState, deps: Deps) -> dict:
    """Render the system prompt for the next call.

    Rebuilt every round, not once per turn: a ``core_memory_append`` in round 1
    has to be visible to the model in round 2.
    """
    core_render = deps.memory.render_core_memory()
    tokens, limit = state["input_tokens"], state["limit"]
    system = render_system_prompt(
        core_render, deps.memory.memory_stats(), format_usage(tokens, limit)
    )
    return {
        "core_render": core_render,
        "system": system,
        "context_pct": usage_fraction(tokens, limit),
    }


def pressure_check(state: AgentState, deps: Deps) -> dict:
    """Decide whether this turn opens with a memory-pressure warning.

    Only on the first pass: the cycle re-enters through ``build_prompt`` every
    heartbeat, and warning on each round would flood the queue with copies.
    """
    first_pass = state.get("heartbeat_count", 0) == 0
    under = is_under_pressure(state["input_tokens"], state["limit"], deps.pressure_threshold)
    return {"needs_warning": first_pass and under}


def inject_warning(state: AgentState, deps: Deps) -> dict:
    """Tell the agent to summarize into archival before it runs out of room."""
    warning = memory_pressure_warning(format_usage(state["input_tokens"], state["limit"]))
    deps.memory.record_event("system_event", EVENT_PRESSURE_WARNING, warning)
    return {"messages": [*state["messages"], {"role": "user", "content": warning}]}


def call_llm(state: AgentState, deps: Deps) -> dict:
    """The one node that talks to a provider — and only through the router."""
    call_messages = [{"role": "system", "content": state["system"]}, *state["messages"]]
    try:
        result = deps.router.chat(call_messages, tools=deps.tools, tool_choice="auto")
    except AllProvidersExhausted as exc:
        # Degrade to a sentence, not a stack trace. Anything already delivered
        # this turn stands; the user just learns why it stopped.
        _log.warning("All providers exhausted: %s", exc)
        deps.memory.record_event(
            "system_event", EVENT_INTERNAL, f"All providers exhausted: {exc}"
        )
        deps.memory.record_event("assistant", EVENT_MESSAGE, PROVIDERS_EXHAUSTED_MESSAGE)
        return {
            "final_reply": [*state.get("final_reply", []), PROVIDERS_EXHAUSTED_MESSAGE],
            "pending_tool_calls": [],
            "done": True,
        }

    # Echo the assistant turn back into history (content may be None when the
    # model only issues tool calls — that is valid OpenAI shape).
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in result.tool_calls
        ]

    update: dict = {
        "messages": [*state["messages"], assistant_msg],
        "served_by": result.served_by,
        "input_tokens": result.usage.prompt_tokens
        or approx_tokens(_serialize(call_messages)),
        "limit": deps.limit_for(result.served_by),
        "pending_tool_calls": list(result.tool_calls),
    }
    if result.content and result.content.strip():
        update["last_text"] = result.content
        deps.memory.record_event(
            "assistant", EVENT_INTERNAL, result.content, served_by=result.served_by
        )
    return update


def security_gate(state: AgentState, deps: Deps) -> dict:
    """Pass-through stub (spec §6.3).

    Phase 3 inspects ``pending_tool_calls`` here and parks a destructive one in
    ``gated_action`` for a human interrupt. Phase 2 dispatches only the six own
    memory tools, none of which is destructive, so nothing is gated yet.
    """
    return {"gated_action": None}


def dispatch_tools(state: AgentState, deps: Deps) -> dict:
    """Run the tool calls, answer every one of them, and page out if offloaded."""
    messages = list(state["messages"])
    outputs = list(state.get("final_reply", []))
    served_by = state.get("served_by")
    sent_message = False
    wants_heartbeat = False
    offloaded = False

    for tc in state["pending_tool_calls"]:
        args = tc.parsed_arguments()
        if tc.name == "send_message":
            text = args.get("text", "")
            outputs.append(text)
            sent_message = True
            deps.memory.record_event("assistant", EVENT_MESSAGE, text, served_by=served_by)
            tool_content = "Message delivered to the user."
        else:
            deps.memory.record_event(
                "assistant", EVENT_TOOL_CALL, f"{tc.name}({tc.arguments})", served_by=served_by
            )
            tool_content = deps.memory.dispatch(tc.name, args)
            deps.memory.record_event(
                "tool", EVENT_TOOL_RESULT, f"{tc.name} -> {tool_content}"
            )
            if tc.name == "archival_memory_insert" and not tool_content.startswith("Error:"):
                offloaded = True
            if args.get("request_heartbeat"):
                wants_heartbeat = True
        # Every tool_call must be answered before the next model call.
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})

    update: dict = {
        "messages": messages,
        "final_reply": outputs,
        "pending_tool_calls": [],
        "heartbeat_count": state.get("heartbeat_count", 0) + 1,
        # Delivered a reply and not explicitly chaining -> the turn is over.
        "done": sent_message and not wants_heartbeat,
    }

    # The summary is safe on "disk" now, so the summarized messages must leave
    # "RAM" — this is the paging half of the MemGPT mechanic.
    under = is_under_pressure(state["input_tokens"], state["limit"], deps.pressure_threshold)
    if offloaded and under:
        update.update(_evict_offloaded({**state, **update}, deps))
    return update


def sanitize_results(state: AgentState, deps: Deps) -> dict:
    """Pass-through stub (spec §6.2).

    Phase 3 wraps results from trust=untrusted MCP servers in
    ``<untrusted_content>`` markers here, before they enter state. Phase 2 has
    only the trusted-internal memory server, whose results pass through.
    """
    return {}


def respond(state: AgentState, deps: Deps) -> dict:
    """Terminal node: guarantee the user gets something back.

    ``send_message`` is the only real channel, so a model that ends its turn on
    plain prose would otherwise say nothing at all.
    """
    outputs = list(state.get("final_reply", []))
    last_text = (state.get("last_text") or "").strip()
    if not outputs and last_text:
        outputs.append(last_text)
        deps.memory.record_event(
            "assistant", EVENT_MESSAGE, last_text, served_by=state.get("served_by")
        )
    return {"final_reply": outputs, "done": True}


# --- eviction helpers -----------------------------------------------------
def _evict_offloaded(state: AgentState, deps: Deps) -> dict:
    """Drop the oldest half of the FIFO after it has been offloaded.

    Without this the queue only ever grows: the agent summarizes into archival,
    frees nothing, and the pressure warning fires on every subsequent turn.
    """
    messages = list(state["messages"])
    total = len(messages)
    if total < MIN_MESSAGES_BEFORE_EVICTION:
        return {}
    cut = _safe_cut(messages, total // 2)
    if cut <= 0 or cut >= total:
        return {}

    retained = messages[cut:]
    notice = eviction_notice(cut)
    retained.insert(0, {"role": "user", "content": notice})
    deps.memory.record_event("system_event", EVENT_EVICTION, notice)
    return {
        "messages": retained,
        "input_tokens": _recompute_usage({**state, "messages": retained}, deps),
    }


def _safe_cut(messages: list[dict], cut: int) -> int:
    """Advance ``cut`` past any orphaned tool results.

    Evicting an assistant message while keeping its ``tool`` replies leaves a
    tool result with no matching tool_call — a transcript every
    OpenAI-compatible provider rejects.
    """
    total = len(messages)
    while cut < total and messages[cut].get("role") == "tool":
        cut += 1
    return cut


def _recompute_usage(state: AgentState, deps: Deps) -> int:
    """Re-estimate usage from the retained queue after an eviction.

    The provider's ``prompt_tokens`` describes the PRE-eviction prompt, so it
    must not survive; this estimate is replaced by the real count on the next
    call.
    """
    system = build_prompt(state, deps)["system"]
    return approx_tokens(
        _serialize([{"role": "system", "content": system}, *state["messages"]])
    )


def _serialize(messages) -> str:
    try:
        return json.dumps(messages, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(messages)
