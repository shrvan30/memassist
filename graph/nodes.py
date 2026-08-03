"""The nodes of the turn cycle (spec §4.3).

Each node takes ``(state, deps)`` and returns a partial ``AgentState``. The
behaviour is the Phase 1.5 ``AgentLoop`` verbatim — only the seams between the
steps are new, so the benchmark score is a real regression gate.

``security_gate`` and ``sanitize_results`` are the security layer's two seams:
nothing untrusted reaches the model unsanitized, and nothing untrusted reaches
core memory at all (spec §6).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.types import interrupt

import observability
from agent.prompts import eviction_notice, memory_pressure_warning, render_system_prompt
from agent.token_budget import approx_tokens, format_usage, is_under_pressure, usage_fraction
from llm.errors import AllProvidersExhausted
from security.guards import check_tool_call
from security.sanitizer import sanitize_external

from .state import AgentState, Deps

TRUST_UNTRUSTED = "untrusted"

# request_heartbeat is ours, not the external server's — sending it along would
# be an unknown argument and the call would fail schema validation.
_AGENT_ONLY_ARGS = frozenset({"request_heartbeat"})

EVENT_MESSAGE = "message"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_PRESSURE_WARNING = "pressure_warning"
EVENT_EVICTION = "eviction"
EVENT_INTERNAL = "internal"
EVENT_SECURITY = "security"
EVENT_FALLBACK = "fallback"

# Below this many messages there is nothing worth paging out, so an offload
# leaves the queue alone.
MIN_MESSAGES_BEFORE_EVICTION = 8

# Shown when every provider in the chain is rate-limited, out of quota, or
# misconfigured. The user gets plain language; the detail goes to the log and to
# the provider-status panel, where it is actually actionable.
DENIED_BY_USER = (
    "Error: the user denied this action, so it was not performed. Do not retry "
    "it or attempt an equivalent action by another route. Acknowledge the "
    "refusal and continue with what you can do."
)

PROVIDERS_EXHAUSTED_MESSAGE = (
    "I've hit the free-tier limit on every language-model provider I can reach, "
    "so I can't think of a reply right now. Your message is saved in my memory — "
    "please try again in a minute, or check the provider panel for details."
)

# Said when the turn produced nothing at all: no message, no prose, and nothing
# found worth quoting. Short and honest — inventing an answer here would be the
# worse failure.
FALLBACK_NOTHING_FOUND = (
    "I couldn't compose an answer this turn — could you rephrase?"
)

FALLBACK_PREAMBLE = (
    "I used up my tool rounds before writing a proper reply. Here is what I "
    "found, quoted from my memory rather than summarized:"
)

FALLBACK_CLOSING = "Ask me again and I'll answer directly."

# How much of the turn's findings the fallback quotes. Bounded because this text
# goes to a human, not to a model: three results is enough to be useful and
# short enough to read.
FALLBACK_MAX_FINDINGS = 3
FALLBACK_CHARS_PER_FINDING = 600

# Result strings that carry no finding. Matched case-insensitively against the
# start of the tool result, whose wording is this repository's own
# (memory_server/memory_tools.py) — not a model's.
_EMPTY_RESULT_PREFIXES = (
    "no messages in recall memory",
    "no results on page",
    "no archival passages",
    "archival memory is empty",
    "error:",
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
    """Rule on every pending tool call before anything runs (spec §6.3).

    Deny-by-default on the node's allowlist, core memory closed once untrusted
    content is in the turn, and archival writes forced to ``source=external``.
    Decisions are recorded per tool_call_id; ``dispatch_tools`` executes them.
    """
    allowed = deps.allowed_tools()
    external = deps.external
    saw_untrusted = bool(state.get("saw_untrusted"))
    decisions: dict[str, dict] = {}
    gated: list[dict] = []

    # Pure evaluation first. Everything above the interrupt() below runs again
    # on resume, so nothing here may have a side effect.
    for tc in state.get("pending_tool_calls") or []:
        decision = check_tool_call(
            tc.name,
            tc.parsed_arguments(),
            allowed_tools=allowed,
            saw_untrusted=saw_untrusted,
            jail=external.jail_of(tc.name) if external else None,
            gated=external.is_gated(tc.name) if external else False,
        )
        decisions[tc.id] = {
            "allowed": decision.allowed,
            "arguments": decision.arguments,
            "reason": decision.reason,
            "rewritten": decision.rewritten,
        }
        if decision.requires_approval:
            gated.append(
                {"tool_call_id": tc.id, "name": tc.name, "arguments": decision.arguments}
            )

    approved = True
    if gated:
        # Suspends the whole turn until a human answers. The checkpointer is
        # what makes this possible — the graph resumes into the state it left.
        approved = bool(interrupt({"kind": "tool_approval", "actions": gated}))
        for action in gated:
            if not approved:
                decisions[action["tool_call_id"]] = {
                    "allowed": False,
                    "arguments": action["arguments"],
                    "reason": DENIED_BY_USER,
                    "rewritten": "",
                }

    # Side effects only below the interrupt, so they happen exactly once.
    blocked: list[str] = []
    for tc in state.get("pending_tool_calls") or []:
        decision = decisions[tc.id]
        if not decision["allowed"]:
            blocked.append(tc.name)
            _log.warning("Guard refused %s: %s", tc.name, decision["reason"])
            deps.memory.record_event(
                "system_event", EVENT_SECURITY, f"Guard refused {tc.name}: {decision['reason']}"
            )
            observability.event(
                "security.guard_denied",
                level="WARNING",
                metadata={"tool": tc.name, "reason": decision["reason"]},
            )
        elif decision["rewritten"]:
            deps.memory.record_event(
                "system_event",
                EVENT_SECURITY,
                f"Guard rewrote {tc.name} arguments: {decision['rewritten']}",
            )
    if gated:
        deps.memory.record_event(
            "system_event",
            EVENT_SECURITY,
            f"Human {'approved' if approved else 'denied'}: "
            f"{', '.join(a['name'] for a in gated)}",
        )
        observability.event(
            "security.interrupt_answered",
            level="DEFAULT" if approved else "WARNING",
            metadata={"approved": approved, "tools": [a["name"] for a in gated]},
        )

    return {
        "tool_decisions": decisions,
        "blocked_tools": [*state.get("blocked_tools", []), *blocked],
        "gated_action": None,
    }


def dispatch_tools(state: AgentState, deps: Deps) -> dict:
    """Run the tool calls, answer every one of them, and page out if offloaded."""
    messages = list(state["messages"])
    outputs = list(state.get("final_reply", []))
    served_by = state.get("served_by")
    sent_message = False
    wants_heartbeat = False
    offloaded = False

    external = deps.external
    external_names = external.names() if external else frozenset()
    decisions = state.get("tool_decisions") or {}
    untrusted: list[dict] = []
    # Accumulates across heartbeats, so `respond` sees the whole turn's work.
    findings: list[dict] = list(state.get("turn_findings") or [])

    for tc in state["pending_tool_calls"]:
        decision = decisions.get(tc.id)
        if decision is not None and not decision["allowed"]:
            # Refused by security_gate. The call is still ANSWERED — an
            # unanswered tool_call is a transcript providers reject — but with
            # the refusal text instead of a result.
            deps.memory.record_event(
                "assistant", EVENT_TOOL_CALL, f"{tc.name}({tc.arguments}) [refused]",
                served_by=served_by,
            )
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": decision["reason"]}
            )
            wants_heartbeat = True  # let the model recover on the next round
            continue

        args = decision["arguments"] if decision is not None else tc.parsed_arguments()
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
            if tc.name in external_names:
                with observability.span(
                    tc.name,
                    as_type="tool",
                    input=args,
                    metadata={"trust": external.trust_of(tc.name), "zone": "external"},
                ):
                    tool_content = external.call(tc.name, _external_args(args))
                # The VERBATIM result goes to recall before anything rewrites it
                # — the sanitized copy is what the model sees, the original is
                # what an audit reads (spec §6.2).
                deps.memory.record_event(
                    "tool", EVENT_TOOL_RESULT, f"{tc.name} -> {tool_content}"
                )
                if external.trust_of(tc.name) == TRUST_UNTRUSTED:
                    untrusted.append(
                        {
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "server": external.server_of(tc.name),
                        }
                    )
            else:
                with observability.span(
                    tc.name,
                    as_type="tool",
                    input=args,
                    metadata={"trust": "internal", "zone": "memory"},
                ) as s:
                    tool_content = deps.memory.dispatch(tc.name, args)
                    if s is not None:
                        s.update(output=tool_content[:500])
                deps.memory.record_event(
                    "tool", EVENT_TOOL_RESULT, f"{tc.name} -> {tool_content}"
                )
                if tc.name == "archival_memory_insert" and not tool_content.startswith("Error:"):
                    offloaded = True
            # Only ALLOWED, EXECUTED tools are recorded. A refused call never
            # reaches here, so the fallback can never quote something the guards
            # stopped, and never has to re-check them.
            findings.append(
                {
                    "tool_call_id": tc.id,
                    "tool": tc.name,
                    "detail": str(args.get("query") or args.get("content") or "")[:80],
                }
            )
            if args.get("request_heartbeat"):
                wants_heartbeat = True
        # Every tool_call must be answered before the next model call.
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})

    update: dict = {
        "messages": messages,
        "final_reply": outputs,
        "pending_tool_calls": [],
        "untrusted_results": untrusted,
        "turn_findings": findings,
        "heartbeat_count": state.get("heartbeat_count", 0) + 1,
        # Delivered a reply and not explicitly chaining -> the turn is over.
        "done": sent_message and not wants_heartbeat,
    }

    # The summary is safe on "disk" now, so the summarized messages must leave
    # "RAM" — this is the paging half of the MemGPT mechanic.
    under = is_under_pressure(state["input_tokens"], state["limit"], deps.pressure_threshold)
    # …but that path needs the MODEL to cooperate by calling
    # archival_memory_insert. A model that ignores the pressure warning would
    # otherwise grow the queue for ever: the stress tier drove 100 turns to 219%
    # of the limit with zero evictions. Paging is a safety property, so it
    # cannot be contingent on the model doing as it is told.
    over_hard_cap = usage_fraction(state["input_tokens"], state["limit"]) >= deps.hard_evict_fraction
    if (offloaded and under) or over_hard_cap:
        update.update(
            _evict_offloaded({**state, **update}, deps, summarized=offloaded and under)
        )
    return update


def sanitize_results(state: AgentState, deps: Deps) -> dict:
    """Rewrite untrusted tool results as marked-up data (spec §6.2).

    The raw result is already in recall memory (``dispatch_tools`` logs it
    verbatim for audit); what the *model* sees is only ever the sanitized copy,
    because the next ``call_llm`` happens strictly after this node.

    ``saw_untrusted`` latches for the rest of the turn: once hostile content is
    in the window, no later tool call in the same turn can be assumed to be the
    user's idea, so the guards close core memory (§6.3).
    """
    pending = state.get("untrusted_results") or []
    if not pending:
        return {}

    by_id = {item["tool_call_id"]: item for item in pending}
    messages = list(state["messages"])
    flagged: list[str] = []

    for i, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        item = by_id.get(message.get("tool_call_id"))
        if item is None:
            continue
        clean = sanitize_external(
            str(message.get("content", "")),
            source=f"{item['name']} via {item['server'] or 'external server'}",
            char_cap=deps.tool_result_char_cap,
        )
        messages[i] = {**message, "content": clean.text}
        if clean.flags:
            flagged.extend(clean.flags)
            _log.warning(
                "Prompt injection flagged in %s result: %s", item["name"], ", ".join(clean.flags)
            )
            deps.memory.record_event(
                "system_event",
                EVENT_SECURITY,
                f"Injection patterns flagged in {item['name']} result: {', '.join(clean.flags)}",
            )
            observability.event(
                "security.sanitizer_flagged",
                level="WARNING",
                metadata={
                    "tool": item["name"],
                    "server": item["server"],
                    "patterns": clean.flags,
                    "marker_collisions": clean.marker_collisions,
                },
            )

    return {
        "messages": messages,
        "untrusted_results": [],
        "saw_untrusted": True,
        "injection_flags": [*state.get("injection_flags", []), *flagged],
    }


def respond(state: AgentState, deps: Deps) -> dict:
    """Terminal node: guarantee the user gets something back.

    ``send_message`` is the only real channel, so a model that ends its turn on
    plain prose would otherwise say nothing at all. That fallback existed
    already. What it did not cover is the turn that ends on a TOOL CALL: the
    model spends every heartbeat searching, the cap stops the cycle, and the
    last assistant message carries tool calls and no content — so there is no
    prose to fall back to and the user gets silence. The benchmark reproduces
    it (T12b) and the prompt alone does not prevent it, because "always finish
    with send_message" is an instruction, and an instruction is not a
    guarantee.

    So the guarantee is here instead, in code: this node cannot return an empty
    reply. What it says is assembled from the turn's own tool results — already
    dispatched, already guarded, already sanitized — so it costs no model call,
    no quota, and adds no words the results did not already contain.
    """
    # A blank string is not a reply. `send_message` with an empty `text`
    # argument lands in final_reply as "" and would otherwise count as having
    # answered, which is the same silence wearing a different hat.
    outputs = [text for text in state.get("final_reply", []) if str(text).strip()]
    if outputs:
        return {"final_reply": outputs, "done": True}

    last_text = (state.get("last_text") or "").strip()
    if last_text:
        deps.memory.record_event(
            "assistant", EVENT_MESSAGE, last_text, served_by=state.get("served_by")
        )
        return {"final_reply": [last_text], "done": True}

    reply, quoted = _fallback_reply(state)
    _log.warning(
        "Turn ended with no message; composed a fallback from %d finding(s).", quoted
    )
    deps.memory.record_event(
        "system_event",
        EVENT_FALLBACK,
        f"Turn produced no send_message and no prose; replied from "
        f"{quoted} tool result(s) after {state.get('heartbeat_count', 0)} round(s).",
    )
    deps.memory.record_event(
        "assistant", EVENT_MESSAGE, reply, served_by=state.get("served_by")
    )
    observability.event(
        "turn.silent_fallback",
        level="WARNING",
        metadata={
            "quoted_findings": quoted,
            "heartbeats": state.get("heartbeat_count", 0),
            "tools_used": [f.get("tool") for f in state.get("turn_findings") or []],
        },
    )
    return {"final_reply": [reply], "done": True}


def _fallback_reply(state: AgentState) -> tuple[str, int]:
    """Compose a reply from this turn's tool results. Returns (text, quoted).

    Reads the result text back off ``messages`` rather than from a copy taken
    when the tool ran, because ``sanitize_results`` rewrites untrusted results
    in place afterwards — quoting the pre-sanitizer text would hand the user
    exactly the content the sanitizer exists to defang.

    Everything here is verbatim. The only words this function adds are its own
    framing; the findings themselves are the tools' output, timestamps and all,
    so a fact keeps the source it arrived with and nothing is invented to fill
    a gap.
    """
    by_id = {
        message.get("tool_call_id"): str(message.get("content") or "")
        for message in state.get("messages", [])
        if message.get("role") == "tool"
    }

    blocks: list[str] = []
    for finding in state.get("turn_findings") or []:
        content = by_id.get(finding.get("tool_call_id"), "").strip()
        if not content or content.lower().startswith(_EMPTY_RESULT_PREFIXES):
            continue
        label = finding.get("tool") or "tool"
        detail = (finding.get("detail") or "").strip()
        header = f"{label} — {detail!r}" if detail else label
        blocks.append(f"• {header}\n{_truncate(content, FALLBACK_CHARS_PER_FINDING)}")

    if not blocks:
        return FALLBACK_NOTHING_FOUND, 0

    # The last few, not the first: when a model searches repeatedly it is
    # refining, so its later queries are the ones aimed at the question asked.
    kept = blocks[-FALLBACK_MAX_FINDINGS:]
    body = "\n\n".join(kept)
    return f"{FALLBACK_PREAMBLE}\n\n{body}\n\n{FALLBACK_CLOSING}", len(kept)


def _truncate(text: str, cap: int) -> str:
    return text if len(text) <= cap else text[: cap - 1].rstrip() + "…"


def _external_args(args: dict) -> dict:
    return {k: v for k, v in args.items() if k not in _AGENT_ONLY_ARGS}


# --- eviction helpers -----------------------------------------------------
def _evict_offloaded(state: AgentState, deps: Deps, summarized: bool = True) -> dict:
    """Drop the oldest half of the FIFO to free context.

    Without this the queue only ever grows: the agent summarizes into archival,
    frees nothing, and the pressure warning fires on every subsequent turn.

    ``summarized=False`` is the forced path — the hard cap fired because the
    model never offloaded. The messages still leave the window, because the
    alternative is a prompt the provider rejects outright, but the notice says
    so honestly rather than claiming a summary that was never written. Nothing
    is actually lost: the recall log has had every message since it arrived.
    """
    messages = list(state["messages"])
    total = len(messages)
    if total < MIN_MESSAGES_BEFORE_EVICTION:
        return {}
    cut = _safe_cut(messages, total // 2)
    if cut <= 0 or cut >= total:
        return {}

    retained = messages[cut:]
    notice = eviction_notice(cut, summarized=summarized)
    if not summarized:
        _log.warning(
            "Hard context cap hit at %d%% — evicting %d message(s) the model never "
            "offloaded to archival.",
            int(usage_fraction(state["input_tokens"], state["limit"]) * 100),
            cut,
        )
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
