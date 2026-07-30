"""The MemGPT-style agent loop, driven by the free-tier failover router.

Responsibilities:
1. Build the system prompt each turn from base instructions + rendered core
   memory + memory stats (counts + % of the active provider's context window).
2. Maintain the FIFO message queue (OpenAI chat format) and call
   ``router.chat(messages, tools)``.
3. Dispatch tool calls, append tool results, honor ``request_heartbeat`` (cap 5).
4. Inject a memory-pressure warning once per turn when usage crosses the
   threshold.
5. Persist every event to recall memory, tagged with the serving provider.

The loop talks to memory only through ``MemoryInterface`` and to the LLM only
through ``LLMRouter``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, Sequence, runtime_checkable

# Part of the LLMRouter contract below; the exception type is the only import.
from llm.errors import AllProvidersExhausted

from .prompts import eviction_notice, memory_pressure_warning, render_system_prompt
from .token_budget import approx_tokens, format_usage, is_under_pressure

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

        self.messages: list[dict] = []  # in-context FIFO queue (OpenAI format)
        self.last_input_tokens = 0
        self.served_by: str | None = None
        self.last_limit = self._limit_for(None)

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

        # Memory pressure: inject a warning once, at the top of the turn.
        if self.under_pressure():
            warning = memory_pressure_warning(self.usage_string)
            self.memory.record_event("system_event", EVENT_PRESSURE_WARNING, warning)
            self.messages.append({"role": "user", "content": warning})

        outputs: list[str] = []
        last_text = ""
        heartbeats = 0

        while True:
            system = self._build_system()
            call_messages = [{"role": "system", "content": system}, *self.messages]
            try:
                result = self.router.chat(call_messages, tools=self.tools, tool_choice="auto")
            except AllProvidersExhausted as exc:
                # Degrade to a sentence, not a stack trace. Anything already
                # delivered this turn stands; the user just learns why it stopped.
                _log.warning("All providers exhausted: %s", exc)
                self.memory.record_event(
                    "system_event", EVENT_INTERNAL, f"All providers exhausted: {exc}"
                )
                outputs.append(PROVIDERS_EXHAUSTED_MESSAGE)
                self.memory.record_event("assistant", EVENT_MESSAGE, PROVIDERS_EXHAUSTED_MESSAGE)
                return outputs

            self.served_by = result.served_by
            self.last_input_tokens = result.usage.prompt_tokens or approx_tokens(
                _serialize(call_messages)
            )
            self.last_limit = self._limit_for(result.served_by)

            # Echo the assistant turn back into history (content may be None when
            # the model only issues tool calls — that is valid OpenAI shape).
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
            self.messages.append(assistant_msg)

            if result.content and result.content.strip():
                last_text = result.content
                self.memory.record_event(
                    "assistant", EVENT_INTERNAL, result.content, served_by=result.served_by
                )

            if not result.tool_calls:
                break  # plain reply without send_message → surfaced by the fallback

            sent_message = False
            wants_heartbeat = False
            offloaded = False
            for tc in result.tool_calls:
                args = tc.parsed_arguments()
                if tc.name == "send_message":
                    text = args.get("text", "")
                    outputs.append(text)
                    sent_message = True
                    self.memory.record_event(
                        "assistant", EVENT_MESSAGE, text, served_by=result.served_by
                    )
                    tool_content = "Message delivered to the user."
                else:
                    self.memory.record_event(
                        "assistant",
                        EVENT_TOOL_CALL,
                        f"{tc.name}({tc.arguments})",
                        served_by=result.served_by,
                    )
                    tool_content = self.memory.dispatch(tc.name, args)
                    self.memory.record_event("tool", EVENT_TOOL_RESULT, f"{tc.name} -> {tool_content}")
                    if tc.name == "archival_memory_insert" and not tool_content.startswith("Error:"):
                        offloaded = True
                    if args.get("request_heartbeat"):
                        wants_heartbeat = True
                # Every tool_call must be answered before the next model call.
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_content}
                )

            # The summary is safe on "disk" now, so the summarized messages must
            # leave "RAM" — this is the paging half of the MemGPT mechanic.
            if offloaded and self.under_pressure():
                self._evict_offloaded()

            heartbeats += 1
            # Delivered a reply and not explicitly chaining -> stop. Hit the cap
            # -> stop. Otherwise keep looping so the model can chain memory tools
            # and eventually send a message.
            if sent_message and not wants_heartbeat:
                break
            if heartbeats >= self.max_heartbeats:
                break

        # Safety net: if the model never used send_message, surface its text.
        if not outputs and last_text.strip():
            outputs.append(last_text.strip())
            self.memory.record_event(
                "assistant", EVENT_MESSAGE, last_text.strip(), served_by=self.served_by
            )
        return outputs

    # -- internals ---------------------------------------------------------
    def _evict_offloaded(self) -> int:
        """Drop the oldest half of the FIFO after it has been offloaded.

        Without this the queue only ever grows: the agent summarizes into
        archival, frees nothing, and the pressure warning fires on every
        subsequent turn. Returns the number of messages evicted.
        """
        total = len(self.messages)
        if total < MIN_MESSAGES_BEFORE_EVICTION:
            return 0
        cut = self._safe_cut(total // 2)
        if cut <= 0 or cut >= total:
            return 0

        del self.messages[:cut]
        notice = eviction_notice(cut)
        self.messages.insert(0, {"role": "user", "content": notice})
        self.memory.record_event("system_event", EVENT_EVICTION, notice)
        self._recompute_usage()
        return cut

    def _safe_cut(self, cut: int) -> int:
        """Advance ``cut`` past any orphaned tool results.

        Evicting an assistant message while keeping its ``tool`` replies leaves
        a tool result with no matching tool_call — a transcript every
        OpenAI-compatible provider rejects.
        """
        total = len(self.messages)
        while cut < total and self.messages[cut].get("role") == "tool":
            cut += 1
        return cut

    def _recompute_usage(self) -> None:
        """Re-estimate usage from the retained queue after an eviction.

        The provider's ``prompt_tokens`` describes the PRE-eviction prompt, so
        it must not survive; this estimate is replaced by the real count on the
        next call.
        """
        call_messages = [{"role": "system", "content": self._build_system()}, *self.messages]
        self.last_input_tokens = approx_tokens(_serialize(call_messages))

    def _build_system(self) -> str:
        return render_system_prompt(
            self.memory.render_core_memory(),
            self.memory.memory_stats(),
            self.usage_string,
        )

    def _limit_for(self, provider: str | None) -> int:
        window = (
            self.router.context_window(provider)
            if provider
            else self.router.min_context_window()
        )
        cap = self.planning_context_limit
        if cap and cap > 0:
            return min(window, cap)
        return window


def _serialize(messages: Sequence[dict]) -> str:
    try:
        return json.dumps(messages, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(messages)
