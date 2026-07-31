"""End-to-end tests for the agent loop, using a fake router (no network)."""

from __future__ import annotations

from agent.loop import AgentLoop
from llm.router import ChatResult, ToolCall, Usage


# --- fake router ----------------------------------------------------------
def result(content=None, tool_calls=None, served_by="gemini", prompt_tokens=100):
    return ChatResult(
        served_by=served_by,
        model=f"{served_by}-model",
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=5, total_tokens=prompt_tokens + 5),
    )


def tool(name, arguments, id="call_1"):
    return ToolCall(id=id, name=name, arguments=arguments)


class FakeRouter:
    def __init__(self, scripted, default=None, window=100_000):
        self.scripted = list(scripted)
        self.default = default
        self.window = window
        self.calls = 0

    def chat(self, messages, tools=None, *, tool_choice="auto", lane="interactive"):
        self.calls += 1
        if self.scripted:
            return self.scripted.pop(0)
        if self.default is not None:
            return self.default()
        raise AssertionError("FakeRouter ran out of scripted responses")

    def context_window(self, provider):
        return self.window

    def min_context_window(self):
        return self.window


def make_loop(mem, router, **kw):
    params = dict(planning_context_limit=1000, pressure_threshold=0.7, max_heartbeats=5)
    params.update(kw)
    return AgentLoop(router, mem, tools=[], **params)


def test_tool_then_send(mem):
    router = FakeRouter(
        [
            result(
                tool_calls=[
                    tool(
                        "core_memory_append",
                        '{"block":"human","content":"Name: Alice. Loves coffee.","request_heartbeat":true}',
                    )
                ],
                prompt_tokens=100,
            ),
            result(
                tool_calls=[tool("send_message", '{"text":"Nice to meet you, Alice!"}', id="call_2")],
                prompt_tokens=150,
            ),
        ]
    )
    loop = make_loop(mem, router)

    outputs = loop.step("My name is Alice and I love coffee.")

    assert outputs == ["Nice to meet you, Alice!"]
    assert "Alice" in mem.core_blocks()["human"]
    assert router.calls == 2
    assert loop.served_by == "gemini"
    assert loop.last_input_tokens == 150
    # OpenAI-shaped transcript: last entry is a tool result.
    assert loop.messages[0]["role"] == "user"
    assert loop.messages[-1]["role"] == "tool"
    assert loop.messages[-1]["tool_call_id"] == "call_2"
    # The reply is searchable in recall, tagged with the provider.
    rows, total = mem.store.search_messages("Alice")
    assert total >= 1
    assert any(r["served_by"] == "gemini" for r in rows)


def test_heartbeat_cap(mem):
    default = lambda: result(  # noqa: E731 - never sends a message
        tool_calls=[tool("core_memory_append", '{"block":"human","content":"loop"}')]
    )
    router = FakeRouter([], default=default)
    loop = make_loop(mem, router, max_heartbeats=3)

    outputs = loop.step("hello")

    assert outputs == []
    assert router.calls == 3  # capped


def test_text_fallback_when_no_send_message(mem):
    router = FakeRouter([result(content="Hi there, I forgot to use the tool.")])
    loop = make_loop(mem, router)

    outputs = loop.step("hello")

    assert outputs == ["Hi there, I forgot to use the tool."]
    assert router.calls == 1


def test_memory_pressure_warning_injected(mem):
    router = FakeRouter([result(tool_calls=[tool("send_message", '{"text":"ok"}')], prompt_tokens=50)])
    loop = make_loop(mem, router)
    # Simulate a nearly-full context from a previous turn (limit is 1000).
    loop.seed_context(input_tokens=800, limit=1000)

    loop.step("please continue")

    injected = [
        m
        for m in loop.messages
        if isinstance(m.get("content"), str) and "[MEMORY PRESSURE]" in m["content"]
    ]
    assert injected, "expected a memory-pressure warning in the message queue"
    assert mem.store.count_messages(event_types=("pressure_warning",)) == 1


def test_hard_cap_evicts_even_when_the_model_never_offloads(mem):
    """Paging cannot depend on the model doing as it is told.

    The pressure warning ASKS the agent to offload to archival, and eviction
    used to fire only after it complied. A model that ignores the warning grew
    the queue for ever — the stress tier drove 100 turns to 219% of the limit
    with zero evictions. Above hard_evict_fraction the cut happens regardless.
    """
    reply = lambda: result(  # noqa: E731
        tool_calls=[tool("send_message", '{"text":"ok"}')], prompt_tokens=990
    )
    loop = make_loop(mem, FakeRouter([], default=reply))
    preload = [{"role": "user", "content": f"old message {i}"} for i in range(40)]
    loop.seed_context(messages=preload, input_tokens=990, limit=1000)  # 99%

    loop.step("and another thing")

    assert len(loop.messages) < 40, "the queue must shrink without an archival offload"
    head = loop.messages[0]["content"]
    assert "[CONTEXT EVICTED]" in head
    assert "NOT summarized" in head, "must not claim a summary that was never written"
    assert mem.store.count_messages(event_types=("eviction",)) == 1


def test_a_cooperative_offload_still_reports_itself_as_summarized(mem):
    """The forced path must not swallow the ordinary one."""
    router = FakeRouter(
        [
            result(
                tool_calls=[
                    tool(
                        "archival_memory_insert",
                        '{"content":"Summary of the early turns.","source":"inferred",'
                        '"request_heartbeat":true}',
                    )
                ],
                prompt_tokens=800,
            ),
            result(tool_calls=[tool("send_message", '{"text":"done"}', id="c2")], prompt_tokens=800),
        ]
    )
    loop = make_loop(mem, router)
    preload = [{"role": "user", "content": f"old message {i}"} for i in range(40)]
    loop.seed_context(messages=preload, input_tokens=800, limit=1000)  # 80%: over warn, under cap

    loop.step("summarize things")

    head = loop.messages[0]["content"]
    assert "[CONTEXT EVICTED]" in head
    assert "summarized into" in head and "NOT summarized" not in head


def test_fifo_accumulates_across_turns_and_reset_clears_it(mem):
    reply = lambda: result(  # noqa: E731
        tool_calls=[tool("send_message", '{"text":"ok"}')], prompt_tokens=42
    )
    loop = make_loop(mem, FakeRouter([], default=reply))

    loop.step("first")
    after_one = len(loop.messages)
    loop.step("second")
    assert len(loop.messages) > after_one, "turn state must survive between steps"
    assert loop.last_input_tokens == 42

    loop.reset()
    assert loop.messages == []
    assert loop.last_input_tokens == 0
    assert loop.served_by is None
    # Saved memory is deliberately untouched by a reset.
    assert mem.store.count_messages(event_types=("message",)) > 0


def test_seed_context_sets_state_without_running_a_turn(mem):
    loop = make_loop(mem, FakeRouter([]))
    preload = [{"role": "user", "content": "old"}]

    loop.seed_context(messages=preload, input_tokens=800, limit=1000)

    assert loop.messages == preload
    assert (loop.last_input_tokens, loop.last_limit) == (800, 1000)
    assert loop.under_pressure()


def test_pressure_uses_active_provider_window(mem):
    # Planning cap 1000, provider window 500 -> effective limit is the smaller.
    router = FakeRouter([result(tool_calls=[tool("send_message", '{"text":"hi"}')], prompt_tokens=10)], window=500)
    loop = make_loop(mem, router, planning_context_limit=1000)
    loop.step("hey")
    assert loop.last_limit == 500
