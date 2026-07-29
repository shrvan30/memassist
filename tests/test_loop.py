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
    loop.last_input_tokens = 800
    loop.last_limit = 1000

    loop.step("please continue")

    injected = [
        m
        for m in loop.messages
        if isinstance(m.get("content"), str) and "[MEMORY PRESSURE]" in m["content"]
    ]
    assert injected, "expected a memory-pressure warning in the message queue"
    assert mem.store.count_messages(event_types=("pressure_warning",)) == 1


def test_pressure_uses_active_provider_window(mem):
    # Planning cap 1000, provider window 500 -> effective limit is the smaller.
    router = FakeRouter([result(tool_calls=[tool("send_message", '{"text":"hi"}')], prompt_tokens=10)], window=500)
    loop = make_loop(mem, router, planning_context_limit=1000)
    loop.step("hey")
    assert loop.last_limit == 500
