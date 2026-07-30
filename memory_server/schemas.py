"""Pydantic models + OpenAI function-calling tool definitions for the memory tools.

Each tool has a Pydantic model (used to validate the model-supplied arguments at
dispatch time) and a hand-written JSON parameter schema (what the LLM sees),
colocated here so the pair stays in sync. Parameter schemas are kept flat (no
nested objects) for reliable tool-calling across all four providers.

Every memory tool carries a ``request_heartbeat`` flag so the agent can chain
another action before replying. ``send_message`` is the terminal reply and has
no heartbeat.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

_HEARTBEAT_DESC = (
    "Set true to immediately regain control after this tool runs so you can chain "
    "another action (e.g. search, then answer) before replying. Set false when you "
    "are done working and about to send your message."
)

# Provenance defaults to 'inferred' everywhere: a fact is only ever labelled
# 'stated' by an explicit claim, so a mislabelled write is an under-claim rather
# than words put in the user's mouth.
_SOURCE_DESC = (
    "Provenance of this fact. Use 'stated' ONLY when the user said it explicitly in "
    "conversation. Use 'inferred' when you worked it out yourself. When unsure, use "
    "'inferred'."
)
_SOURCE_PROP = {"type": "string", "enum": ["stated", "inferred"], "description": _SOURCE_DESC}


# --- Pydantic input models (runtime validation) --------------------------
class SendMessageInput(BaseModel):
    text: str = Field(..., description="The message to show the user.")


class CoreMemoryAppendInput(BaseModel):
    block: Literal["persona", "human"]
    content: str = Field(..., description="The line of text to append.")
    source: Literal["stated", "inferred"] = Field(default="inferred", description=_SOURCE_DESC)
    request_heartbeat: bool = Field(default=False, description=_HEARTBEAT_DESC)


class CoreMemoryReplaceInput(BaseModel):
    block: Literal["persona", "human"]
    old_text: str = Field(..., description="Exact substring to replace (must match verbatim).")
    new_text: str = Field(..., description="Replacement text (may be empty to delete).")
    request_heartbeat: bool = Field(default=False, description=_HEARTBEAT_DESC)


class ConversationSearchInput(BaseModel):
    query: str = Field(..., description="Keywords to search for in past messages.")
    page: int = Field(default=0, ge=0, description="Zero-based results page.")
    request_heartbeat: bool = Field(default=False, description=_HEARTBEAT_DESC)


class ConversationSearchDateInput(BaseModel):
    start: str = Field(..., description="Inclusive start date, 'YYYY-MM-DD'.")
    end: str = Field(..., description="Inclusive end date, 'YYYY-MM-DD'.")
    page: int = Field(default=0, ge=0, description="Zero-based results page.")
    request_heartbeat: bool = Field(default=False, description=_HEARTBEAT_DESC)


class ArchivalMemoryInsertInput(BaseModel):
    content: str = Field(..., description="The passage to store for long-term recall.")
    source: Literal["stated", "inferred"] = Field(default="inferred", description=_SOURCE_DESC)
    request_heartbeat: bool = Field(default=False, description=_HEARTBEAT_DESC)


class ArchivalMemorySearchInput(BaseModel):
    query: str = Field(..., description="Natural-language query to search archival memory.")
    top_k: int = Field(default=5, ge=1, le=50, description="Max passages to return.")
    page: int = Field(default=0, ge=0, description="Zero-based results page.")
    request_heartbeat: bool = Field(default=False, description=_HEARTBEAT_DESC)


INPUT_MODELS: dict[str, type[BaseModel]] = {
    "send_message": SendMessageInput,
    "core_memory_append": CoreMemoryAppendInput,
    "core_memory_replace": CoreMemoryReplaceInput,
    "conversation_search": ConversationSearchInput,
    "conversation_search_date": ConversationSearchDateInput,
    "archival_memory_insert": ArchivalMemoryInsertInput,
    "archival_memory_search": ArchivalMemorySearchInput,
}


# --- OpenAI function tool definitions (what the model sees) ---------------
def _fn(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


_HEARTBEAT_PROP = {"type": "boolean", "description": _HEARTBEAT_DESC}

SEND_MESSAGE_TOOL: dict = _fn(
    "send_message",
    (
        "Send a message to the user. This is the ONLY way to communicate with the "
        "user — any text you write outside this tool is never shown to them. Call it "
        "once when you are ready to reply."
    ),
    {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The message to show the user."}},
        "required": ["text"],
    },
)

MEMORY_TOOLS: list[dict] = [
    _fn(
        "core_memory_append",
        (
            "Append a line of durable, high-value information to a core memory block. "
            "'persona' holds your own identity and behavior; 'human' holds facts about "
            "the user (name, preferences, goals). Core memory is always visible in your "
            "context, so keep entries short and important."
        ),
        {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["persona", "human"]},
                "content": {"type": "string", "description": "The line of text to append."},
                "source": _SOURCE_PROP,
                "request_heartbeat": _HEARTBEAT_PROP,
            },
            "required": ["block", "content"],
        },
    ),
    _fn(
        "core_memory_replace",
        (
            "Replace an exact substring in a core memory block with new text — use it to "
            "correct or update a fact. 'old_text' must match verbatim; if it is not found, "
            "nothing changes and you receive an error. 'new_text' may be empty to delete."
        ),
        {
            "type": "object",
            "properties": {
                "block": {"type": "string", "enum": ["persona", "human"]},
                "old_text": {"type": "string", "description": "Exact substring to replace."},
                "new_text": {"type": "string", "description": "Replacement text."},
                "request_heartbeat": _HEARTBEAT_PROP,
            },
            "required": ["block", "old_text", "new_text"],
        },
    ),
    _fn(
        "conversation_search",
        (
            "Keyword search over the full conversation history (recall memory) that is not "
            "currently in your context window. Returns matching past messages, newest first, "
            "paginated (page starts at 0)."
        ),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."},
                "page": {"type": "integer", "minimum": 0, "description": "Zero-based page."},
                "request_heartbeat": _HEARTBEAT_PROP,
            },
            "required": ["query"],
        },
    ),
    _fn(
        "conversation_search_date",
        (
            "Search recall memory within an inclusive date range. 'start' and 'end' are "
            "'YYYY-MM-DD'. Use when the user refers to something from a specific time."
        ),
        {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "Inclusive start date YYYY-MM-DD."},
                "end": {"type": "string", "description": "Inclusive end date YYYY-MM-DD."},
                "page": {"type": "integer", "minimum": 0, "description": "Zero-based page."},
                "request_heartbeat": _HEARTBEAT_PROP,
            },
            "required": ["start", "end"],
        },
    ),
    _fn(
        "archival_memory_insert",
        (
            "Save a passage to long-term archival memory (a semantic vector store) for "
            "facts, documents, or summaries you may need much later. Use this to offload "
            "detail out of your limited context window."
        ),
        {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The passage to store."},
                "source": _SOURCE_PROP,
                "request_heartbeat": _HEARTBEAT_PROP,
            },
            "required": ["content"],
        },
    ),
    _fn(
        "archival_memory_search",
        (
            "Semantic search over archival memory. Returns the most relevant saved passages "
            "for a query. Use it before answering when the user asks about something you may "
            "have stored in an earlier session."
        ),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max passages."},
                "page": {"type": "integer", "minimum": 0, "description": "Zero-based page."},
                "request_heartbeat": _HEARTBEAT_PROP,
            },
            "required": ["query"],
        },
    ),
]

MEMORY_TOOL_NAMES: frozenset[str] = frozenset(t["function"]["name"] for t in MEMORY_TOOLS)

# Full tool list handed to the router (send_message + the memory tools).
ALL_TOOLS: list[dict] = [SEND_MESSAGE_TOOL, *MEMORY_TOOLS]
