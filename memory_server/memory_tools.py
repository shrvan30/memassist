"""Deterministic memory tools over SQLite (core + recall) and Chroma (archival).

Each of the six memory tools returns a plain string. No LLM calls happen here —
these are pure storage operations, unit-tested directly. The class also exposes
helpers the agent loop needs without importing storage: rendering core memory,
memory stats, recording recall events, and dispatching a validated tool call.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from .schemas import INPUT_MODELS, MEMORY_TOOL_NAMES, MEMORY_TOOLS
from .storage import sqlite as sql
from .storage.chroma import ArchivalStore
from .storage.sqlite import SQLiteStore


# A provenance tag the model typed into `content` itself, rather than passing it
# as the `source` argument. Matched at the end of the line only.
_PROVENANCE_TAG_RE = re.compile(r"\s*\[(?:stated|inferred)\]\s*$", re.IGNORECASE)


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n… [truncated to {cap} characters]"


class MemoryTools:
    def __init__(
        self,
        store: SQLiteStore,
        archival: ArchivalStore | None = None,
        session_id: str = "default",
        page_size: int = 5,
        archival_top_k: int = 5,
        result_char_cap: int = 4000,
    ) -> None:
        self.store = store
        self.archival = archival
        self.session_id = session_id
        self.page_size = page_size
        self.archival_top_k = archival_top_k
        self.result_char_cap = result_char_cap

    # -- the six tools -----------------------------------------------------
    def core_memory_append(self, block: str, content: str, source: str = "inferred") -> str:
        # Models copy the rendered format and write the tag into `content` as
        # well, which would double-tag the line ("... [stated] [inferred]"). The
        # `source` argument is the only authority, so strip any tag they typed.
        content = _PROVENANCE_TAG_RE.sub("", content)
        # Only the human block carries provenance: 'persona' is the agent's own
        # identity, where "did the user say this?" has no meaning.
        line = f"{content} [{source}]" if block == "human" else content
        try:
            new_content = self.store.append_block(block, line)
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Appended to core memory block '{block}'. It now reads:\n{new_content}"

    def core_memory_replace(self, block: str, old_text: str, new_text: str) -> str:
        try:
            new_content = self.store.replace_block(block, old_text, new_text)
        except ValueError as exc:
            return f"Error: {exc}"
        return f"Updated core memory block '{block}'. It now reads:\n{new_content}"

    def conversation_search(self, query: str, page: int = 0) -> str:
        rows, total = self.store.search_messages(query, page=page, page_size=self.page_size)
        return self._format_recall(f"keyword '{query}'", rows, total, page)

    def conversation_search_date(self, start: str, end: str, page: int = 0) -> str:
        try:
            rows, total = self.store.search_messages_by_date(
                start, end, page=page, page_size=self.page_size
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return self._format_recall(f"dates {start}..{end}", rows, total, page)

    def archival_memory_insert(self, content: str, source: str = "inferred") -> str:
        if self.archival is None:
            return "Error: archival memory is not available in this session."
        passage_id = self.archival.insert(content, source=source)
        return f"Saved to archival memory (id {passage_id[:8]}, source {source})."

    def archival_memory_search(self, query: str, top_k: int = 5, page: int = 0) -> str:
        if self.archival is None:
            return "Error: archival memory is not available in this session."
        items, total = self.archival.search(query, top_k=top_k, page=page)
        if total == 0:
            return "Archival memory is empty."
        if not items:
            return f"No archival passages on page {page} (total stored: {total})."
        lines = [f"Archival search for '{query}' — {total} passage(s) stored, page {page}:"]
        for i, item in enumerate(items, start=1 + page * top_k):
            ts = item.get("created_at") or "?"
            lines.append(f"{i}. [{ts}] {item['content']}")
        return "\n".join(lines)

    # -- shared formatting -------------------------------------------------
    def _format_recall(self, label: str, rows: list[dict], total: int, page: int) -> str:
        if total == 0:
            return f"No messages in recall memory match {label}."
        if not rows:
            return f"No results on page {page} for {label} (total matches: {total})."
        shown_end = page * self.page_size + len(rows)
        lines = [
            f"Recall search ({label}) — {total} match(es), "
            f"showing {page * self.page_size + 1}-{shown_end} (page {page}):"
        ]
        for row in rows:
            lines.append(f"[{row['created_at']}] {row['role']}: {row['content']}")
        if shown_end < total:
            lines.append(f"… more results available; request page {page + 1}.")
        return "\n".join(lines)

    # -- dispatch (validated) ---------------------------------------------
    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Validate ``arguments`` against the tool's Pydantic model and run it.

        Returns a length-capped string. Never raises — errors become strings the
        model can read and recover from.
        """
        if name not in MEMORY_TOOL_NAMES:
            return f"Error: '{name}' is not a memory tool."
        model = INPUT_MODELS[name]
        try:
            parsed = model(**arguments)
        except ValidationError as exc:
            return f"Error: invalid arguments for {name}: {exc}"
        # Handlers don't take request_heartbeat; the loop reads it separately.
        kwargs = {k: v for k, v in parsed.model_dump().items() if k != "request_heartbeat"}
        handler = getattr(self, name)
        try:
            result = handler(**kwargs)
        except Exception as exc:  # defensive: surface as text, don't crash the loop
            result = f"Error: {name} failed: {exc}"
        return _truncate(result, self.result_char_cap)

    # -- helpers for the loop / app ---------------------------------------
    def core_blocks(self) -> dict[str, str]:
        return self.store.get_core_blocks()

    def render_core_memory(self) -> str:
        blocks = self.core_blocks()
        return (
            "<persona>\n"
            f"{blocks.get('persona', '').strip()}\n"
            "</persona>\n"
            "<human>\n"
            f"{blocks.get('human', '').strip()}\n"
            "</human>"
        )

    def memory_stats(self) -> dict[str, int]:
        return {
            "recall_messages": self.store.count_messages(event_types=(sql.EVENT_MESSAGE,)),
            "recall_events": self.store.count_messages(),
            "archival_passages": self.archival.count() if self.archival else 0,
        }

    def record_event(
        self, role: str, event_type: str, content: str, served_by: str | None = None
    ) -> None:
        self.store.add_message(
            self.session_id, role, event_type, content, served_by=served_by
        )

    def tool_definitions(self) -> list[dict]:
        return list(MEMORY_TOOLS)
