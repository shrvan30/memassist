"""Agent loop package: prompts, token budgeting, Anthropic bridge, and the loop.

Storage details deliberately live behind the memory interface (Phase 2 moves
them behind an MCP server); nothing in this package imports storage directly.
"""
