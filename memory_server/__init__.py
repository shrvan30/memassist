"""Memory server package.

Phase 1: exposes deterministic memory tools operating over SQLite (core + recall)
and Chroma (archival) storage, dispatched locally by the agent loop.

Phase 2: a FastMCP server in ``__main__.py`` will expose these same tools over
the Model Context Protocol (stdio), and the agent loop becomes an MCP client.
"""
