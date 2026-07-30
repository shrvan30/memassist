Scratch space for the filesystem MCP server.

Everything the assistant reads or writes on disk is confined here (spec §6.3).
The server is launched with this directory as its only allowed root, and
security/guards.py re-checks every path argument before the call leaves the
process. Writes additionally require explicit approval in the UI.
