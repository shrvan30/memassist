"""Phase 2 entry point: the FastMCP memory server (stdio).

Not implemented in Phase 1 — this stub exists so ``python -m memory_server`` and
the ``.mcp.json`` registration have a clear target. Phase 2 will expose the six
memory tools from ``memory_tools.py`` over the Model Context Protocol, and the
agent loop will become an MCP client.
"""

from __future__ import annotations

import sys


def main() -> None:
    sys.stderr.write(
        "The MemAssist MCP memory server is a Phase 2 feature and is not yet "
        "implemented.\nPhase 1 dispatches the memory tools locally inside the "
        "agent loop (see agent/loop.py).\n"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
