"""Entry point for the FastMCP memory server (stdio).

Not yet implemented — this stub exists so ``python -m memory_server`` and the
``.mcp.json`` registration have a clear target. The memory tools are currently
dispatched locally inside the agent loop.
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
