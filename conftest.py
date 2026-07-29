"""Pytest bootstrap.

Ensures the repository root is importable so ``import config`` / ``from agent...``
work even when the package has not been installed with ``pip install -e .``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
