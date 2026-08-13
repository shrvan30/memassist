"""AI security layer (docs/security.md).

Two modules, two different jobs:

- ``sanitizer`` — everything that comes back from an UNTRUSTED source is data,
  never instructions. It gets wrapped in markers, pattern-flagged, and capped
  before the model sees it (§6.2, OWASP LLM01/LLM02).
- ``guards`` — what a tool call is *allowed* to do, given where the content
  driving it came from. This is the memory-poisoning defense (§6.3).
"""

from .guards import CORE_MEMORY_TOOLS, GuardDecision, check_tool_call
from .sanitizer import SanitizedResult, sanitize_external

__all__ = [
    "CORE_MEMORY_TOOLS",
    "GuardDecision",
    "SanitizedResult",
    "check_tool_call",
    "sanitize_external",
]
