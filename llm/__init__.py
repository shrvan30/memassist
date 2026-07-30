"""Free-tier failover LLM router.

All LLM traffic goes through this package — never call a provider directly from
agent code. Providers are OpenAI-compatible (Gemini, Groq, OpenRouter, Mistral)
and differ only in base_url / api_key / model, so a single ``openai`` client
shape works for all of them.
"""

from .errors import AllProvidersExhausted, ProviderConfigError, RouterError
from .router import ChatResult, Router, ToolCall, Usage, build_default_router

__all__ = [
    "Router",
    "ChatResult",
    "ToolCall",
    "Usage",
    "build_default_router",
    "RouterError",
    "AllProvidersExhausted",
    "ProviderConfigError",
]
