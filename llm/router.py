"""Priority-ordered free-tier failover router.

Every LLM call goes through ``Router.chat``. The router walks a priority-ordered
provider chain, proactively skips providers that are cooling down or over budget
(per the persistent :class:`BudgetLedger`), fails over on 429/402/5xx, records
usage, and tags the result with ``served_by`` so the UI can show a provider
badge.

The provider transport is injected via ``client_factory``, so the whole failover
algorithm is unit-tested with fake clients and never touches the network.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import yaml

from . import errors
from .budgets import BudgetLedger

DEFAULT_TEMPERATURE = 0.3
DEFAULT_PROVIDERS_YAML = Path(__file__).with_name("providers.yaml")

_log = logging.getLogger(__name__)


# --- config + result types ------------------------------------------------
@dataclass(frozen=True)
class ProviderConfig:
    name: str
    priority: int
    model: str
    base_url: str
    api_key_env: str
    context_window: int = 32_768
    lanes: tuple[str, ...] = ("interactive",)
    daily_request_limit: int | None = None
    daily_token_limit: int | None = None
    rpm_limit: int | None = None
    role: str = ""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model produced it

    def parsed_arguments(self) -> dict:
        try:
            parsed = json.loads(self.arguments or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}


@dataclass
class ChatResult:
    served_by: str  # provider name — surfaced as the UI badge
    model: str
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


class ChatClient(Protocol):
    """Minimal transport interface the router needs from a provider client.

    The router passes an OpenAI chat-completions kwargs dict (``model``,
    ``messages``, ``temperature``, and ``tools``/``tool_choice`` only when tools
    are present — some providers reject a null ``tools`` field).
    """

    def create(self, **kwargs: Any) -> Any: ...


# --- real OpenAI-compatible transport -------------------------------------
class OpenAIChatClient:
    """Wraps an ``openai`` client pointed at a provider's base_url."""

    def __init__(self, base_url: str, api_key: str) -> None:
        import openai  # imported lazily so tests need no network SDK path

        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def create(self, **kwargs: Any) -> Any:
        return self._client.chat.completions.create(**kwargs)


# --- provider config loading ----------------------------------------------
def load_providers(path: str | Path = DEFAULT_PROVIDERS_YAML) -> list[ProviderConfig]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    providers: list[ProviderConfig] = []
    for entry in data.get("providers", []):
        providers.append(
            ProviderConfig(
                name=entry["name"],
                priority=int(entry["priority"]),
                model=entry["model"],
                base_url=entry["base_url"],
                api_key_env=entry["api_key_env"],
                context_window=int(entry.get("context_window", 32_768)),
                lanes=tuple(entry.get("lanes", ["interactive"])),
                daily_request_limit=entry.get("daily_request_limit"),
                daily_token_limit=entry.get("daily_token_limit"),
                rpm_limit=entry.get("rpm_limit"),
                role=entry.get("role", ""),
            )
        )
    providers.sort(key=lambda c: c.priority)
    return providers


# --- the router -----------------------------------------------------------
class Router:
    def __init__(
        self,
        providers: Sequence[ProviderConfig],
        ledger: BudgetLedger,
        *,
        client_factory: Callable[[ProviderConfig, str], ChatClient],
        api_keys: dict[str, str | None] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        sleep_fn: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        server_retries: int = 1,
    ) -> None:
        self.providers = sorted(providers, key=lambda c: c.priority)
        self.ledger = ledger
        self.temperature = temperature
        self.server_retries = server_retries
        self._client_factory = client_factory
        self._sleep = sleep_fn
        self._rng = rng or random.Random()
        # Resolve API keys once (explicit override wins over environment).
        overrides = api_keys or {}
        self._keys: dict[str, str | None] = {
            cfg.name: overrides.get(cfg.name, os.getenv(cfg.api_key_env))
            for cfg in self.providers
        }
        self._clients: dict[str, ChatClient] = {}
        # Providers ruled out by a ProviderPermanentError this process (bad key,
        # retired model, zero free-tier quota). Never retried, unlike a cooldown.
        self._disabled: dict[str, str] = {}

    # -- public API --------------------------------------------------------
    def chat(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        *,
        tool_choice: str = "auto",
        lane: str = "interactive",
    ) -> ChatResult:
        chain = self._chain_for(lane)
        if not chain:
            raise errors.ProviderConfigError(f"No providers configured for lane '{lane}'.")

        attempts: dict[str, str] = {}
        for cfg in chain:
            available, reason = self._availability(cfg)
            if not available:
                attempts[cfg.name] = reason
                continue
            try:
                raw = self._call_with_retry(cfg, messages, tools, tool_choice)
            except errors.RateLimitError:
                self.ledger.cooldown_for_seconds(cfg.name, 60)
                attempts[cfg.name] = "rate_limited"
                continue
            except errors.QuotaExceededError:
                self.ledger.cooldown_until_utc_midnight(cfg.name)
                attempts[cfg.name] = "quota_exceeded"
                continue
            except errors.ProviderServerError:
                attempts[cfg.name] = "server_error"
                continue
            except errors.ProviderPermanentError as exc:
                # No cooldown: this needs a config change, not a retry timer.
                # Disable for the process and say so loudly, so a dead provider
                # can never again sit at priority 1 silently serving nothing.
                self._disabled[cfg.name] = str(exc)
                _log.error("Provider '%s' disabled for this process: %s", cfg.name, exc)
                attempts[cfg.name] = f"permanently_unavailable ({exc})"
                continue

            result = self._normalize(cfg, raw)
            self.ledger.record_request(cfg.name, tokens=result.usage.total_tokens)
            return result

        raise errors.AllProvidersExhausted(attempts)

    def chat_background(
        self,
        messages: Sequence[dict],
        tools: Sequence[dict] | None = None,
        *,
        tool_choice: str = "auto",
    ) -> ChatResult:
        """Background/batch lane — routes straight to Mistral."""
        return self.chat(messages, tools, tool_choice=tool_choice, lane="background")

    def has_key(self, provider: str) -> bool:
        return bool(self._keys.get(provider))

    def context_window(self, provider: str) -> int:
        for cfg in self.providers:
            if cfg.name == provider:
                return cfg.context_window
        return self.min_context_window()

    def min_context_window(self) -> int:
        """Smallest window in the chain — the safe planning default."""
        return min((cfg.context_window for cfg in self.providers), default=32_768)

    def provider_status(self) -> list[dict[str, Any]]:
        """Per-provider availability snapshot for the UI/debug panel."""
        status: list[dict[str, Any]] = []
        for cfg in self.providers:
            available, reason = self._availability(cfg)
            requests, tokens = self.ledger.get_usage(cfg.name)
            status.append(
                {
                    "name": cfg.name,
                    "priority": cfg.priority,
                    "model": cfg.model,
                    "available": available,
                    "reason": reason,
                    "requests": requests,
                    "tokens": tokens,
                    "cooldown_remaining": self.ledger.cooldown_remaining(cfg.name),
                }
            )
        return status

    # -- internals ---------------------------------------------------------
    def _chain_for(self, lane: str) -> list[ProviderConfig]:
        return [cfg for cfg in self.providers if lane in cfg.lanes]

    def _availability(self, cfg: ProviderConfig) -> tuple[bool, str]:
        if not self.has_key(cfg.name):
            return False, "no_api_key"
        if cfg.name in self._disabled:
            return False, f"permanently_unavailable ({self._disabled[cfg.name]})"
        if self.ledger.is_cooling(cfg.name):
            return False, "cooling_down"
        requests, tokens = self.ledger.get_usage(cfg.name)
        if cfg.daily_request_limit is not None and requests >= cfg.daily_request_limit:
            return False, "daily_requests_exhausted"
        if cfg.daily_token_limit is not None and tokens >= cfg.daily_token_limit:
            return False, "daily_tokens_exhausted"
        return True, "ok"

    def _client(self, cfg: ProviderConfig) -> ChatClient:
        if cfg.name not in self._clients:
            self._clients[cfg.name] = self._client_factory(cfg, self._keys[cfg.name])
        return self._clients[cfg.name]

    def _call_with_retry(self, cfg, messages, tools, tool_choice) -> Any:
        client = self._client(cfg)
        # Build the request kwargs once. tools/tool_choice are included only when
        # tools are present, so providers that reject a null `tools` field are safe.
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = tool_choice
        attempt = 0
        while True:
            try:
                return client.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below or re-raised
                perr = errors.classify_provider_error(cfg.name, exc)
                if isinstance(perr, errors.ProviderServerError) and attempt < self.server_retries:
                    attempt += 1
                    self._sleep(self._backoff(attempt))
                    continue
                raise perr

    def _backoff(self, attempt: int) -> float:
        base = 0.5 * (2 ** (attempt - 1))
        return base + self._rng.uniform(0.0, 0.25)

    def _normalize(self, cfg: ProviderConfig, raw: Any) -> ChatResult:
        choice = raw.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    name=getattr(fn, "name", "") or "",
                    arguments=getattr(fn, "arguments", "") or "",
                )
            )
        usage_obj = getattr(raw, "usage", None)
        usage = Usage(
            prompt_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
        )
        return ChatResult(
            served_by=cfg.name,
            model=getattr(raw, "model", cfg.model) or cfg.model,
            content=getattr(message, "content", None),
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage,
            raw=raw,
        )


# --- default wiring -------------------------------------------------------
def _default_db_path() -> str:
    try:
        import config

        return config.DB_PATH
    except Exception:  # pragma: no cover - config import is best-effort
        return str(Path("data") / "memassist.db")


def build_default_router(
    db_path: str | None = None,
    providers_path: str | Path = DEFAULT_PROVIDERS_YAML,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Router:
    """Production wiring: real OpenAI-compatible clients + a persistent ledger."""
    providers = load_providers(providers_path)
    ledger = BudgetLedger(db_path or _default_db_path())

    def factory(cfg: ProviderConfig, api_key: str) -> ChatClient:
        return OpenAIChatClient(cfg.base_url, api_key)

    return Router(providers, ledger, client_factory=factory, temperature=temperature)
