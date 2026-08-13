# The failover router

Four free-tier providers behaving like one reliable LLM. The $0
constraint produced this module, and it is the project's most distinctive
engineering.

## The chain (`llm/providers.yaml`)

| Priority | Provider | Role |
|---|---|---|
| 1 | Gemini 2.5-flash-lite | Primary: best quality per free request |
| 2 | Groq Llama-3.3-70B | Fast fallback; tight token/day ceilings |
| 3 | OpenRouter free slug | Emergency lane; small daily cap |
| 4 | Mistral Small | Interactive last resort; the background lane |

## The error taxonomy (`llm/errors.py`)

| Signal | Treatment | Why |
|---|---|---|
| 429 (rate) | 60s cooldown, next provider | Healthy, just throttling |
| 402 / daily quota | Cooldown until UTC midnight | Retrying sooner is waste |
| 5xx / transport | One jittered retry, then next | Transient; one retry is cheap |
| 401 / 403 / 404 / **zero-quota** | Permanent disable, ERROR log | Cannot heal with time; a timer would hide a config bug |
| Other 4xx | Raise | Unknown means our model of the world is wrong |

The zero-quota row is the story to know: Google returns 429 both for
"too fast" and for a project with `limit: 0` free allowance. Only the
response body distinguishes them. Under the original taxonomy the second
looped in cooldown forever — the primary provider served **zero requests
across entire phases** while the panel read `cooling_down`. *A cooldown
implies "try again later," which is false here.* Permanent errors
disable-and-fail-over: not cool down (hides the bug), not raise (one
misconfigured provider cannot end a turn the other three could serve).

## Proactive skipping and the budget ledger

Before spending a request the router checks: key present? not disabled?
not cooling? under today's budget? Any hit skips without a network call —
discovering exhaustion via a 429 costs a request and seconds of a user's
turn. Budgets live in the `provider_usage` table keyed by
`(provider, UTC date)`: they persist across container restarts (an
in-memory ledger would cheerfully re-spend an exhausted tier after every
deploy), and they reset at midnight because the key changes — there is no
reset code.

## Transport invariants

- `max_retries=0` on the SDK client — the router is the only retry
  authority. Two retry systems fight: removing the SDK's hidden ~17s of
  inline backoff took the full chain walk from 7.1s to 1.5s.
- 30s per-request cap (`MEMASSIST_REQUEST_TIMEOUT`), replacing the SDK's
  600s default: 4 providers x 30s = the API's 120s turn ceiling. The
  numbers must compose.
- `_normalize` maps four response dialects into one `ChatResult`;
  tool-call arguments pass through as the raw JSON string the model
  produced — validation happens exactly once, in
  `memory_server/schemas.py`.

## Lanes

`chat()` walks the chain (interactive, latency-optimized).
`chat_background()` reaches only Mistral: batch consolidation traffic
belongs on the provider whose huge monthly quota and ~2 RPM ceiling suit
it — and can never drain the daily budgets interactive users need.

Every reply is tagged `served_by` and shown as a badge in the UI: kill
your primary key mid-conversation and watch the badge flip without a
dropped turn.
