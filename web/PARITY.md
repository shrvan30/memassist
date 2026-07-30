# Streamlit → Next.js parity checklist

Streamlit stays until every row here is checked, then goes in one removal
commit. The point of the list is that "we rebuilt the UI" is not the same claim
as "nothing was lost".

| Capability | Streamlit (`app/streamlit_app.py`) | Next.js (`web/`) | ✓ |
|---|---|---|---|
| Send a message, see the reply | `st.chat_input` → `loop.step()` | `POST /chat`, SSE | ✅ |
| Reply badged with the serving provider | `⚡ served by …` caption | `⚡ served by …` under each bubble | ✅ |
| Live core memory (persona + human) | sidebar `st.code` blocks | sidebar `<pre>` blocks | ✅ |
| Context usage bar + token counts | `st.progress` + `usage_string` | bar + `usage` string | ✅ |
| Memory-pressure warning | `st.warning` when over threshold | amber bar + banner | ✅ |
| Tier counts (recall / archival) | two `st.metric` | three metrics (adds in-context) | ✅ |
| Archival-unavailable notice | caption when Chroma failed | amber line in tiers | ✅ |
| Provider panel (✅/⛔, reason, requests) | `provider_status()` loop | same, from `GET /providers` | ✅ |
| Cooldown remaining | seconds in the reason line | in the reason line | ✅ |
| Reset conversation (keeps saved memory) | button → `loop.reset()` | button → `POST /sessions/{id}/reset` | ✅ |
| Onboarding gate when no keys | `st.text_input` for a key | *not ported* — see below | ⚠️ |
| Approve/deny for gated tools | buttons → `loop.resume()` | `ApprovalDialog` → `POST …/approve` | ✅ |
| Suspended turn blocks new input | early `return` in `main()` | input disabled + 409 from the API | ✅ |
| Exact arguments shown before approving | `st.json(arguments)` | `<pre>` of the arguments | ✅ |
| Router error surfaced, not crashed | `st.error` | `error` SSE event → red bubble | ✅ |

## Beyond parity

- **Live tool activity.** Every tool call, tool result, eviction and security
  decision streams as it happens. Streamlit only ever showed the finished turn.
- **Typing output.** The reply arrives as `token` events.
- **External-tool panel.** Which MCP tools loaded, their trust zone, and which
  are approval-gated — the security posture was invisible in Streamlit.
- **Sessions.** One checkpointer thread per session id, so several
  conversations can share one memory.

## Deliberately not ported

**The onboarding key gate.** Streamlit ran as one local process, so pasting a
key into the browser and calling `os.environ[...] = key` was reasonable. In the
split stack the browser is a different machine from the API, and posting a
provider key through the UI would put a secret in transit and in a server
process's environment on behalf of an unauthenticated caller. Keys belong in
`.env` / compose environment (spec §6.3: "secrets only via env"). The API
reports its own readiness at `GET /providers` instead, and the panel shows
`⛔ no_api_key` per provider — the same diagnosis, without the secret handling.
