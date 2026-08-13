# Security

Trust is engineered, not prompted. Three boundaries, one component each —
and the honest limits, stated.

## 1. What the model READS — `security/sanitizer.py`

External MCP results (web search, files) are `trust=untrusted` in
`mcp_servers.yaml`. Before the model sees them, four ordered steps:

1. **Strip marker collisions — first, on the raw input.** A hostile page
   containing the literal closing marker would otherwise end the envelope
   early and place its payload outside the untrusted markers.
2. **Neutralize instruction-shaped patterns** into visible
   `[flagged: family]` markers — never silent deletion (the model keeps a
   coherent page; an auditor sees the defense fired). Seven families,
   matched on grammar (an imperative aimed at the assistant): override,
   system-prompt exfiltration, memory-write, role-redirect,
   role-impersonation, imperative-to-assistant, credential exfiltration.
3. **Length-cap** at 4,000 characters.
4. **Wrap in untrusted-content markers**, paired with the standing prompt
   rule: content inside is data, never instructions.

The verbatim original is written to recall **before** sanitization: the
model reads the defanged copy; an auditor reads the truth.

## 2. What the model WRITES — `security/guards.py`

- **Deny-by-default tool allowlist** — an unregistered tool name is
  refused, full stop.
- **The `saw_untrusted` latch** — once untrusted content enters a turn,
  core memory is closed for the rest of it, because by the time the model
  chooses tools it has already read the content. Core poisoning is the
  attack that matters: *wrong once, wrong forever* — a poisoned core line
  re-enters the system prompt on every future turn.
- **Forced provenance** — post-untrusted writes go to archival tagged
  `source='external'` regardless of the model's claim.
- **Filesystem**: paths re-checked on our side of the subprocess boundary,
  jailed to `./workspace`; the jail check runs **before** the approval
  prompt — a path escape is refused, never offered to a human. Writes
  suspend the turn on a resumable interrupt (approve/deny in the UI);
  code above the interrupt is side-effect free because it replays on
  resume.

These rules live in code, not the prompt: *a prompt rule is a request and
this has to be a guarantee.*

## 3. What LEAVES the machine — `security/sensitivity.py`

One deterministic regex detector guards both exits — consolidation
payloads to Mistral and Langfuse trace exports. Regex, not an LLM,
because asking a model whether something is sensitive means sending it
the thing first. Tuned for recall over precision: a false positive skips
one row in a batch job; a false negative puts a credential in someone
else's training set. Detected: provider/Google/AWS keys, bearer tokens,
JWTs, private keys, credential assignments, card numbers (Luhn — the
check strips separators, not digits: a dead-rule bug found live and now
isolation-tested), SSN, IBAN, Aadhaar, and user-marked confidentiality.

## The corpus, and the limits

`security/injections/*.yaml` was written **before** the defenses, so
they are graded by attacks they were not shaped by; every bypass found in
the wild becomes a corpus entry (that is how role-impersonation exists).
Enforced by bench T11 and in CI. Honest limits: regex and turn-scoped
latches do not stop semantic injections in plain prose, multi-turn social
engineering, or encoded payloads — the layered roadmap (quarantined LLM
screening, provenance-aware ranking, egress allow-lists) is stated in
[design-decisions.md](design-decisions.md), not claimed as done.
