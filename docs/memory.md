# Memory

The heart of the project: where facts live, how they are tagged, and what
survives a restart. (Everything below survives a restart.)

## The three tiers

| Tier | Store | In context? | Reached by |
|---|---|---|---|
| Core — persona + human blocks | `core_blocks` table | Every turn, rendered into the system prompt | `core_memory_append` / `core_memory_replace` |
| Recall — complete event log | `messages` table | Never | `conversation_search` / `conversation_search_date` |
| Archival — long-term passages | Chroma or pgvector | Never | `archival_memory_insert` / `archival_memory_search` |

Core blocks are capped at 2,000 characters each because they ride in
every prompt — the limit forces curation (replace, consolidate) over
hoarding. Recall stores every event with role, event_type, `served_by`,
and timestamp; a malformed date returns a correctable `Error:` string the
model can retry, never silent empty results. Archival embeds locally
(bge-small-en-v1.5, 384-d, L2-normalized, cosine) so "which medication
makes him unwell?" retrieves a penicillin-allergy note sharing no words
with the query. `send_message` is the seventh tool and the only channel
to the user.

## Provenance

Every fact carries `stated` (the user said it), `inferred` (the model
concluded it — the safe default: an under-claim rather than words put in
the user's mouth), or `external` (came from the web). The system assigns
provenance where it knows better: after untrusted content enters a turn,
guards force writes to `external` regardless of the model's claim, and a
"[stated]" tag typed into content itself is stripped by the storage layer
— tags are columns, not prose.

## Paging and eviction (the MemGPT mechanic)

- At **70%** of the active provider's window (capped at the 32k planning
  limit — the smallest window in the chain, so failover mid-conversation
  never strands a transcript) a warning is injected once per turn.
- The cooperative path: the model summarizes old turns into archival;
  eviction then removes what was summarized.
- The forced path: at **95%**, eviction runs regardless — because the
  100-turn stress test once grew the window to **219%** when eviction
  depended on model cooperation. *Paging is a safety property, so it
  cannot be contingent on the model doing as it is told.*
- `_safe_cut` never separates an assistant tool-call from its results
  (every provider rejects such transcripts); half the queue is evicted
  with an 8-message floor; the marker left behind tells the truth about
  whether a summary exists — a false "summarized" is a lie the model
  then acts on.

## Consolidation (recall -> archival)

`jobs/consolidate.py` summarizes spans of recall into archival passages
tagged `source='consolidation'`, through the Mistral background lane.
Four withhold rules guard the outbound payload, in order: only
`event_type='message'` rows are ever selected (selection-level security —
stronger than any detector); system events excluded; untrusted-marked
content excluded; sensitive content excluded (see
[security.md](security.md)). Withheld rows are counted and reported by
category — *a silent filter is one you cannot tell from a broken one.*
Because derived data is tagged, the whole consolidation layer can be
dropped and rebuilt from recall at any time; text is the source of truth
and `migrate_embeddings.py` re-derives vectors when the model changes.

## Retrieval quality (measured, unscored)

Over 50 facts: 84% precision@1, 94% within top-3, ~40ms/query. The agent
retrieves top-5, so most P@1 misses are recovered by the model reading
the candidates. Every observed miss was a neighboring fact in the same
topic; mitigations (metadata filtering, cross-encoder reranker, hybrid
BM25) are v1.2 roadmap with costs stated in
[design-decisions.md](design-decisions.md).
