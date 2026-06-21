# Track J — Cloud/local model routing — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-21 | track-planner producing plan.md |

## Decisions (from intake 2026-06-21)
- J-Q1 granularity: layered (global → notebook → document; `private` wins, sticky).
- J-Q2 scope: LLM stages only (extraction/summarization/chat); embeddings + parsing stay local/fixed.
- J-Q3 providers: track-planner proposes default chain (Anthropic → OpenAI → local), configurable.
- J-Q4 failover: per-document (whole doc fails over to next provider; local last in cloud mode).
