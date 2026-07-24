# Track F — Operations & quality — Retrospective

Shipped 2026-07-24 across seven PRs (#58–#64). Track F adds an always-on quality
audit, an opt-in periodic "librarian", and failure-stage provenance on the
ingest pipeline.

## What shipped

| Phase | PR | What |
|---|---|---|
| F.1 | #58 | `audit_findings` table (migration 75) + `AuditService` (6 LLM-free checks as pure functions) + `POST/GET /api/notebooks/{id}/audit` |
| F.2 | #59 | `AuditWidget` — severity-grouped findings + Re-run, on the schema tab |
| F.3 | #60 | Deep audit (`POST …/audit/deep`) — conflicting facts + provenance gaps |
| F.4 | #61 | Deep-audit trigger + LLM badge in the widget |
| F.5 | #62 | Opt-in librarian: `LibrarianService.enqueue_due` + `handle_librarian_audit` consumer |
| F.6 | #63 | `LibrarianSettings` — opt-in toggle + last-run |
| F.7 | #64 | Failure-stage provenance (`set_failed_stage`/`read_failed_stage`) |

## Deliberate deviations from the plan

- **F.3 is LLM-free by reuse.** The plan framed checks 7–8 as *new* LLM checks,
  but the codebase already has an LLM contradiction judge (Z.2,
  `contradiction_judge_service`) persisting `source_verdict` edges. F.3 SURFACES
  those verdicts (conflicting_facts) + flags empty-`source_documents` relations
  (provenance_gaps) — no redundant model call, no second contradiction path. The
  cheap-path no-LLM guard stays trivially green.
- **F.1 checks are pure Python, not SurrealQL aggregates.** The six checks take
  already-fetched entity/source/relation/orphan lists, so they unit-test with
  plain lists (no DB); only the fetch + persistence touch SurrealDB.
- **F.7 shipped failure-provenance only; the `chunked` stage split is deferred to
  F.7b.** Resumability already exists (`advance_source` resumes from
  `processing_stage`), and parse+chunk is one atomic INGEST handler, so
  `ingested` already means "parsed + chunked". A distinct `chunked` stage would
  mean splitting a core handler — real ingest regression risk, marginal benefit.
  The genuinely-missing part (which stage failed) shipped.

## Open follow-ups

- **F.7b** — split parse/chunk into distinct retry boundaries *if* a per-step
  retry is ever needed (currently `ingested` ⊇ chunked).
- **E2E specs** — an end-to-end audit → deep → librarian → resume flow (deferred:
  the Playwright suite needs the full compose stack and is separately red).
- **Librarian interval** — F.5 stores no per-notebook interval (schedule is
  external); when one is added, surface it in `LibrarianSettings` (F.6 left a
  documented gap rather than a non-functional control).

## Notes

- No migration beyond 75: `librarian_enabled` and `failed_stage` ride existing
  schemaless/flexible fields (F-D2/F-D6 recommendations).
- Reuses the proven command/job seam (F.5) and the declarative `SOURCE_PIPELINE`
  driver (F.7) — no new infra, no new always-on daemon.
