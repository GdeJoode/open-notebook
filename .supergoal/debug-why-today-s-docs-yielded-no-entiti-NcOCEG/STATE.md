# State: Track B.8 — KG live-validation + provenance hardening

**Status:** IN_PROGRESS
**Current phase:** 3
**Started:** 2026-06-20
**Last update:** 2026-06-20
**Run root:** .supergoal/debug-why-today-s-docs-yielded-no-entiti-NcOCEG
**Baseline ref:** b796f7c599cbfe7c448f6f4449cf43cf69284eef

## Phase progress

| # | Phase | Status | Started | Completed | Notes |
|---|-------|--------|---------|-----------|-------|
| 1 | B.8a Model + provenance fix | DONE (APPROVED) | 2026-06-20 | 2026-06-20 | commits c662354, 8c98ce5; gate attempt 2 APPROVED |
| 2 | B.8b Deploy + UI/KG verification | DONE | 2026-06-21 | 2026-06-21 | new image live; ORDER BY fix verified; filter finding |
| 3 | B.8c Live validation + 11-doc resolution assessment | pending | — | — | — |
| 4 | B.8d Polish, Harden + track ledger | pending | — | — | — |

## Engineering check status
- Build: —
- Typecheck: —
- Lint: —
- Tests: —

## Notable events
- 2026-06-20 — Plan created, then revised after reconciling with Track B methodology + substep goals.
- Decisions: qwen2.5:14b · all 11 Convenant PDFs · rebuild image · Supergoal+review-gate · resolution assess-only.
- Per phase: adversarial-reviewer APPROVED required + a status.md ledger row in docs/tracks/B-kg-quality/status.md.

- 2026-06-20 — B.8a DONE: adversarial APPROVED (attempt 2). 2 minors → B.8d. Advancing to B.8b (image rebuild).
- 2026-06-20 — B.8b BLOCKED: docker credsStore=desktop.exe not on WSL PATH; rebuild can't pull base image. Escalated.
- 2026-06-20 — B.8b UNBLOCKED: docker cred helper symlinked onto PATH (structural fix). Rebuild re-running.
- 2026-06-20 — B.8a-2 DONE (APPROVED): independent extraction model; chat=llama3.1, extraction=qwen2.5:14b. Dockerfile reordered. B.8b deploy done; verification + B.8c live extraction pending.
- 2026-06-20 — B.8c BLOCKED: Docker engine down after over-aggressive restart; needs user wsl --shutdown / tray restart. Code all committed.
- 2026-06-20 — Docker restored by user; new image deployed, open_notebook UP. Resuming B.8b.
