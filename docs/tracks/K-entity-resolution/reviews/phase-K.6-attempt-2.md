# Review — Track K Phase K.6 attempt 2

**Branch**: `track/k6-resolution-ui`
**Fix commit**: `b5a15dc`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-22

## Summary

Both attempt-1 majors are genuinely fixed. The fuzzy-candidate merge now routes
through the same `MergeConfirmDialog` as the cluster card, and there is no
remaining code path where a candidate or cluster merge issues `POST /apply`
without a dialog confirm. AC4 (per-entity alias add/remove) is honestly DEFERRED
with the dead `AliasManager.tsx` deleted and the missing backend endpoint named
as the follow-up. No regression to attempt-1 passes. Approved.

## Major fixes verified

### MAJOR 1 — destructive-merge gate now covers fuzzy candidates (FIXED)

- Two `applyMerge.mutate([cluster])` call sites exist (resolution/page.tsx:77
  `onApprove`, :91 `onApproveCandidate`). Both are reachable **only** via a
  card's `onApprove` prop.
- `MergeClusterCard`: "Approve merge" → `setConfirmOpen(true)` (line 117). The
  page `onApprove` is wired solely to `MergeConfirmDialog.onConfirm` (line 143),
  which fires only from the `AlertDialogAction` onClick. The pre-rev2 direct
  apply path is gone (diff confirms the button now opens the dialog).
- `CandidateMergeCard` (new): "Merge" → `setConfirmOpen(true)` (line 66); page
  `onApprove` reached only via `MergeConfirmDialog.onConfirm` (line 89). No
  bypass.
- Dialog shows the surviving winner (Crown + `winnerLabel`) and losers
  (`loserLabels` badges) before applying (MergeConfirmDialog.tsx:66-88).
- Candidate id correctness: `candidateToApplyCluster` maps `winner_id`/`loser_id`
  to the apply payload (entity-resolution.ts:95-105); unit test pins it.
- E2E (`entity-resolution.spec.ts:197`): asserts `capture.applyBody` is
  **undefined** after opening the dialog (line 241) and, post-confirm, that the
  payload carries `winner_id: entity:vws1` / `loser_ids: [entity:vws2]`
  (lines 249-250). Genuine pre-confirm no-fire proof.
- Double-submit guard: `AlertDialogAction` has `disabled={confirmDisabled}`
  (MergeConfirmDialog.tsx:96), fed by `applyMerge.isPending || addOverlay.isPending`
  from both cards. Confirmed.

### MAJOR 2 — AC4 deferral is honest (FIXED)

- `AliasManager.tsx` is DELETED (128 lines removed in b5a15dc); no dangling
  import (only doc references remain in plan/status/escalations).
- escalations.md K.6 entry explicitly **reverses** the rev1 "wire to overlay"
  decision, explains the wrong-semantics reasoning (add-alias silently became a
  graph-wide force-merge), and names the needed backend
  `POST/DELETE /entities/{id}/aliases` as the K.6 follow-up. Clearly scoped, not
  hiding a gap.
- Read-only alias count + `ExternalIdBadges` + `OverlayEditor` remain as the
  honest shipped alias surface.

## No-regression

- Cluster gate (AC2): MergeClusterCard refactor still resolves the user-picked
  winner via `buildApplyCluster(cluster, winnerId)` and sends correct ids through
  the shared dialog. Verified.
- reject→force-split (AC3), states (AC1), a11y scope toggle (AC5),
  ExternalIdBadges-empty→null (AC6), E2E (AC7) unchanged and intact.
- DOI minor: `providerLabel('10.xxxx')` → 'DOI', `toHref` → `https://doi.org/...`;
  `ExternalIdBadges` wraps each badge in an `<a href target=_blank rel=noopener>`
  — still links. Pure util + test added.

## Test status

```
vitest: Test Files 9 passed (9) | Tests 64 passed (64)
playwright --list: Total 7 tests in track-k/entity-resolution.spec.ts (incl. candidate-confirm)
tsc --noEmit: exit 0
eslint (changed files): exit 0
backend: branch diff to api/open_notebook/services/commands = 0 lines (untouched)
```

## Issues found

None blocking. No blockers, no majors.

### Minor (optional, non-blocking)

1. `frontend/e2e/track-k/entity-resolution.spec.ts:241` — the pre-confirm
   `expect(capture.applyBody).toBeUndefined()` is a synchronous check; it proves
   no apply fired by the open action but cannot catch an apply that races in
   after the assertion. Acceptable here (the mutate is user-gesture-driven), but
   a short negative poll would be marginally stronger.

## Decision rationale

0 blockers, 0 majors. Both attempt-1 majors are fixed with the destructive path
provably gated (no apply without confirm for either entry point) and AC4
deferred honestly. APPROVED.

## Next steps

Ready for human approval / merge. Follow-up (separate, additive): backend
`POST/DELETE /entities/{id}/aliases` to un-defer AC4 with a real per-entity
alias editor.
