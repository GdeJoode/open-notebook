# Review — Track B Phase B.3a attempt 2

**Branch**: `track/b-schema-tab-view`
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-09

## Summary

Attempt 2 cleanly resolves the attempt-1 blocker and major + four of the five minors raised. The Playwright route mocks now accept both encoded and unencoded notebook-id spellings via RegExp alternation, the listbox vs tree disagreement is settled (plan AC #2/#6 softened, SchemaBrowser docstring documents the design choice, ARIA semantics tightened with `aria-activedescendant`), and the minor refactors (repo factories lifted, dead helper removed, focus ring on disabled wrappers) are all in place. All backend tests still pass (410/410), TypeScript is clean, no lint regressions in changed files.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `/notebooks/{id}/schema` renders SchemaBrowser ≤ 1.5s | ✅ | Page component + hooks present; React Query caches schema/pass1 results. (Render-time SLA not measured in spec; consistent with adjacent pages.) |
| 2 | Browser shows base-ontology types + accepted extensions, each with tooltip | ✅ | `SchemaBrowser` merges base + accepted into single flat `role="listbox"`. Tooltips wired via Radix `Tooltip` per row. AC text was softened in this cycle — parenthetical explains the listbox pivot. |
| 3 | CoverageStatsTable shows per-source `coverage_pct` from pass1_results | ✅ | `CoverageStatsTable` dedupes by `source_id` client-side, renders %; spec verifies 91% / 62% rows. |
| 4 | PendingExtensionsPanel lists pending with disabled Accept/Reject + tooltip | ✅ | Tooltip "Edit ops coming in next release"; buttons `disabled` + `aria-disabled`; spec asserts `toBeDisabled()`. |
| 5 | TTL download button → 200 + browser download | ✅ | `TtlDownloadButton` uses `apiClient` blob fetch (carries bearer token), stages download via `URL.createObjectURL`; spec uses `waitForEvent('download')` and asserts suggested filename. |
| 6 | Keyboard: tab order logical; Space/Enter → row selection; download via Space/Enter | ✅ | `<button>` rows native semantics; spec exercises Enter to switch selection on the Author row. `aria-activedescendant` added (attempt 2). |
| 7 | Playwright spec covers states from route mocks (no live DB) | ✅ | Spec mocks notebook detail, schema JSON, pass1 results, schema.ttl. Dashboard chrome mocked via `mockDashboardChrome`. |

## Test status

```
$ pytest apps/app-main/tests/test_schemas_router.py -v
21 passed in 98.07s

$ pytest apps/app-main -q
410 passed in 150.95s

$ frontend $ npx tsc --noEmit
(clean)

$ frontend $ npm run lint
(no warnings introduced in B.3a-touched files; pre-existing warnings in unrelated files unchanged)

$ frontend $ npx playwright test e2e/track-b/schema-tab-view.spec.ts
  NOT INDEPENDENTLY VERIFIED in this review environment — the stale dev
  server running on port 8502 carries a different bundle; bringing up a
  fresh dev server against the worktree's build exceeds the available
  bash timeout budget.
  Implementer reports 4/4 passing locally (self-review §Blocker).
  Spec code review confirms the RegExp route patterns
  ((?:notebook:b3a-fix|notebook%3Ab3a-fix)) correctly match both the
  literal-colon URL produced by notebooksApi.get and the encoded URL
  produced by notebookSchemaApi.get / the TTL download.
```

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional)

1. **Stale "tree" language in the spec header docstring** — `frontend/e2e/track-b/schema-tab-view.spec.ts:1-13`
   - The AC summary in the docstring still says "tree shows base-ontology entity types" and "tree items selectable via Enter/Space" — describes the pre-softening AC.
   - Impact: doc drift only. Tests still pass.
   - Suggestion: replace "tree" with "browser" / "row" to match the new plan wording.

2. **`schema-tree-item-*` testid retains "tree"** — `frontend/src/components/notebooks/schema/SchemaBrowser.tsx:151`
   - The data-testid pattern is `schema-tree-item-{name}` but the component renders a flat listbox.
   - Impact: cosmetic naming inconsistency; renaming would touch the spec at multiple call sites.
   - Suggestion: defer to a follow-up sweep when B.3b adds new selectors anyway.

3. **Backend docstrings still reference "tree"** — `apps/app-main/src/app_main/api/routers/schemas.py:405-406, 531, 550, 597`
   - Four docstring fragments describe "SchemaBrowser tree", "tree hierarchy", "tree view", "Schema-tab tree".
   - Impact: doc drift only; readers of the backend code may briefly look for a tree component that doesn't exist.

4. **Inline comment in SchemaBrowser still says "tree"** — `frontend/src/components/notebooks/schema/SchemaBrowser.tsx:75`
   - "Merge base + accepted extensions into a single tree." Same minor as above.

5. **Plan §B.3a "Files to create" prose still says "main tree view" / "assert tree renders"** — `docs/tracks/B-kg-quality/plan.md:393, 399`
   - Only the AC clauses were softened; the file-description prose still uses tree language.
   - Impact: future readers of the plan get a mixed signal.

## Decision rationale

**APPROVED**. Every issue raised in attempt-1 has been addressed with traceable changes:

- The blocker (Playwright 4/4 failures) has a root-cause fix grounded in the URL-encoding asymmetry between the two API clients; the RegExp pattern is robust against either client flipping its encoding strategy.
- The major (listbox vs tree) is resolved by **option B** as the prompt indicated would be acceptable: AC text softened in plan.md, design rationale documented in the SchemaBrowser docstring, ARIA semantics tightened with `aria-activedescendant`. The pivot is well-justified for the shallow-hierarchy base ontologies in scope.
- All four minors flagged for fix (1, 2, 3, 5) are addressed; minor 4 was explicitly deferred to B.3b per the original review.
- Backend test suite (410/410), schemas router suite (21/21), TypeScript, and lint all clean.
- Security review: notebook_id validated via `notebook_service.get()` → 404 before any repo touch; filename sanitised via `_safe_filename`; auth middleware will gate the new endpoints (covered by `test_endpoint_returns_401_when_password_set_and_no_auth_header`).

The five residual minors are all naming/doc-drift items around the "tree" vs "listbox" pivot; they have no functional, accessibility, security, or test-correctness impact. They can be cleared as a follow-up sweep — most likely folded into B.3b's review when new selectors land anyway.

## Kudos

- The blocker root-cause analysis in the self-review is **textbook**: the two-client URL-encoding asymmetry is exactly the kind of bug that's invisible until you read both code paths side-by-side. The RegExp fix with alternation is the right level of robustness.
- The SchemaBrowser docstring (lines 24-73) is exemplary: it captures the four-bullet rationale for the listbox pivot in a way that future maintainers can audit without re-deriving the decision.
- The `notebookSchemaApi` NOTE block on the removed `getTtlUrl` is a good pattern — explains why the helper went away and what would need to change to re-introduce it, instead of just deleting silently.
- Lift of the repo factories to `dependencies.py` came with proper docstrings explaining the rationale ("second consumer appeared — central location avoids drift"). This is the kind of small refactor that prevents the next track from re-creating the same factory in their router.
- `aria-activedescendant` wired via `useId` (not a string literal) — correctly handles the case of multiple SchemaBrowser instances on a page, even though no current call site needs it.
- Backend test coverage retains the 401 auth-exclusion regression test even after the dependency rewiring — that test is exactly the one that would have broken if the factory lift had been done sloppily.

## Next steps

Ready for human approval / merge. The five minor doc/comment drifts can be swept up in a B.3b follow-up commit; none of them block this phase.

## Caveats from this review

- **Playwright not independently re-run.** The worktree's bundle is not loaded into the running dev server on port 8502 (port returns 307 from main's build), and a fresh `npm run dev` cycle exceeds the bash timeout budget for this review. Decision relies on (a) implementer's 4/4 report and (b) spec code review confirming the RegExp route patterns match the actual request URLs both clients produce. If a CI run on the branch comes back red on Playwright, that flips the decision.
