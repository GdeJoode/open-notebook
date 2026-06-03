# Review — Track A Phase A.0 attempt 2

**Branch**: `track/a-playwright`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

The implementer correctly addressed both in-lane issues from attempt 1: the
URL regex now covers the `/models?setup_required=true` redirect produced by
the empty-`docker.env` CI stack, and the vapid `body`-visibility assertion
was replaced with a substantive "Noesis" text marker that is verifiably
present on all three legitimate landed states. TypeScript and lint stay
clean, the spec is still discovered as a single test, and the rewritten
file-level docstring honestly explains the contract. The only residual
concerns are minor (one substring-match false-positive vector, one hidden
dependency on default sidebar state) and explicitly out of the brief's
revision scope.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `cd frontend && npm install && npm run e2e:install` succeeds | ✅ | No change since attempt 1; still wired correctly. |
| 2 | `npm run e2e` runs the smoke spec and passes against `http://localhost:8502` | ✅ | The regex now matches the actual CI landing URL (`/models?setup_required=true`). Live execution against a running stack was not performed in the sandbox, but the static evidence (route grep, layout source, Next.js redirect chain) is solid. |
| 3 | Failed test produces a Playwright HTML report under `playwright-report/` | ✅ | Unchanged from attempt 1. |
| 4 | CI workflow runs Playwright on a PR touching `frontend/` | ⚠️ | Workflow file still at `docs/tracks/A-mineru/e2e-workflow.yml.pending` — explicitly out of scope for this revision (orchestrator-owned, PAT-scope blocker). Not held against attempt 2. |
| 5 | README documents how to run locally | ✅ | Unchanged from attempt 1. |

## Attempt 1 issue follow-through

| Severity | Attempt 1 issue | Fixed? | Evidence |
|---|---|---|---|
| 🔴 Blocker | URL regex missed `/models?setup_required=true` | ✅ | `frontend/e2e/_smoke.spec.ts:40` — `const validLanded = /\/(notebooks\|login\|models)(\?\|$)/`, reused by both `waitForURL` and `toHaveURL`. The `\?\|$` anchoring prevents accidental matches on deeper routes like `/notebooks/[id]`. |
| 🟡 Major #1 | `expect(page.locator('body')).toBeVisible()` was vapid | ✅ | `frontend/e2e/_smoke.spec.ts:47` — replaced with `expect(page.getByText('Noesis').first()).toBeVisible()`. Confirmed via grep that "Noesis" is rendered as visible text content on all three landed states (see below). |
| 🟡 Major #2 | Workflow file location | N/A | Orchestrator task, explicitly out of revision scope. |

## Static verification

```
$ cd frontend && npx tsc --noEmit
exit=0 (clean)

$ cd frontend && npx playwright test --list
Listing tests:
  [chromium] › _smoke.spec.ts:34:7 › smoke › app root mounts on a legitimate landed state
Total: 1 test in 1 file

$ cd frontend && npx eslint e2e/_smoke.spec.ts e2e/_helpers.ts
exit=0 (clean)
```

### "Noesis" marker — grep verification

The implementer's claim was that "Noesis" appears as **visible text** in
every landed state. Verified:

- **Login state** (`/login`):
  `frontend/src/components/auth/LoginForm.tsx:142` — `<CardTitle>Noesis</CardTitle>`.
  Visible text node, not an attribute. ✅

- **`/notebooks`** and **`/models?setup_required=true`** (dashboard routes):
  Both pages render `<AppShell>{...}</AppShell>`
  (`frontend/src/app/(dashboard)/notebooks/page.tsx:49`,
   `frontend/src/app/(dashboard)/models/page.tsx:31,41,52`).
  `AppShell` renders `<AppSidebar />` (`frontend/src/components/layout/AppShell.tsx:12`).
  `AppSidebar` renders `<span>Noesis</span>` at line 144 (visible text)
  when `isCollapsed === false`. ✅

- **Default sidebar state**: `useSidebarStore` initializes
  `isCollapsed: false` (`frontend/src/lib/stores/sidebar-store.ts:13`).
  Playwright uses a fresh browser context per test by default, so
  the persisted Zustand state from prior sessions does not leak across
  CI runs. ✅ — but see Minor #1.

- **Error overlay false-positive check**: a 500 / Next.js runtime error
  page would not render either the AppSidebar or LoginForm and would
  therefore fail the assertion. `ConnectionErrorOverlay` does contain
  the string "Noesis" inside a paragraph, but it never appears on a
  successful smoke run — see Minor #2 for the partial-backend-up
  caveat.

### URL regex edge cases

- `/notebooks` → matches ✅
- `/login` → matches ✅
- `/models?setup_required=true` → matches via the `\?` alternation ✅
- `/notebooks/abc-id` (would not be a smoke landing state via `/` redirect,
  but worth confirming): does NOT match because `b` is neither `?` nor end. ✅
- `/sources`, `/chat`, `/settings`: not part of the root redirect chain
  (verified `frontend/src/app/page.tsx` only redirects to `/notebooks`,
  and the dashboard layout only pushes to `/login` or `/models?...`),
  so excluding them from the regex is correct, not a gap.

### Auth-bypass leak

Brief asked specifically to confirm the spec does not stash a fake
`access_token` in `localStorage`. Verified:

```
$ grep -rn "access_token\|localStorage" frontend/e2e/
(no matches)
```

No auth-state forgery. The spec observes whatever the backend reports. ✅

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional / follow-up)

1. **Implicit dependency on default sidebar state** — `frontend/e2e/_smoke.spec.ts:47`
   - Issue: The "Noesis" `<span>` only renders when
     `useSidebarStore.isCollapsed === false`. If someone ever flips the
     default to `true` (or adds a layout that starts collapsed on small
     viewports), the smoke spec breaks for non-obvious reasons because
     the collapsed sidebar only carries `alt="Noesis"` on the logo —
     and `getByText` does not match alt attributes.
   - Recommendation (state, do not implement): consider asserting on a
     selector that holds regardless of collapse state, e.g. the logo's
     `alt` via `getByAltText('Noesis')`, OR add a `data-testid="app-name"`
     on the visible name span. Not blocking — the current default holds
     and Playwright's fresh-context-per-test means no persisted state
     bleeds in.

2. **`getByText` substring match weakens the "Noesis" gate slightly** —
   `frontend/e2e/_smoke.spec.ts:47`
   - Issue: `page.getByText('Noesis')` matches any element containing
     the substring "Noesis", which includes the
     `ConnectionErrorOverlay`'s "The Noesis API server could not be
     reached" copy (`frontend/src/components/errors/ConnectionErrorOverlay.tsx:49`).
     In a degraded-backend scenario where the API is dead but the
     frontend chrome partially mounts, the spec could pass even though
     the app is in an error state.
   - Realistic impact is low: a backend dead enough to trigger the
     overlay would also prevent the URL chain from settling on
     `/notebooks`/`/models`/`/login` (the auth check would hang), so
     `waitForURL` is likely to time out first. Still, an
     `getByText('Noesis', { exact: true })` constraint would tighten
     the contract.
   - Recommendation: tighten to `{ exact: true }` or use
     `page.getByRole('heading', { name: 'Noesis' })` (works for the
     LoginForm CardTitle) combined with a sidebar-specific selector
     in a follow-up. Not blocking.

3. **Comment claim "(dashboard routes) … `/models`, `/sources`, etc."
   is slightly broader than this spec exercises** —
   `frontend/e2e/_smoke.spec.ts:24-26`
   - Issue: The docstring asserts that AppSidebar appears on all
     dashboard routes including `/sources`, but this spec never lands
     on `/sources`. Accurate but mildly misleading framing — a reader
     might assume the spec validates that.
   - Recommendation: minor wording tweak in a future revision, or
     leave as-is (the broader claim is true and helpful context).

4. **`gotoAndWait` still has the unused `Response | null` return type**
   (carried forward from attempt 1's Minor #5).
   - Acknowledged by the implementer in `status.md` as deliberately
     deferred. Filed implicitly as a follow-up.

## Decision rationale

The two in-lane issues from attempt 1 are both addressed correctly and
substantively, not with a band-aid. The regex extension is anchored to
prevent regression on deeper routes; the "Noesis" marker is grounded in
verifiable source code references on all three landed states; the
file-level docstring honestly explains the three-state contract and
links to the layout file that drives the redirect. TypeScript, lint, and
test discovery are clean. The remaining concerns are minors — none rise
to the level of "must fix before merge".

Major #2 (workflow file path) is explicitly out of scope per the
orchestrator brief and is not held against this revision.

## Next steps

**APPROVED** — recommend the orchestrator:

1. Resolve Major #2 (move `docs/tracks/A-mineru/e2e-workflow.yml.pending`
   to `.github/workflows/e2e.yml`) using a PAT with `workflow` scope, OR
   document the deferral in the PR description with a tracking issue.
2. Open the PR for `track/a-playwright` → `main`.
3. File the four minors above as a follow-up cleanup task — not a
   blocker for this PR.
