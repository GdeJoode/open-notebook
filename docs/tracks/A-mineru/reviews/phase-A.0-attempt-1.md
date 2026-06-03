# Review — Track A Phase A.0 attempt 1

**Branch**: `track/a-playwright`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

The Playwright wiring (config, scripts, helpers, docs, gitignore, README pointer)
is solid, minimal, and free of YAGNI overgrowth. TypeScript compiles cleanly,
the spec is discovered correctly, and the staged CI workflow is well-formed.
However, the smoke spec's URL assertion is **not robust against the exact
configuration the CI workflow produces** (empty `docker.env` → unconfigured
models → dashboard layout redirects `/notebooks` → `/models?setup_required=true`).
This is a real defect that will make CI red on the first PR that touches
`frontend/`, breaking acceptance criterion #2 in practice.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `cd frontend && npm install && npm run e2e:install` succeeds | ✅ | `npm install` validated; `e2e:install` script wires to `playwright install --with-deps chromium`, canonical. |
| 2 | `npm run e2e` runs the smoke spec and passes against `http://localhost:8502` | ⚠️ | Pass is **conditional on models being configured**. In the exact CI config the implementer wrote (empty `docker.env`, no API keys), the spec will fail. See Blocker #1. |
| 3 | Failed test produces a Playwright HTML report under `playwright-report/` | ✅ | `reporter: ['html', { open: 'never' }]` set; `playwright-report/` gitignored; workflow uploads it as `if: always()` artifact. |
| 4 | CI workflow runs Playwright on a PR touching `frontend/` | ⚠️ | Workflow file is staged at `docs/tracks/A-mineru/e2e-workflow.yml.pending`, not installed at `.github/workflows/e2e.yml`. Implementer flagged this as a tooling-scope escalation (acceptable per orchestrator brief), but criterion is **not satisfied** until the file is moved. |
| 5 | README documents how to run locally | ✅ | `frontend/e2e/README.md` covers install, run, debug, codegen, debugging flakes, CI; top-level `README.md` adds a pointer. |

## Test status

```
$ cd frontend && npx tsc --noEmit
(no output — clean)

$ cd frontend && npx playwright test --list
Listing tests:
  [chromium] › _smoke.spec.ts:16:7 › smoke › dashboard root loads (auth-on or auth-off)
Total: 1 test in 1 file

$ cd frontend && npx next lint
(unchanged from main — no new warnings introduced by e2e/_smoke.spec.ts or e2e/_helpers.ts)
```

Test discovery and TypeScript both pass. **Live execution** of the spec against
a running stack was not performed in this sandbox (matches what status.md
admits); the static checks alone do not prove acceptance #2.

## Issues found

### 🔴 Blockers (must fix)

1. **Smoke spec URL regex misses the dashboard layout's `/models?setup_required=true` redirect** — `frontend/e2e/_smoke.spec.ts:21`
   - Issue: The spec assumes the redirect chain settles at `/notebooks` (auth-off) or `/login` (auth-on). It misses a **third** redirect performed client-side by `frontend/src/app/(dashboard)/layout.tsx:43-50`: when the user is authenticated (or auth is disabled) but `useModelStatus()` returns `valid: false`, the layout pushes to `/models?setup_required=true`.
   - The CI workflow at `docs/tracks/A-mineru/e2e-workflow.yml.pending:74-75` does `touch docker.env`, producing a stack with **no LLM API keys and no default model defaults**. `apps/app-main/src/app_main/api/routers/models.py:118-264` returns `valid=False` whenever `default_chat_model` or `default_embedding_model` is missing (line 208-215), which is exactly the empty-DB state on first CI boot. Consequently the layout will redirect away from `/notebooks` to `/models?setup_required=true`.
   - Even when `waitForURL(/\/(notebooks|login)/)` resolves momentarily on the brief `/notebooks` flash, the subsequent `await expect(page).toHaveURL(/\/notebooks/)` (line 30) has a 5s expect-timeout polling re-check and will see the URL has moved to `/models?...` once `useModelStatus` resolves. The test outcome is timing-dependent at best and a deterministic CI failure at worst.
   - Impact: Acceptance criterion #2 ("smoke spec passes against running app") is not met in the CI configuration the implementer themselves authored.
   - Recommendation (state, do not implement): extend the URL regex to accept `/models` as a valid terminal state, OR seed default models in the workflow before running the spec, OR assert at the network layer (e.g. `expect(navigationResponse).toBeOK()`) rather than coupling to the dashboard layout's auth/setup gate.

### 🟡 Major (must fix)

1. **Vapid post-redirect assertion in no-auth branch** — `frontend/e2e/_smoke.spec.ts:31`
   - Issue: `await expect(page.locator('body')).toBeVisible()` is true on *any* rendered HTML, including a Next.js runtime-error overlay, a 500 page, or the `/models?setup_required=true` page. The spec comment explicitly justifies this as "proving the route mounted (no Next.js error overlay) is enough" — but the assertion does not actually check for the absence of an error overlay. It is a false-positive contract.
   - Impact: The smoke test claims to detect "the dashboard rendered" but cannot detect a broken `/notebooks` page (e.g. unhandled hydration error rendering only an error overlay). It is effectively a connectivity probe with a URL check, not a smoke test of the dashboard route.
   - Recommendation: assert on something that uniquely identifies the notebooks dashboard rendered correctly (a role/heading/landmark from that page, mirroring the `getByText('Noesis')` pattern used in the auth branch). The reviewer is not prescribing the exact element — but `body.toBeVisible()` is below the bar for a smoke test.

2. **Workflow file not at canonical path; acceptance criterion #4 unmet** — `docs/tracks/A-mineru/e2e-workflow.yml.pending`
   - Issue: Per the orchestrator-supplied brief, the missing-`workflow` PAT scope is a known tooling limitation, not a code-quality issue. The escalation note and the documented `git mv` recipe are correct and complete. **However**, the phase plan's acceptance criterion #4 is literally "CI workflow runs Playwright on a PR touching `frontend/`" — that does not happen while the file lives outside `.github/workflows/`. This must be resolved by the orchestrator before A.0 can be claimed Done.
   - Severity is "major" only in the sense that acceptance is gated on it. The work product itself is correct.
   - Recommendation: orchestrator must execute the `git mv` + push with a workflow-scoped PAT before merging or before opening the first PR that should trigger the workflow.

### 🔵 Minor (optional)

1. **CI runs `playwright install --with-deps` unconditionally even on cache hits** — `docs/tracks/A-mineru/e2e-workflow.yml.pending:67-69`
   - Issue: The `Install Playwright Chromium` step has no `if: steps.playwright-cache.outputs.cache-hit != 'true'` guard. The browser-binary download itself is skipped by Playwright when binaries already exist in `~/.cache/ms-playwright`, but `--with-deps` still triggers an `apt-get install` for system libs on every run. Minor wallclock cost (~10-20s).
   - Recommendation: consider gating browser install behind the cache-hit boolean, leaving a separate non-`--with-deps` install for the cached case if system libs are already present on `ubuntu-latest` (they largely are). Not a blocker.

2. **GitHub Actions workflow lacks explicit `permissions:` block** — `docs/tracks/A-mineru/e2e-workflow.yml.pending`
   - Issue: No top-level `permissions:` declaration. Repos with the default-permissive `GITHUB_TOKEN` scope get write-contents implicitly. For a read-only test workflow (checkout + docker + npm), `permissions: { contents: read }` would follow least-privilege.
   - Recommendation: add `permissions: { contents: read }` at the top level. Doesn't change behaviour for this workflow but reduces blast radius if a step is ever compromised.

3. **`copy_original` and other compose dependencies not explicitly waited on** — `docs/tracks/A-mineru/e2e-workflow.yml.pending:80-93`
   - Issue: The wait loop polls `http://localhost:8502` but does not check `surrealdb` health independently. If SurrealDB is slow to come up, the open_notebook container will still respond on :8502 (returning 5xx from any DB-backed endpoint). The smoke test never hits a DB-backed endpoint, so this is fine **today** — but the moment a track-a spec assumes the DB is up, the workflow will need a separate `surrealdb` readiness probe.
   - Recommendation: note in `frontend/e2e/README.md` "CI policy" section that future specs touching DB-backed pages must add their own readiness step or seed step. Not blocking.

4. **`copy_original` workflow `Tear down` runs `docker compose down -v`** — `docs/tracks/A-mineru/e2e-workflow.yml.pending:124-125`
   - Issue: `-v` removes named volumes including any `notebook_data` data. Correct for CI ephemerality, but worth documenting why (a future contributor might be tempted to remove `-v` to "preserve cache" and then accidentally bleed state between runs).
   - Recommendation: one-line comment above the step. Not blocking.

5. **`gotoAndWait` returned `Response | null` is not used by the smoke spec** — `frontend/e2e/_helpers.ts:19-21` + `frontend/e2e/_smoke.spec.ts:17`
   - Issue: Helper returns a Response or null; spec discards it. Either drop the return type (helper becomes `Promise<void>`) or actually use it for an `expect(response.ok()).toBeTruthy()` style assertion. Currently it's dead surface area.
   - Recommendation: drop the return type, keeping the helper minimal per the implementer's own stated principle in `_helpers.ts:7-10`.

## Decision rationale

One blocker (smoke spec will deterministically fail in the CI configuration the
implementer authored) and two majors (vapid post-redirect assertion;
acceptance #4 unmet pending workflow-file move). The minors are quality
nits and can be filed as follow-ups.

The implementer's broader execution was high-quality: minimal helper module
matching their own stated principle, well-commented config, sensible CI
structure, honest status.md including a "What was NOT validated" section,
and an upfront escalation for the PAT scope. The blocker is a specific
oversight in the smoke spec's redirect-chain assumptions, not a systemic
quality issue.

## Next steps

REVISIONS_NEEDED: the implementer should:
1. Fix the URL regex / assertion strategy in `frontend/e2e/_smoke.spec.ts` so the smoke spec is robust against the empty-config CI stack the workflow produces (Blocker #1). Re-validate by running the smoke spec against `docker compose up surrealdb open_notebook` with an **empty** `docker.env`.
2. Replace the `body.toBeVisible()` assertion in the no-auth branch with something that meaningfully proves the notebooks dashboard route mounted (Major #1).
3. Orchestrator: execute the documented `git mv` for the workflow file (Major #2) — this is the gate for acceptance #4.

Minors can be addressed in this revision or filed as follow-ups; either is fine.
