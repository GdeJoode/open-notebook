# Track A — Implementation status

Living document. Implementers append a section per phase as work lands.
Reviewers/orchestrator consult this before approving PRs and before kicking
off the next phase.

---

## Phase A.0 — IMPLEMENTED (2026-06-03)

**Branch**: `track/a-playwright` (pushed to origin; PR creation is up to the orchestrator)

**Commits** (3, oldest → newest):

| Hash | Message |
|------|---------|
| `670cfa1` | `chore(frontend): add Playwright config + scripts + devDep (A.0)` |
| `444e86e` | `test(frontend): smoke E2E spec + shared helper skeleton (A.0)` |
| `da3b9e3` | `ci: add Playwright E2E workflow for PRs touching frontend/apps/services (A.0)` |

(Plus a docs-only commit once this status file + README pointer land.)

### What was done

**Created**:

- `frontend/playwright.config.ts` — Chromium-only, retry-once-on-CI, workers=cpu/2
  locally / 2 on CI, 30s test timeout, `baseURL` = `http://localhost:8502`
  (overridable via `PLAYWRIGHT_BASE_URL`), trace+screenshot+video retained
  on failure only. `testMatch` excludes files prefixed with `_` so the
  helper module isn't treated as a spec.
- `frontend/e2e/_helpers.ts` — minimal shared utilities (`gotoAndWait`,
  `isOnLoginPage`). Deliberately small: future helpers are added by the
  phase that first needs them, not preemptively.
- `frontend/e2e/_smoke.spec.ts` — single spec that hits `/`, waits for the
  redirect chain to settle at `/notebooks` (auth-off) or `/login` (auth-on),
  and asserts the appropriate UI is visible. Doubles as install-validation.
- `frontend/e2e/README.md` — local setup, running, debugging, reports,
  authoring new specs, CI pointer.
- `.github/workflows/e2e.yml` — runs on PRs touching `frontend/`,
  `apps/app-main/`, `services/`, `packages/`, the workflow itself, or
  `Dockerfile` / `docker-compose.yml`. Concurrency-cancels in-flight
  runs on the same ref. Caches npm + Playwright browsers. Brings up
  GPU-free services (`surrealdb` + `open_notebook`), polls
  `http://localhost:8502` for readiness, then runs `npm run e2e`.
  Uploads the HTML report (always) and traces + compose logs (on failure).

**Modified**:

- `frontend/package.json` — added `@playwright/test` (`^1.50.0`, npm
  resolved to 1.60.0) as a devDependency and 5 scripts:
  `e2e`, `e2e:headed`, `e2e:debug`, `e2e:install`, `e2e:report`.
- `frontend/.gitignore` — added `/playwright-report/`, `/test-results/`,
  `/playwright/.cache/`.
- `README.md` — added a pointer to `frontend/e2e/README.md` under
  "For Contributors".

### Acceptance criteria status

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | `cd frontend && npm install && npm run e2e:install` succeeds | **Partial** | `npm install` validated locally (`@playwright/test 1.60.0` resolved). `npm run e2e:install` not executed in this sandbox (would download Chromium + sudo for system deps). CI exercises the equivalent step on every PR. |
| 2 | `npm run e2e` runs the smoke spec and passes against `http://localhost:8502` | **Not validated locally** | Docker-compose stack is not guaranteed to be up in this sandbox. The CI workflow validates this on PR. Local validation: `npx playwright test --list` succeeds and discovers exactly the smoke spec, proving config + spec are well-formed. |
| 3 | Failed test produces an HTML report under `playwright-report/` | **Configured** | `reporter: ['html', { open: 'never' }]` in `playwright.config.ts`. Cannot be exercised without a live failure, but the config is the canonical Playwright pattern. |
| 4 | CI workflow runs Playwright on a PR touching `frontend/` | **Implemented** | `.github/workflows/e2e.yml`. Will be exercised by the PR that lands this branch (which itself touches `frontend/`). |
| 5 | README documents how to run locally | **Done** | `frontend/e2e/README.md` (canonical) + top-level `README.md` pointer. |

### What was NOT validated

- **Live smoke run**: `npm run e2e` against a running stack — requires
  `docker compose up open_notebook` which takes ~5+ min for the first
  image build in this environment. Deferred to CI / orchestrator
  verification on PR.
- **Browser download**: `npm run e2e:install` not executed here. The
  CI step `npx playwright install --with-deps chromium` covers it
  the first time the workflow runs (cached afterwards).

### Caveats & follow-ups

1. **CI-only GPU-free services**: The workflow deliberately does NOT
   bring up `docling`, `mineru`, `whisperx`, `extraction`, `summarization`
   — these need the nvidia container runtime and won't start on
   `ubuntu-latest`. Phase A.2 specs that touch GPU-backed flows (parser
   engine badge, mineru health chip) MUST mock the corresponding HTTP
   clients at the API boundary, or be tagged for skip-on-CI. Document
   the policy explicitly in those specs.

2. **First CI run will be slow**: Building the `open_notebook` image
   from scratch is ~5-10 min on GitHub runners. Subsequent runs benefit
   from buildx layer caching and the npm/Playwright caches. If this
   becomes painful, consider building once in a separate job and
   passing the image via `docker save` / `docker load` between jobs.

3. **Playwright version**: `@playwright/test` is pinned with `^1.50.0`
   to allow patch+minor updates; `npm install` resolved to 1.60.0 today.
   Bump deliberately via a follow-up PR if a new minor breaks anything.

4. **Auth in CI**: The smoke spec tolerates both auth-on and auth-off
   stacks. CI runs without `OPEN_NOTEBOOK_PASSWORD`, so the assertion
   path through `/notebooks` is what executes by default. If a future
   spec requires authenticated state, add a `setupAuth` helper to
   `_helpers.ts` and a global setup file rather than logging in inside
   every spec.

5. **PR creation**: This branch is pushed but no PR is opened — that's
   the orchestrator's responsibility after review.
