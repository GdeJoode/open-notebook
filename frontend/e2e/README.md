# Frontend E2E tests (Playwright)

End-to-end tests that drive a real browser against a running Open Notebook stack.
The runner is [Playwright](https://playwright.dev/) — Chromium only for now
(Firefox/WebKit can be added later by extending `projects` in
`frontend/playwright.config.ts`).

## When to write an E2E test

Reach for E2E when the behaviour spans **frontend + API + DB** and a unit/integration
test cannot prove it (e.g. settings persist after reload, a job-status badge flips
when a backend job completes, a graceful-degradation flow when a service is down).

For pure component logic, prefer a future React component test (not set up yet) or
a unit test inside the affected module. E2E is the slowest and flakiest layer —
keep specs small and focused.

## One-time setup

From the repo root:

```bash
cd frontend
npm install            # installs @playwright/test
npm run e2e:install    # downloads Chromium + OS deps
```

`e2e:install` runs `playwright install --with-deps chromium` — on Linux it
may prompt for sudo to install the browser's system libs. On macOS/Windows
it's a plain download.

## Running locally

The default `baseURL` is `http://localhost:8502`, matching the Next.js
frontend port in `docker-compose.yml`. Start the stack first:

```bash
# From repo root — bring up at least surrealdb + open_notebook
docker compose up -d surrealdb open_notebook
```

Then, from `frontend/`:

```bash
npm run e2e             # headless, parallel
npm run e2e:headed      # see the browser window
npm run e2e:debug       # step through with the Playwright Inspector
npm run e2e:report      # open the last HTML report
```

Override the target stack:

```bash
PLAYWRIGHT_BASE_URL=http://my-host:8502 npm run e2e
```

## Reports & artefacts

A failed run writes:

- `frontend/playwright-report/` — HTML report (open with `npm run e2e:report`)
- `frontend/test-results/` — per-test traces, screenshots, videos

All three directories are gitignored. On CI they are uploaded as workflow
artefacts so you can download and open the HTML report locally.

## Authoring a new spec

1. Create `frontend/e2e/<feature>.spec.ts` (or group under
   `frontend/e2e/<track>/<feature>.spec.ts` for multi-spec tracks — e.g.
   `e2e/track-a/parser-engine-settings.spec.ts`).
2. Files starting with `_` (like `_helpers.ts`) are treated as modules,
   not specs — see `testMatch` in the config.
3. Import shared utilities from `./_helpers`. Add new helpers there
   **only** when a spec actually needs them.
4. Prefer **role/label/text** selectors (`page.getByRole`, `getByLabel`,
   `getByText`) over CSS selectors. They survive refactors and double as a11y
   assertions.
5. Keep each spec runnable in isolation. If a test depends on backend state
   (a notebook, a source), seed it via the API inside the test — not by
   relying on data left over from another spec.

## Recording a flow

```bash
cd frontend
npx playwright codegen http://localhost:8502
```

Click through the app; Playwright emits the corresponding code. Useful as a
starting point — but always trim auto-generated selectors down to role/label
based ones before committing.

## Debugging a flake

1. Re-run with the inspector: `npm run e2e:debug -- -g "<test name>"`.
2. Open the failure report: `npm run e2e:report` → view trace.
3. Common culprits in this codebase:
   - Backend not fully ready — increase `actionTimeout` for the specific
     test or add a `page.waitForResponse(...)` for the API call you depend on.
   - Auth state — see `_helpers.ts::isOnLoginPage`; auth-on vs auth-off stacks
     need different assertions.
   - Hot-reloading dev server — `next dev` can race with assertions; run
     `npm run build && npm run start` for stable runs.

## CI

`.github/workflows/e2e.yml` runs the suite on PRs that touch
`frontend/`, `apps/app-main/`, or `services/`. It spins up the docker-compose
stack (sans GPU services) before invoking `npm run e2e`. See that file for
the full configuration.
