# Track A — Escalations

## 2026-06-03 — Phase A.0: workflow-scope PAT required for `.github/workflows/`

**Phase**: A.0 (Playwright E2E test infrastructure)
**Severity**: Low (does not block merge; one-line follow-up for orchestrator)
**Status**: Open — needs orchestrator action

### Issue

The agent's PAT (active scopes: `gist`, `read:org`, `repo`) cannot create or
update files under `.github/workflows/` via:

- `git push` → server-side hook rejects: `refusing to allow a Personal Access
  Token to create or update workflow .github/workflows/e2e.yml without
  workflow scope`
- `PUT /repos/{owner}/{repo}/contents/.github/workflows/...` → returns 404
  (GitHub deliberately obscures missing-scope errors as 404 on this endpoint)

### Workaround applied

The full `e2e.yml` content is committed at
`docs/tracks/A-mineru/e2e-workflow.yml.pending` on branch `track/a-playwright`.
The file is byte-identical to its intended canonical location.

### Required action

When opening the PR (or right before), with a workflow-scoped token:

```bash
git checkout track/a-playwright
git pull
git mv docs/tracks/A-mineru/e2e-workflow.yml.pending .github/workflows/e2e.yml
git commit -m "ci: install Playwright E2E workflow at canonical path (A.0)"
git push
```

Acceptance criterion #4 ("CI workflow runs Playwright on a PR touching
`frontend/`") completes only after this move. The first PR that touches
`frontend/` after the move will trigger the workflow.

### Prevention

Future tracks needing `.github/workflows/` changes should either:

- Run the implementer with a PAT that has `workflow` scope, or
- Apply the same staging pattern (commit content under a docs path, move at PR-creation time).
