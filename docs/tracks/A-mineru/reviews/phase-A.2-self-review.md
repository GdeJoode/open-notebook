# Phase A.2 — Self-review

> Branch: `track/a-mineru-ui` (stacked on `track/a-mineru-fallback`)
> Implementer: Claude Opus 4.7
> Date: 2026-06-04
> Status: ready for reviewer

## Commits landed (7, matching plan-A.2.md §7)

| # | SHA       | Message |
|---|-----------|---------|
| 1 | `fdfb52e` | `feat(app-main): add MineruHttpClient.health_check() + tests` |
| 2 | `47be625` | `feat(api): add GET /api/health/mineru endpoint` |
| 3 | `bd406fe` | `feat(frontend/api): typed health client + Source.metadata types` |
| 4 | `1d1e153` | `feat(frontend/settings): parser_engine dropdown + confidence slider + health chip` |
| 5 | `53cad43` | `feat(frontend/source): ParserEngineBadge + insertion in detail` |
| 6 | `764b4d4` | `feat(frontend/reparse): per-source parser-engine override` |
| 7 | `ef77765` | `test(e2e/track-a): 4 Playwright specs covering A.2 acceptance criteria` |

## Pre-resolved decisions honored

- **Badge unit test → Playwright spec, not vitest.** No new dev
  dependencies; `parser-engine-badge.spec.ts` exercises all four
  visual states with route-mocked `GET /api/sources/{id}` payloads.
- **MinerU red chip does NOT disable the `mineru` engine option.**
  The Settings `<SelectItem value="mineru">` is always enabled; the
  health chip is a passive indicator next to the "Document Processing
  Engine" label.
- **New `health.py` router** at `apps/app-main/src/app_main/api/routers/health.py`.
  Did not extend `services_proxy.py`.

## What was built

### Backend

- `MineruHttpClient.health_check()` — async, 2s default timeout,
  never raises. Connection errors / non-2xx / non-JSON body are
  surfaced as `MineruHealthResult(healthy=False, error=…)`. New
  frozen dataclass `MineruHealthResult` exported via `__all__`.
- `GET /api/health/mineru` — instantiates a fresh `MineruHttpClient`
  per request, calls `health_check()`, returns the dataclass shape
  as JSON. Wraps the client call in a try/except so the route handler
  itself can never 5xx — meaningful for the chip's "offline" state.

### Frontend

- `frontend/src/lib/api/health.ts` — typed `healthApi.mineru()`.
- `frontend/src/lib/hooks/use-mineru-health.ts` —
  `useMineruHealth()` with `refetchInterval: 30_000`,
  `staleTime: 25_000`, `retry: false`.
- `frontend/src/lib/types/api.ts` extended with `SourceMetadata`
  (parser_engine_used, extraction_confidence, …) and
  `MineruHealthResponse`; `SourceDetailResponse.metadata?` typed.
- `SettingsForm.tsx` — four engine options visible, `parser_engine`
  + `docling_min_confidence` round-trip via existing `useSettings`
  hook; `<MineruServiceHealthChip />` mounted next to the label;
  slider only renders in `auto` mode.
- `ParserEngineBadge.tsx` — pure presentational; four `data-state`
  values (`docling-default` returns null, `mineru-manual`,
  `mineru-auto-fallback`, `extraction-failed`); inserted into the
  Processing Status Bar in `SourceDetailContent.tsx`.
- `PipelineConfigPanel.tsx` — new "Parser engine for this run"
  Select at the top of the panel; defaults to `""` → `undefined`
  on save (= use global); also renders the same confidence slider
  when set to `auto`.

### E2E

Four Playwright specs under `frontend/e2e/track-a/`:
- `parser-engine-settings.spec.ts`
- `parser-engine-badge.spec.ts` (4 sub-tests, one per state)
- `mineru-health-chip.spec.ts` (2 sub-tests: online→offline, checking→online)
- `parser-engine-override.spec.ts`

A shared `_track-a-helpers.ts` mocks the dashboard chrome
(`/api/auth/status`, `/api/models/status`, `/api/config`) and seeds
the auth localStorage so specs land on the target page without a
login redirect race.

## Quality gates

| Gate | Result |
|---|---|
| `apps/app-main` pytest (`test_mineru_http_client.py` + `test_health_router.py`) | **22 passed** (18 pre-existing + 6 new health-check tests + 4 new router tests counted together as the file) |
| `frontend` `npx tsc --noEmit` | **clean** |
| `frontend` `npm run lint` | **clean (no errors)** — only pre-existing warnings (e.g. `DEFAULT_PIPELINE_CONFIG` unused in `PipelineConfigPanel.tsx`, `Network` + `Play` icons in `SourceDetailContent.tsx`). No new warnings introduced by A.2. |
| `frontend` Playwright `e2e/track-a/` (against `next dev`) | **8/8 passed** in 35.5s |
| 7 commits on `track/a-mineru-ui` | confirmed via `git log` |
| Branch pushed to origin | done at end (see "Hand-off" below) |

### Backend test breakdown

- `test_mineru_http_client.py` — 18 tests pass (12 pre-existing,
  6 new for `health_check`):
  `test_health_check_returns_healthy_with_version`,
  `test_health_check_200_without_version_still_healthy`,
  `test_health_check_connection_error_returns_unhealthy`,
  `test_health_check_5xx_returns_unhealthy_with_error`,
  `test_health_check_non_json_body_still_healthy_on_200`,
  `test_mineru_health_result_is_frozen_dataclass`.
- `test_health_router.py` — 4 tests (all new):
  `test_returns_healthy_when_service_responds_ok`,
  `test_returns_unhealthy_when_service_returns_unhealthy_result`,
  `test_endpoint_never_raises_even_when_client_raises`,
  `test_returns_healthy_without_version`.

### E2E test breakdown

8 sub-tests across 4 spec files; all pass against a fresh
`npx next dev` Next.js server on port 3100 with backend endpoints
route-mocked.

## Gotchas hit during implementation

1. **Stale Docker frontend bundle**. The first Playwright run targeted
   the Docker-built frontend container at `:8502`, which was built
   from main and lacked the A.2 code (placeholder said "Select
   document processing engine" instead of "Select document parser
   engine"). Fix: started a fresh `next dev` on port 3100 and ran
   the suite against that. CI will exercise the built bundle, so
   this is a local-dev quirk only.
2. **`getByText('Document Processing Engine')` strict-mode collision**.
   The Radix `SelectValue` placeholder is rendered as additional
   text content inside the trigger's accessible tree; the label
   text matched twice. Fix: `{ exact: true }` on the locator. Already
   committed in commit #7.
3. **Reparse modal had two role=button matches for "Reprocess
   Document"** — the menuitem and the modal-footer button. By the
   time the modal is open the menuitem has unmounted, so a plain
   `getByRole('button', { name: /Reprocess Document/i })` is
   unambiguous in practice. Spec uses this pattern.

## Deviations from plan-A.2.md (none material)

- **Badge component test** is a Playwright spec (decided pre-flight).
  Plan §5 open question #1 resolved in the prompt — no vitest added.
- **Help me choose copy update** (plan §3): the existing Collapsible
  copy was edited to describe all four engines (no "follow-up
  release" line). Verified in commit `1d1e153`.
- All other implementation matched plan exactly.

## Risks / things to flag to reviewer

1. **MinerU health-chip relies on the backend container being
   reachable from `app-main`**. If `MINERU_SERVICE_URL` is unset
   the client defaults to `http://mineru:8104`, which fails outside
   docker-compose. The route handler surfaces this as
   `{healthy: false, error: "…"}` — chip shows red, no exception.
   Acceptable for dev; A.3 manual QA checklist should confirm it
   goes green when the compose stack runs.
2. **No vitest setup** means the badge presentational logic relies
   on the e2e spec for coverage. Track B may revisit if more pure-UI
   tests pile up.
3. **Pre-existing lint warnings** in `PipelineConfigPanel.tsx`
   (`DEFAULT_PIPELINE_CONFIG` unused) are not from this PR but are
   visible in the lint output. Left in place to keep the diff
   minimal — separate cleanup PR if desired.

## Manual smoke (deferred to reviewer / A.3)

The plan §6 smoke checklist (keyboard navigation, slider arrows,
container stop/start) was not executed by the implementer — those
steps require a running mineru container with GPU, which is not in
this dev env. Documented in `docs/tracks/A-mineru/MANUAL_QA.md`
already, picked up by A.3.

## Hand-off

- Branch `track/a-mineru-ui` pushed to origin.
- All 7 commits present, in order.
- Ready for review and PR creation.
