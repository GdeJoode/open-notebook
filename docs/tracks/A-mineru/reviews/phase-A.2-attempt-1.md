# Review — Track A Phase A.2 attempt 1

**Branch**: `track/a-mineru-ui`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-04

## Summary

Phase A.2 ships a clean, well-tested MinerU UI integration. All ten acceptance criteria are met; the backend health endpoint is bulletproof against transport failures; the badge correctly returns `null` for legacy docling; the per-source override Select uses a `"default"` sentinel that maps to `undefined` (cleaner than the `""` discussed in the prompt and immune to Radix's empty-string-value invariant); React Query polling is configured exactly as specified (`refetchInterval: 30_000`, `retry: false`, default `refetchIntervalInBackground: false`). Quality gates all pass: 22 backend pytest, 8/8 Playwright track-a specs, `tsc --noEmit` clean, `npm run lint` produces only pre-existing warnings. A small set of minor cosmetic concerns is noted but none block merge.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Settings dropdown renders four engines; slider toggles in auto mode | ✅ | `SettingsForm.tsx:211-214` (four options); `:219` (`watchedEngine === 'auto'` gate). Verified in `parser-engine-settings.spec.ts`. |
| 2 | Slider value round-trips via PATCH/PUT + reload | ✅ | PUT body asserted (`parser-engine-settings.spec.ts:117`); post-reload slider visibility asserted (line 122-123). See minor #1 — explicit post-reload value not asserted. |
| 3 | ParserEngineBadge four states; docling returns `null` | ✅ | `ParserEngineBadge.tsx:72-74` (truly returns `null`, NOT a "subtle gray Docling badge"). All four states covered in `parser-engine-badge.spec.ts`. |
| 4 | Health chip polls every 30s | ✅ | `use-mineru-health.ts:25` (`refetchInterval: 30_000`); `refetchIntervalInBackground` not set (defaults to `false`); `retry: false` (line 27). Online→offline transition verified within 40s in `mineru-health-chip.spec.ts:60-62`. |
| 5 | `GET /api/health/mineru` never returns 5xx | ✅ | `health.py:54-64` wraps client call in `try / except Exception`; backed by `MineruHttpClient.health_check()` which itself has belt-and-braces (`except httpx.HTTPError` + `except Exception`) in `mineru_http_client.py:218-236`. Endpoint-never-raises path covered by `test_endpoint_never_raises_even_when_client_raises` (`test_health_router.py:61-74`). |
| 6 | Per-source override Select sends `parser_engine` in body | ✅ | `PipelineConfigPanel.tsx:82-100`. Sentinel `"default"` maps to `undefined` (line 85). axios drops undefined keys → omitted from JSON. Spec `parser-engine-override.spec.ts:146` asserts `{ parser_engine: 'mineru' }` in body. |
| 7 | Playwright specs fully route-mocked (no live backend) | ✅ | All 4 specs route-mock `/api/auth/status`, `/api/models/status`, `/api/config`, `/api/settings`, `/api/sources/*`, `/api/health/mineru`. Helpers in `_track-a-helpers.ts` pre-seed `auth-storage` localStorage to dodge the login redirect race. |
| 8 | A11y: aria-label on form controls, role="status" + aria-live on chip | ✅ | Trigger `aria-label="Document processing engine"` (`SettingsForm.tsx:207`); slider `aria-label="Docling confidence threshold"` (line 236); override Select `aria-label="Parser engine for this run"` (`PipelineConfigPanel.tsx:88`); chip `role="status"` + `aria-live="polite"` (all three states in `MineruServiceHealthChip.tsx`). |
| 9 | Type safety: `Source.metadata: SourceMetadata`; no `any` | ✅ | `api.ts:59-72` defines `SourceMetadata`; `:74-78` types `SourceDetailResponse.metadata?`; `:81-85` exports `MineruHealthResponse` (used by hook + client). `git diff` shows zero `any` usage in added lines. |
| 10 | No regression in Settings sub-sections; SourceDetailContent diff minimal | ✅ | `SettingsForm.tsx` diff (112 lines total) is entirely contained in the Document Processing Engine card. `SourceDetailContent.tsx` diff is exactly 1 import + 4-line JSX insertion at line 554. |

## Test status

**Backend (pytest)** — 22 passed in 99.7s:

```
tests/test_mineru_http_client.py ... 18 passed (incl. 6 new health_check tests)
tests/test_health_router.py ... 4 passed (TestMineruHealthEndpoint::*)
========== 22 passed in 99.72s ==========
```

`test_health_check_connection_error_returns_unhealthy` simulates `httpx.ConnectError` (not a happy-path mock — exercises the real exception branch). `test_health_check_5xx_returns_unhealthy_with_error` covers the non-2xx branch. Endpoint-level `test_endpoint_never_raises_even_when_client_raises` confirms 200 is returned even when the client raises `RuntimeError`.

**Frontend type-check + lint**:

```
$ npx tsc --noEmit
(no output — clean)

$ npm run lint
(no errors; only pre-existing warnings: DEFAULT_PIPELINE_CONFIG unused in PipelineConfigPanel.tsx, Network/Play unused in SourceDetailContent.tsx, etc.)
```

**Playwright e2e/track-a** — 8/8 passed in 48.6s against `next dev` on port 3100:

```
8 passed (48.6s)
  parser-engine-badge.spec.ts (4 sub-tests)
  mineru-health-chip.spec.ts (2 sub-tests)
  parser-engine-override.spec.ts (1 sub-test)
  parser-engine-settings.spec.ts (1 sub-test)
```

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional, file as follow-up)

1. **`parser-engine-settings.spec.ts` does not assert the exact post-reload slider value** — `frontend/e2e/track-a/parser-engine-settings.spec.ts:121-123`.
   - Currently asserts (a) PUT body has `docling_min_confidence < 0.95` (line 117) and (b) slider is visible after reload (line 123). Visibility implies `parser_engine === 'auto'` was restored. The numeric value, however, is not explicitly verified post-reload (e.g. via `aria-valuenow`).
   - The mock state object (`currentSettings`) merges the PUT body, then GET returns it — so the round-trip *is* exercised end-to-end, just not pin-pointed in the final assertion. Adding `expect(sliderThumb).toHaveAttribute('aria-valuenow', String(currentSettings.docling_min_confidence))` after the reload would close the loop.

2. **Dead defensive click in override spec** — `frontend/e2e/track-a/parser-engine-override.spec.ts:116`.
   - `await page.getByRole('button', { name: '' }).nth(1).click().catch(() => {})` is a no-op catch-all that was clearly left in during iteration. Remove it; the real menu open at lines 119-121 (via `aria-haspopup="menu"`) handles the case.

3. **Health-check unit tests don't cover `httpx.TimeoutException`/`ReadTimeout` explicitly** — `apps/app-main/tests/test_mineru_http_client.py:437-453`.
   - `test_health_check_connection_error_returns_unhealthy` covers `httpx.ConnectError`. Adding an analogous test using `httpx.ReadTimeout` (or `httpx.TimeoutException`) would close the timeout-specific coverage gap explicitly. The `except httpx.HTTPError` clause in `mineru_http_client.py:233` already catches both because `httpx.TimeoutException` derives from `httpx.HTTPError`, so this is documentation/coverage rigour, not a behavioural gap.

4. **Pre-existing dangling `htmlFor="doc_engine"` in `SettingsForm.tsx:194`** — not introduced by A.2. The Select trigger correctly carries `aria-label="Document processing engine"`, so accessibility is preserved. Worth filing as a tidy-up.

5. **First-load skeleton in `MineruServiceHealthChip.tsx:30` may never show** — when React Query mounts with no cached data, `isLoading: true` flips to `false` once the request resolves, but the initial render may already include `data` (if SSR-prefetched) or pop straight to `online`/`offline`. The spec `parser-engine-badge.spec.ts: 'chip starts in checking state then resolves online'` deliberately introduces a 300ms delay to catch the skeleton, then explicitly notes "we may catch either the checking or online state depending on exact timing". Net effect: the skeleton path exists in code but is only weakly verified. Not a defect; would benefit from a deterministic test if it matters for QA.

## Decision rationale

Every numbered acceptance criterion is satisfied; every attack vector in the review prompt was probed and found sound:

- `/api/health/mineru` is bulletproofed at TWO levels (broad `except Exception` in the router + in the client) and is verified to remain 200 even when the client raises `RuntimeError`.
- `MineruHealthResult` is a frozen dataclass; `test_mineru_health_result_is_frozen_dataclass` verifies immutability.
- `ParserEngineBadge` returns `null` for legacy docling success — confirmed both by reading the component (lines 72-74) and by `parser-engine-badge.spec.ts:128` asserting `toHaveCount(0)`.
- The settings slider is gated by `watchedEngine === 'auto'` (an explicit conditional, not just `disabled`).
- Per-source override uses `"default"` sentinel → `undefined` (axios omits undefined keys). The `""` empty-string concern from the prompt does not apply.
- React Query: `refetchInterval: 30_000`, `retry: false`, `refetchIntervalInBackground` not set (correctly defaults to false).
- Specs are fully route-mocked; no real backend dependency.
- A11y: every new control has either `aria-label` or `<Label htmlFor>` (slider uses aria-label; the htmlFor target on the Select label is dangling but pre-existing).
- No `any`; types correctly exported; `SourceMetadata` typed.
- No regression in Settings sub-cards; `SourceDetailContent.tsx` diff is precisely +1 import + 4-line JSX insertion.

The five minor items above are cosmetic / follow-up improvements; they do not affect correctness, accessibility, or the documented contract.

## Next steps

APPROVED — ready for human approval / merge.

Suggested follow-ups (file as A.3 backlog items, do not block this PR):
- Add explicit `aria-valuenow` post-reload assertion in `parser-engine-settings.spec.ts`.
- Drop the dead defensive click in `parser-engine-override.spec.ts:116`.
- Add an `httpx.ReadTimeout` case to `test_mineru_http_client.py` for completeness.
- Fix dangling `htmlFor="doc_engine"` in `SettingsForm.tsx:194` (pre-existing).

## Kudos

- The `MineruHealthResult` frozen dataclass with a dedicated immutability test is exactly the right level of rigour for a status type the UI depends on.
- The double-catch (HTTPError + bare Exception) in `MineruHttpClient.health_check` plus the third broad catch in the router is genuinely belt-and-braces; very low chance of a 5xx ever leaking to the chip.
- The `"default"` sentinel in `PipelineConfigPanel` (vs the `""` discussed in the spec) is a better choice — Radix Select rejects empty-string values on `<SelectItem>`, and this approach side-steps that entirely while preserving the "omit on save" semantic.
- The shared `_track-a-helpers.ts` + `addInitScript` localStorage pre-seed avoids the documented `/login` redirect race elegantly; future tracks should reuse this pattern.
- Self-review explicitly calls out the three gotchas (stale Docker bundle, strict-mode collision, dual menuitem/button match) — useful future-me documentation.
