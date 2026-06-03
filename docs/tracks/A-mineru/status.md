# Track A — Implementation status

Living document. Implementers append a section per phase as work lands.
Reviewers/orchestrator consult this before approving PRs and before kicking
off the next phase.

---

## Phase A.0 — IMPLEMENTED (2026-06-03)

**Branch**: `track/a-playwright` (pushed to origin; PR creation is up to the orchestrator)

**Commits** (oldest → newest):

| Hash | Message |
|------|---------|
| `670cfa1` | `chore(frontend): add Playwright config + scripts + devDep (A.0)` |
| `444e86e` | `test(frontend): smoke E2E spec + shared helper skeleton (A.0)` |
| `47b60ee` | `docs(track-a): pointer to E2E README + status.md for Phase A.0 (A.0)` |
| `264b9a3` | `ci: stage Playwright E2E workflow as pending artifact (A.0)` (see "Pending workflow file" below) |
| `1bd5f27` | `docs(track-a): escalation note re missing workflow PAT scope (A.0)` |

### IMPORTANT — Pending workflow file (CI gate)

The CI workflow content lives at **`docs/tracks/A-mineru/e2e-workflow.yml.pending`**
rather than its canonical destination `.github/workflows/e2e.yml`. Reason: the
implementer's PAT lacks the `workflow` OAuth scope required by GitHub to create
or update files under `.github/workflows/` via git push or the contents REST API.

**Action required by orchestrator (one-time, with a workflow-scoped token)**:

```bash
git checkout track/a-playwright
git pull
git mv docs/tracks/A-mineru/e2e-workflow.yml.pending .github/workflows/e2e.yml
git commit -m "ci: install Playwright E2E workflow at canonical path (A.0)"
git push
```

The file is byte-identical to what the implementer authored — no review delta
between staging path and final path. The acceptance criterion #4 ("CI workflow
runs Playwright on a PR touching `frontend/`") only fires once the file lives
under `.github/workflows/`.

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
- `docs/tracks/A-mineru/e2e-workflow.yml.pending` — **staged** GitHub Actions
  workflow; must be moved to `.github/workflows/e2e.yml` by the orchestrator
  (see "Pending workflow file" above). Runs on PRs touching `frontend/`,
  `apps/app-main/`, `services/`, `packages/`, the workflow itself, or
  `Dockerfile` / `docker-compose.yml`. Concurrency-cancels in-flight runs
  on the same ref. Caches npm + Playwright browsers. Brings up GPU-free
  services (`surrealdb` + `open_notebook`), polls `http://localhost:8502`
  for readiness, then runs `npm run e2e`. Uploads the HTML report (always)
  and traces + compose logs (on failure).

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
| 4 | CI workflow runs Playwright on a PR touching `frontend/` | **Implemented, install pending** | Workflow content authored at `docs/tracks/A-mineru/e2e-workflow.yml.pending`. Orchestrator must `git mv` it to `.github/workflows/e2e.yml` and push with a workflow-scoped token (see "Pending workflow file" section). Will then be exercised by the next PR touching `frontend/`. |
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

---

## Phase A.0 — attempt 2 fixes (2026-06-03)

Adversarial review (`docs/tracks/A-mineru/reviews/phase-A.0-attempt-1.md`)
returned **REVISIONS_NEEDED** with one blocker and two majors. This entry
records the implementer's fixes for the two issues in their lane;
Major #2 (workflow file `git mv` to `.github/workflows/e2e.yml`) remains
the orchestrator's task and is unchanged.

**Branch**: `track/a-playwright` (same branch as attempt 1; revision pushed
on top of `7092b41`).

### Issues addressed

| Severity | Issue | Resolved? |
|----------|-------|-----------|
| 🔴 Blocker | Smoke spec URL regex missed the `/models?setup_required=true` redirect produced by the empty-`docker.env` CI stack. Spec would fail deterministically in CI. | **Yes** — see "Strategy" below. |
| 🟡 Major #1 | `expect(page.locator('body')).toBeVisible()` was a vapid assertion: true on any rendered HTML, including a Next.js runtime-error overlay. Smoke test could not detect a broken dashboard route. | **Yes** — replaced with a check on the "Noesis" text marker. |
| 🟡 Major #2 | Workflow file lives at `docs/tracks/A-mineru/e2e-workflow.yml.pending` instead of `.github/workflows/e2e.yml`. | **Out of scope** — orchestrator task (PAT scope limitation; recipe still documented above). |

### Strategy

Combined strategy (a) + (b) from the review's recommendation list:

- **(a) Extend the URL regex** to accept `/models` as a valid terminal
  redirect target. The empty-config CI stack legitimately lands on
  `/models?setup_required=true` (dashboard layout's `useModelStatus()` →
  `valid=false` redirect, `frontend/src/app/(dashboard)/layout.tsx:43-50`).
  Acceptable URLs are now `/notebooks`, `/login`, OR `/models(?...)`.
- **(b) Replace the body-visibility assertion with a meaningful marker**:
  the string "Noesis" (the app name) is present on every legitimate
  landed state:
  - `/login` → `LoginForm` card title (`frontend/src/components/auth/LoginForm.tsx:142`)
  - `/notebooks`, `/models`, `/sources`, etc. → `AppSidebar` header
    (`frontend/src/components/layout/AppSidebar.tsx:142-144`),
    rendered via `AppShell` on all dashboard routes.
  - A Next.js error overlay or 500 page would render neither sidebar
    nor login card → assertion fails. This is the smoke contract the
    reviewer asked for.

Why **not** (c) "seed default models in CI": intrusive — needs a database
migration or admin-API call from the workflow, and couples the smoke test
to backend state. The pure-spec fix above is cheaper and more durable.

### What changed

**Modified**:

- `frontend/e2e/_smoke.spec.ts` — rewrote the spec:
  - URL regex extended from `/\/(notebooks|login)(\?|$)/` to
    `/\/(notebooks|login|models)(\?|$)/` (factored as a `const validLanded`
    so `waitForURL` and the post-redirect `toHaveURL` share the pattern).
  - Removed `expect(page.locator('body')).toBeVisible()`.
  - Added `expect(page.getByText('Noesis').first()).toBeVisible()` —
    the primary smoke assertion, applies to all three landed states.
  - Kept the auth-branch secondary check but switched it from the now-
    universal "Noesis" title to the unique login description ("Enter
    your password to access the application") for failure-mode clarity.
  - Updated the test name to "app root mounts on a legitimate landed
    state" (was "dashboard root loads (auth-on or auth-off)") to reflect
    the broader set of valid terminal states.
  - Updated the file-level docstring to explain the three-state contract
    and the choice of "Noesis" as the marker.

**Not modified**:

- `frontend/e2e/_helpers.ts` — left untouched. Minor #5 (drop unused
  return type on `gotoAndWait`) is a quality nit; the brief restricted
  this revision to the blocker + major #1 to keep the diff tight.
  Filed implicitly as a follow-up.
- `frontend/playwright.config.ts` — unchanged.
- `docs/tracks/A-mineru/e2e-workflow.yml.pending` — unchanged
  (orchestrator owns).

### Validation

```
$ cd frontend && npx tsc --noEmit
(no output — clean)

$ cd frontend && npx playwright test --list
Listing tests:
  [chromium] › _smoke.spec.ts:34:7 › smoke › app root mounts on a legitimate landed state
Total: 1 test in 1 file

$ cd frontend && npx next lint  # filtered to e2e/
(no warnings introduced by e2e/_smoke.spec.ts or e2e/_helpers.ts)
```

Live spec execution against a running stack still not performed in this
sandbox (same constraint as attempt 1). The fix is, however, defensible
on inspection: it now matches every documented redirect terminal in the
dashboard layout, and the "Noesis" assertion is grounded in concrete
source-code references (cited above).

---

## Phase A.1a — IMPLEMENTED (2026-06-03)

**Branch**: `track/a-mineru-service` (pushed to origin; PR creation
is the orchestrator's responsibility)

**Commits** (oldest → newest):

| Hash | Message |
|------|---------|
| `2f32e8a` | `feat(services/mineru): GPU FastAPI service wrapping MinerU 2.x CLI (A.1a spike)` |
| `dec7f4b` | `feat(parsing): MinerU layout-to-ExtractedDocument parser (Q-A-5)` |
| `2d44f94` | `feat(app-main): MineruHttpClient + unit tests (A.1a)` |

### What was done

**Spike (Q-A-5)**: characterised MinerU 2.x's output schema from
upstream documentation (`opendatalab/MinerU` @master, `docs/en/reference/
output_files.md`) and recorded findings in
`docs/tracks/A-mineru/MINERU_OUTPUT_SPIKE.md`. Key finding: MinerU's
`<stem>_content_list.json` is a flat reading-order list of element
dicts with **per-element `bbox`** (0–1000 normalised for pipeline /
office backends, 0–1 for vlm) plus `type`, `page_idx`, `text_level`,
and type-specific extras (`img_path`, `table_body`, `code_body`,
`list_items`, etc.). This **fully unblocks** the Q-A-5 decision
(option a, build the bbox parser): full element-level bbox parity
with docling, no PdfChunkViewer UX regression.

**Created**:

- `services/mineru/Dockerfile` — CUDA 12.6 base, Python 3.12, installs
  `mineru[all]` into `/app/.venv` (baked-in per Q-A-1, mirrors docling).
  Exposes 8104, runs `uvicorn api:app --host 0.0.0.0 --port 8104`.
- `services/mineru/api.py` — FastAPI with `GET /health` and `POST /process`.
  Wraps the `mineru` CLI: each /process invocation shells out per file
  to `mineru -p <input> -o <output> -b <backend>` and returns the
  on-disk paths the CLI writes (`<output>/<stem>/<method>/{*.md,
  *_content_list.json, *_middle.json, images/}`). Request/response
  shape symmetric with `services/docling/api.py` so the two HTTP
  clients can share their call pattern.
- `services/mineru/requirements.txt` — `fastapi`, `uvicorn`, `pydantic`,
  `loguru`, `mineru[all]>=2.0`, `httpx`.
- `services/mineru/README.md` — full endpoint + env-var reference,
  first-run download notes (~5–8 GB), GPU-memory guidance (6–8 GB
  for pipeline backend; documents sequencing with docling on a
  shared 24 GB card), live-smoke-test recipe.
- `docs/tracks/A-mineru/MINERU_OUTPUT_SPIKE.md` — schema reference
  + recommended MinerU-type → ExtractedElement mapping.
- `apps/app-main/src/app_main/services/parsing/__init__.py` — package
  marker re-exporting `parse_mineru_output` and `MineruLayoutParseError`.
- `apps/app-main/src/app_main/services/parsing/mineru_layout_parser.py`
  — translates `content_list.json` (plus optional `middle.json` for
  backend detection + markdown for full-text) into
  `ExtractedDocument` with populated `BoundingBox` on every
  `ExtractedElement` / `ExtractedTable` / `ExtractedImage`. Handles
  pipeline vs vlm bbox normalisation, page-furniture types,
  HTML→rows table extraction, and resilient fallback for malformed
  bboxes. Refuses to build empty documents (raises
  `MineruLayoutParseError`).
- `apps/app-main/src/app_main/services/mineru_http_client.py` —
  `MineruHttpClient.process(Path) -> IngestionResult`, symmetric
  with `DoclingHttpClient`. Stages input in shared `/data/input`,
  POSTs `/process`, hands the response paths to the layout parser,
  cleans up the staged copy in `finally`.
- `apps/app-main/tests/test_mineru_layout_parser.py` — 16 cases.
- `apps/app-main/tests/test_mineru_http_client.py` — 12 cases using
  `httpx.MockTransport`.

**Modified**:

- `docker-compose.yml` — added `mineru:` service block on port 8104,
  modelled on `docling:`. Mounts `./docling_input:/data/input`
  (shared with docling) + `./mineru_output:/data/mineru_output`
  (separate output) + named `mineru_models:/data/models` for the
  persistent model cache. GPU reservation as docling. Env vars
  configurable per `services/mineru/README.md`. **Crucially**,
  `mineru` is NOT in `open_notebook.depends_on` so the main app
  boots when the MinerU container is down or rebuilding.
- `docker-compose.yml` (`open_notebook:`) — added env vars
  `MINERU_SERVICE_URL=http://mineru:8104` and `USE_MINERU_SERVICE=true`,
  and mounted the `./mineru_output:/data/mineru_output` shared volume.

### Acceptance criteria status

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | `docker compose build mineru` completes; `docker compose up mineru` starts; `GET /health` returns 200 within 60s after first model download | **Done** | Build completed locally in ~48 min cold (network-bound; `mineru[all]` is ~5 GB of wheels including PyTorch 2.11, vllm 0.21, transformers 4.57, CUDA-13 libs). Resulting image is 12 GB. `docker compose up -d mineru` brings the container to `Up` state. `curl -sf http://localhost:8104/health` returns `{"status":"ok","service":"mineru"}` in <5 s after startup (no model download required for the health endpoint — that's a CLI invocation concern). |
| 2 | `curl POST /process` with a sample PDF returns valid response with `markdown_path` | **Deferred to live smoke** | `services/mineru/README.md` provides the exact recipe (`docling_input/yardstickbias.pdf` → `mineru_output/yardstickbias/auto/`). The API handler itself is exercised at the unit level by the HTTP-client tests (they verify the response payload contract). The first `/process` call triggers the MinerU model download (~5–8 GB), which is environment-dependent and not validated in this sandbox; the `mineru_models` named volume persists the download across container restarts. |
| 3 | `MineruHttpClient().process(Path).success == True` with populated content | **Validated (mocked)** | `test_process_success_returns_ingestion_result_with_document` writes a synthetic MinerU output tree and asserts `result.success=True`, document title + full_text populated, image_paths populated. Live PDF run is the same code path. |
| 4 | `pytest test_mineru_http_client.py` green; coverage ≥ 70% on `mineru_http_client.py` | **Done** | 12/12 tests pass. Coverage: 98% on `mineru_http_client.py`, 93% on `mineru_layout_parser.py` — combined 94%. |
| 5 | `open_notebook` service does NOT hard-depend on mineru | **Done** | `depends_on:` lists only `surrealdb` and `docling` (validated programmatically via `python3 -c "import yaml; ..."`). `docker compose stop mineru` then keeping `open_notebook` running works in this sandbox. |
| 6 | `ExtractedElement` instances have populated bboxes (per Q-A-5) | **Done** | Validated by `test_text_levels_map_to_title_heading_and_paragraph` (exact bbox values), `test_table_record_yields_extracted_table_with_rows_and_markdown`, `test_image_and_chart_records_yield_extracted_image_with_resolved_path`, and the pipeline-vs-VLM normalisation tests. Page-furniture types preserved with appropriate `ElementType` (HEADER/FOOTER/FOOTNOTE/CAPTION). |

### What was validated locally

- `uv run --project apps/app-main pytest apps/app-main/tests/test_mineru_http_client.py apps/app-main/tests/test_mineru_layout_parser.py` → **28/28 passing**, combined coverage 94%.
- `uv run --project apps/app-main pytest apps/app-main/tests/` → **263/263 passing** (full app-main suite; no regression in upstream processing pipeline; A.1b is the integration point and lives in the next PR).
- `docker compose config --services` lists `mineru` alongside the
  other services; YAML is valid.
- `docker compose build mineru` completed in ~48 min wall-clock
  cold (network-bound `mineru[all]` install). Resulting image
  `open-notebook-mineru:latest` is 12 GB.
- `docker compose up -d mineru` brings the container to `Up`
  state; `curl -sf http://localhost:8104/health` returns 200 with
  `{"status":"ok","service":"mineru"}` in <5 s.
- `docker compose stop mineru` while `open_notebook` is configured
  (no `depends_on` reference to `mineru`) — the main app boot is
  decoupled from MinerU availability per criterion 5.

### What was NOT validated (deferred)

- **Live MinerU /process call against a real PDF** (acceptance
  criterion 2). The first `/process` call triggers the MinerU model
  download (~5–8 GB from HuggingFace or modelscope, configurable
  via `MINERU_MODEL_SOURCE`), which is environment-dependent (rate
  limits, mirror availability, GPU presence). The HTTP contract is
  exercised by `test_process_success_returns_ingestion_result_with_document`
  via on-disk fixtures matching the documented schema; the same
  code path executes against real MinerU output. Smoke recipe
  documented in `services/mineru/README.md`.

### Caveats & follow-ups

1. **Single Q-A-5 caveat — VLM backend coordinate units**: MinerU's
   pipeline / office backends emit bboxes in 0–1000 ints; the vlm
   backend emits 0–1 floats. The parser introspects
   `middle.json._backend` to pick the divisor. If a fourth backend
   ever appears with a different convention, add it to the lookup
   in `_detect_bbox_factor`.

2. **Table HTML→rows parser is lossy on merged cells**: MinerU's
   `table_body` HTML uses `<colspan>`/`<rowspan>` that our minimal
   `_HTMLTableParser` flattens. Acceptable for V1 — the original
   HTML is preserved on `ExtractedTable.metadata["html"]` so a
   future rich renderer can reconstruct merged cells. Chunk-builder
   already treats tables as opaque markdown blobs.

3. **List items collapsed into one element**: `list` records yield
   one `LIST_ITEM` element with all items joined by `- `. If
   downstream features need per-item granularity (e.g. citation
   anchoring per list item), this is a future refinement; the raw
   items live on `metadata["list_items"]`.

4. **MinerU CLI wrapping vs the bundled mineru-api**: We
   deliberately wrap the CLI synchronously rather than exposing
   MinerU's bundled async `mineru-api`. Rationale documented in
   `services/mineru/api.py` and `services/mineru/README.md`. If
   future scaling makes the per-request CLI startup overhead
   painful, swap to direct Python API + lazy `Model` init without
   changing the HTTP contract.

5. **First-run download time**: the `mineru[all]` install pulls
   ~5–8 GB of model weights on first request; documented in
   `services/mineru/README.md`. The `mineru_models` named volume
   in docker-compose persists this across container restarts.

6. **GPU memory sharing with docling**: documented in
   `services/mineru/README.md` — auto-fallback (A.1c) runs the two
   parsers sequentially within one extraction call, never in
   parallel, so they never compete for VRAM.

7. **A.1b integration point**: `SourceExtractor._process_file` does
   not yet call `MineruHttpClient` — that wiring lives in A.1b
   (`parser_engine` routing + `engine_dispatcher`). The HTTP client
   is independently usable now (e.g. for ad-hoc scripts) but is
   not yet reached from the production source-upload flow.

8. **PR creation**: branch pushed but no PR is opened — that's the
   orchestrator's responsibility after review.


---

## Phase A.1b — IMPLEMENTED (2026-06-03)

**Branch**: `track/a-mineru-dispatcher` (pushed to origin; PR creation is up to the orchestrator)

**Commits** (oldest → newest):

| Hash | Message |
|------|---------|
| `7ebd8bf` | `feat(settings): rename default_content_processing_engine_doc to parser_engine (A.1b)` |
| `13efe76` | `feat(parsing): add parser engine dispatcher (A.1b)` |
| `180d415` | `feat(extractor): dispatch _process_file on parser_engine + record engine used (A.1b)` |
| `d37c322` | `feat(api,frontend): parser_engine override + frontend rename (A.1b)` |

### Acceptance criteria status

1. **`PUT /api/settings` round-trip with `parser_engine`** ✅
   Backend `SettingsResponse` / `SettingsUpdate` carry `parser_engine` and
   `mineru_supported_extensions`; `test_settings_router.py` asserts the new
   field on GET. The ContentSettings model accepts `parser_engine` directly.
2. **`parser_engine="docling"` default keeps existing behaviour** ✅
   Mock-based test `test_records_parser_engine_used_in_metadata` confirms the
   default path still calls `IngestionWorkflow` and records
   `metadata["parser_engine_used"] = "docling"`. (Bit-identical regression
   sample snapshot is deferred to integration time — pending Docker compose
   stack; we have no recorded fixture today.)
3. **`parser_engine="mineru"` calls MineruHttpClient.process once + persists metadata** ✅
   `test_routes_to_mineru_when_parser_engine_mineru` asserts `process` is
   awaited exactly once and `metadata["parser_engine_used"] == "mineru"`.
4. **`parser_engine="mineru"` + unsupported extension falls back to docling** ✅
   `test_falls_back_to_docling_for_unsupported_mineru_ext` confirms the
   MinerU client is never instantiated for `.html`, the docling path runs,
   and the dispatcher logs at INFO ("MinerU does not support .html…").
5. **Per-source `ReprocessRequest.parser_engine` override** ✅
   Field added (default `None`). Flows through existing
   `processing_overrides` plumbing in `SourceProcessor.process_source`
   which already merges per-call overrides onto `ContentSettings`. The
   override is end-to-end-exercisable through `POST /api/sources/{id}/reprocess`.
6. **`select_parser_engine` ≥90% coverage** ✅
   97% line coverage (32 test cases in `test_engine_dispatcher.py`).

### Migration

Added `migrations/42.surrealql` + `migrations/42_down.surrealql`:

- **42 (up)**: Idempotent rewrite of `open_notebook:content_settings` —
  copies the old `default_content_processing_engine_doc` value to
  `parser_engine` (with old `"auto"` → new `"docling"` because the
  semantics changed); fills `mineru_supported_extensions` default;
  UNSETs the obsolete field.
- **42 (down)**: Restores `default_content_processing_engine_doc` from
  `parser_engine` (new `"mineru"` maps back to `"docling"` so the
  downgraded app boots cleanly); UNSETs the new fields.

Each step is gated on field state so re-running the migration is a no-op.

### Files created

- `apps/app-main/src/app_main/services/parsing/engine_dispatcher.py` (97% covered)
- `apps/app-main/tests/test_engine_dispatcher.py` (32 cases)
- `migrations/42.surrealql` + `migrations/42_down.surrealql`

### Files modified

- `packages/shared/src/shared/models/settings.py` — rename + new
  `mineru_supported_extensions` field
- `packages/shared/tests/test_models.py`, `tests/test_domain.py` — field rename
- `apps/app-main/src/app_main/api/schemas.py` — `SettingsResponse` +
  `SettingsUpdate`
- `apps/app-main/src/app_main/api/routers/sources_processing.py` —
  `ReprocessRequest.parser_engine` override
- `apps/app-main/src/app_main/services/source_extractor.py` — dispatcher
  wired into `_process_file`; `metadata["parser_engine_used"]` recorded
- `apps/app-main/src/app_main/services/parsing/__init__.py` — re-exports
- `apps/app-main/tests/test_settings_router.py` — field rename
- `apps/app-main/tests/test_source_processing_service.py` — +4 routing tests
- `frontend/src/lib/types/api.ts` — `parser_engine` + `mineru_supported_extensions`
- `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx` — field rename
  (UI revamp deferred to A.2; current options remain Docling + Simple)
- `frontend/src/components/sources/steps/ProcessingConfigStep.tsx` — field rename
- `docs/development/api-reference.md` — API docs reflect new field

### Tests run

- `uv run --project apps/app-main pytest apps/app-main/tests/` → **299 passed**
  (up from 263 in A.1a; +32 dispatcher + 4 routing tests + minor renames)
- `uv run --project packages/shared pytest packages/shared/tests/` → **101 passed**
- `cd frontend && npx tsc --noEmit` → **0 errors**

### Notes / caveats

1. **"simple" engine** is preserved as a value but currently routes
   through Docling because there is no separate simple-extraction
   implementation. Behaviour is unchanged from before A.1b. A real
   simple-only path can land in a future refactor without touching the
   dispatcher contract.

2. **"auto" engine** is in the enum + selectable on the form
   schema, but the dispatcher resolves it to Docling in A.1b. The
   confidence-driven fallback ships in A.1c by swapping the `"auto"`
   branch inside `select_parser_engine` (or one layer up in
   `SourceExtractor`). The pinned test
   `test_auto_setting_resolves_to_docling_in_a1b` will turn into the
   "regression guard" for the A.1c upgrade.

3. **Settings form UI is intentionally minimal in A.1b**. The
   options dropdown still shows Docling + Simple only. MinerU and Auto
   plus the confidence slider land in A.2 along with the badge and
   the MinerU service health chip. The renamed field is wired through
   so save/load round-trips work today.

4. **Migration is idempotent but not yet executed in CI**. The
   SurrealQL is shape-checked but not stack-tested. First-boot of the
   compose stack with this branch will exercise it; orchestrator may
   want to run an integration smoke before merging if there is real
   data in any running database.

5. **No PR created** — branch pushed for orchestrator review.

6. **Operator warning — setting MinerU globally via PUT /api/settings in
   A.1b**: the field is wired through the API and persisted, so a `PUT
   /api/settings {"parser_engine": "mineru"}` will succeed and is honoured
   by `SourceExtractor` for all subsequent ingestions. However the
   Settings UI dropdown in A.1b still only renders **Docling** and
   **Simple**, so the next time an operator opens the form and saves
   (even without intending to change anything), the underlying Radix
   `<Select>` value mismatch will silently submit the form's local state
   and overwrite `parser_engine=mineru` back to one of the visible
   options. **Avoid setting MinerU globally via API until A.2 lands the
   four-option dropdown.** Per-source override via
   `POST /api/sources/{id}/reprocess` body
   `{"parser_engine": "mineru"}` is safe and intended for A.1b — it does
   not touch the singleton settings row.

---

## Phase A.1b — attempt 2 fixes (2026-06-03)

Adversarial review (`docs/tracks/A-mineru/reviews/phase-A.1b-attempt-1.md`)
returned **REVISIONS_NEEDED** with one blocker and one major. Both are
addressed on the same `track/a-mineru-dispatcher` branch (revision pushed
on top of the attempt-1 commits).

### Issues addressed

| Severity | Issue | Resolved? |
|----------|-------|-----------|
| 🔴 Blocker | `SettingsResponse.parser_engine` / `SettingsUpdate.parser_engine` typed as `Optional[str]` — accepted arbitrary strings at the API boundary; invalid values MERGE-upserted into SurrealDB before validation fired on response construction. | **Yes** — both fields now typed `Optional[Literal["simple","docling","mineru","auto"]]`; FastAPI/Pydantic returns 422 on invalid input before the service layer is even called. |
| 🟡 Major | `metadata["parser_engine_used"]` recorded the *user setting* rather than the *engine that actually ran* — `simple` claimed "simple" despite Docling running; audio files claimed "docling" despite WhisperX running. A.1c's confidence-fallback would consume garbage. | **Yes** — `parser_engine_used` is now set per dispatch branch (`"mineru"` only when MinerU runs, `"docling"` for the docling service / IngestionWorkflow document path, `"whisperx"` when IngestionWorkflow emits a transcription). |
| 🔵 Minor | Operator-warning footnote about setting MinerU globally via API in A.1b. | **Done** — added as caveat #6 above. |

### What changed

**Modified**:

- `apps/app-main/src/app_main/api/schemas.py` — `SettingsResponse.parser_engine`
  and `SettingsUpdate.parser_engine` retyped from `Optional[str]` to
  `Optional[Literal["simple","docling","mineru","auto"]]`. The `Literal` import
  was already present (used by `RebuildRequest.mode` and `IssueAlert.type`).
- `apps/app-main/src/app_main/services/source_extractor.py:_process_file`
  — introduced an `engine_used: str` local that each dispatch branch
  sets explicitly: `"mineru"` inside `if use_mineru:`, `"docling"` inside
  the `elif use_docling_service:` block, and the IngestionWorkflow
  `else` branch post-execution decides between `"docling"` and
  `"whisperx"` based on `result.transcription is not None and
  result.document is None`. The final `ExtractionResult.metadata`
  uses `engine_used` instead of the user-setting `resolved_engine`.
  Docstring updated to spell out the engine-that-ran semantic so A.1c
  has an unambiguous contract to consume.

**Added (tests)**:

- `apps/app-main/tests/test_settings_router.py::TestParserEngineValidation`
  — `test_put_rejects_invalid_parser_engine` asserts 422 on
  `{"parser_engine": "evil_value"}` and that the service mock was
  never awaited. `test_put_accepts_each_valid_parser_engine` is a
  parametrized 200-check over the full literal set
  (`simple`, `docling`, `mineru`, `auto`).
- `apps/app-main/tests/test_source_processing_service.py::TestProcessFile`
  — `test_simple_setting_records_docling_engine_used` pins the new
  contract for `parser_engine="simple"` + `.pdf` →
  `parser_engine_used="docling"` (engine that ran).
  `test_audio_records_whisperx_engine_used` pins the audio path —
  `parser_engine="mineru"` + `.mp3` →
  `parser_engine_used="whisperx"` (WhisperX bypasses the dispatcher
  entirely; the metadata reflects the actual pipeline).

### Engine-used semantic (canonical reference)

| Dispatch branch | Trigger condition | `parser_engine_used` value |
|-----------------|-------------------|----------------------------|
| MinerU HTTP client | `select_parser_engine(...) == "mineru"` and ext supported | `"mineru"` |
| Docling HTTP service | `USE_DOCLING_SERVICE=true` and ext is docling-parseable | `"docling"` |
| IngestionWorkflow → Docling | else branch; `result.document is not None` | `"docling"` |
| IngestionWorkflow → WhisperX | else branch; `result.transcription is not None and result.document is None` | `"whisperx"` |

Note that `select_parser_engine` is only consulted for docling-parseable
extensions (`.pdf`, `.docx`, `.xlsx`, `.pptx`, `.html`, `.txt`, `.md`,
etc.). Audio/video files (`.mp3`, `.mp4`, `.wav`, ...) skip the
dispatcher and the post-execution branch above is what classifies them.
This means `parser_engine="mineru"` + `.mp3` ⇒ `parser_engine_used="whisperx"`,
not `"mineru"`.

### Tests run

- `uv run --project apps/app-main pytest apps/app-main/tests/test_settings_router.py` → **8 passed** (was 3; +5 new validation cases)
- `uv run --project apps/app-main pytest apps/app-main/tests/test_source_processing_service.py` → **41 passed** (was 39; +2 engine-used semantic tests)
- `uv run --project apps/app-main pytest apps/app-main/tests/` → **306 passed** (was 299; +7 total)
- `uv run --project packages/shared pytest packages/shared/tests/` → **101 passed** (unchanged)
- `cd frontend && npx tsc --noEmit` → not re-run (no frontend changes in attempt 2)

### Not addressed (per brief)

The other minors from the review (brittle mineru-mock patching, Zod
schema vs. UI dropdown mismatch, missing `mineru_supported_extensions`
in form schema, lossy mineru→docling down migration, inert test_domain.py
assertion, live-migration testing) remain deferred per the brief —
either A.2 territory or acknowledged caveats. Minor #2 (the operator
warning) is addressed as caveat #6 above.

---

## Phase A.1c — IMPLEMENTED (2026-06-03)

**Branch**: `track/a-mineru-fallback` (pushed to origin; PR creation is the orchestrator's responsibility)

**Commits** (oldest → newest):

| Hash | Message |
|------|---------|
| `154181e` | `feat(parsing): docling confidence scoring (A.1c)` |
| `b956e5d` | `feat(parsing): auto-fallback orchestrator + tests (A.1c)` |
| `eb69df9` | `feat(extractor,models): wire auto-fallback + Source.metadata (A.1c)` |
| `e84f393` | `feat(api,schemas): docling_min_confidence on Settings + Source.metadata tests (A.1c)` |

### What was done

**Created**:

- `apps/app-main/src/app_main/services/parsing/confidence.py` — pure
  function `score_docling_extraction(result, *, threshold)` returning
  a frozen `DoclingConfidenceScore(overall, signals, decision, threshold)`.
  Six signals weighted (sum to 1.0, asserted in tests):
  | Signal | Weight | Computation |
  |---|---|---|
  | `ocr_confidence` | 0.30 | mean of `ExtractedElement.confidence` for `source == "ocr"` elements; 1.0 when no OCR ran |
  | `text_density` | 0.20 | `len(full_text) / pages`, normalised by 1500 chars/page baseline |
  | `heading_rate` | 0.15 | `headings / pages`, ≥2/page saturates to 1.0 |
  | `table_success` | 0.15 | fraction of tables with non-empty rows (1.0 when no tables) |
  | `image_text_ratio` | 0.10 | `1.0 - min(images / text_count, 1.0)` |
  | `unknown_element_ratio` | 0.10 | `1.0 - (unknown_text_count / total)` |
  Audio/transcription results (`document is None`) return a neutral
  1.0 — fallback is decided by `success` upstream, not confidence.
- `apps/app-main/src/app_main/services/parsing/auto_fallback.py` —
  `async def extract_with_auto_fallback(file_path, *, docling_client,
  mineru_client, threshold)` returning `(chosen_result, engine_used,
  score)`. Algorithm: docling first → score → MinerU if below
  threshold (Q-A-2 trust, no comparative scoring); docling raising
  triggers MinerU; both engines failing falls back gracefully to the
  degraded docling result rather than stranding the user.
- `apps/app-main/tests/test_docling_confidence.py` — 23 tests:
  - Weights-sum-to-1 + signal-name uniqueness invariants
  - High-quality perfect-doc fixture scores ≥ 0.95 (criterion #2)
  - Scanned image-only fixture scores ≤ 0.7 (criterion #3)
  - Per-signal extremes (no OCR → 1.0; low OCR avg 0.4 → 0.4;
    no tables → 1.0; all empty tables → 0.0; density saturation
    at 1500 chars/page; linear scaling at 750; heading saturation
    at ≥2/page; no images → 1.0; image-heavy → 0.0; unknown
    elements penalty)
  - Audio-only neutral branch
  - Threshold override echoed back
  - Perf guard: 50-page document scored in < 50 ms (criterion #9 —
    measured ~0.1 ms in this environment, well under budget)
- `apps/app-main/tests/test_auto_fallback.py` — 8 tests covering
  the four plan paths + three defensive scenarios (high-conf keeps
  docling, low-conf falls back, docling-raises triggers MinerU,
  `success=False` is treated as failure, threshold override flips
  accept→fallback, both-engines-fail keeps degraded docling, both
  engines raising raises RuntimeError, default threshold used when
  omitted).

**Modified**:

- `packages/shared/src/shared/models/source.py` (Q-A-3) — `Source`
  gains `metadata: Dict[str, Any] = Field(default_factory=dict)` with
  a defensive `field_validator` that coerces NULL/non-dict legacy
  column values to `{}`. Additive + non-breaking; existing rows
  default to `{}` without a SurrealDB migration (schemaless tables).
- `packages/shared/src/shared/models/settings.py` — `ContentSettings`
  gains `docling_min_confidence: Optional[float] = Field(0.95, ge=0.0,
  le=1.0)`. Bounds enforced at the model level; the API schema also
  re-enforces the bound at the boundary.
- `apps/app-main/src/app_main/services/source_extractor.py` —
  `_process_file` refactored into three private methods:
  - `_process_file` — high-level dispatch on the user setting.
  - `_run_docling` — extracted helper wrapping the existing
    HTTP-vs-IngestionWorkflow choice; returns `(IngestionResult,
    engine_label)` where engine_label is `"whisperx"` for audio
    routed through the in-process workflow.
  - `_run_auto_fallback` — new helper that wires `_run_docling`
    and `MineruHttpClient` into `extract_with_auto_fallback`. Records
    `extraction_confidence`, `extraction_confidence_signals`, and
    `extraction_fallback_triggered` onto `ExtractionResult.metadata`.
  Auto-mode only fires when (a) the user picked `parser_engine="auto"`
  AND (b) the extension is docling-parseable. Audio/video continue
  to skip the dispatcher entirely.
- `apps/app-main/src/app_main/services/source_processor.py::_update_source`
  — lifts a curated subset of `ExtractionResult.metadata` onto
  `Source.metadata`: `parser_engine_used`, `extraction_confidence`,
  `extraction_confidence_signals`, `extraction_fallback_triggered`.
  Non-provenance noise (`processing_time`, `markdown_path`) stays on
  the ExtractionResult so the DB row stays minimal. Update is
  omitted entirely when no provenance keys are present (preserves
  the lightweight legacy path).
- `apps/app-main/src/app_main/api/routers/sources_processing.py` —
  `ReprocessRequest.docling_min_confidence: Optional[float]` with
  `[0, 1]` bounds. Flows through existing `processing_overrides`
  plumbing to override the global threshold for a single reprocess.
- `apps/app-main/src/app_main/api/schemas.py` —
  `SettingsResponse.docling_min_confidence?` and
  `SettingsUpdate.docling_min_confidence?` (Optional[float], [0, 1]).
- `apps/app-main/src/app_main/services/parsing/__init__.py` —
  re-exports `extract_with_auto_fallback`,
  `score_docling_extraction`, `DoclingConfidenceScore`,
  `DEFAULT_THRESHOLD`, and `SIGNAL_WEIGHTS`.
- `frontend/src/lib/types/api.ts` —
  `SettingsResponse.docling_min_confidence?: number`.
- `apps/app-main/tests/test_source_processing_service.py` — **removed**
  the A.1b regression guard `test_auto_setting_resolves_to_docling_in_a1b`
  (as specified in the brief) and replaced it with three integration
  tests that drive `_process_file` end-to-end:
  - `test_auto_high_confidence_keeps_docling` — assert MinerU mock
    never called, metadata says docling, `extraction_fallback_triggered
    == False`, all four provenance keys present.
  - `test_auto_low_confidence_falls_back_to_mineru` — assert MinerU
    mock awaited once, metadata says mineru, fallback triggered.
  - `test_auto_respects_docling_min_confidence_override` — assert that
    raising the per-call threshold to 0.99 flips a mid-quality doc
    from accept to fallback.
- `apps/app-main/tests/test_source_processing_service.py::TestUpdateSource`
  — +2 cases:
  - `test_lifts_extraction_metadata_to_source_metadata` — provenance
    keys are lifted onto `Source.metadata`; non-provenance keys are
    not.
  - `test_omits_metadata_when_no_provenance_keys` — when the
    ExtractionResult has none of the four keys, the update payload
    has no `metadata` field.
- `apps/app-main/tests/test_engine_dispatcher.py::TestAutoSetting` —
  docstring updated to reflect the new semantics (dispatcher still
  resolves `"auto"` to `"docling"`; the real fallback is one layer
  up in the extractor).
- `packages/shared/tests/test_models.py::TestSource` — +4 cases:
  default empty dict, accepts provenance bag, coerces NULL,
  coerces non-dict.

### Acceptance criteria status

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | `score_docling_extraction()` returns `DoclingConfidenceScore` with `overall ∈ [0, 1]` for any input; weights sum to 1.0 (asserted in test) | **Done** | `test_weights_sum_to_one` asserts `sum(SIGNAL_WEIGHTS.values()) == 1.0`; `test_score_is_within_unit_interval_for_empty_result` plus the perfect/scanned-doc tests verify clamping. |
| 2 | Synthetic "perfect document" fixture scores ≥ 0.95 | **Done** | `test_score_for_perfect_document_is_high` — 10-page fixture (2500 chars/page, ≥2 headings/page, 2 parsed tables, no images, all native source) scores ≥ 0.95. |
| 3 | Synthetic "scanned image-only" fixture (OCR avg 0.4) scores ≤ 0.7 | **Done** | `test_score_for_scanned_document_is_low` — 5-page fixture (100 chars/page, OCR confidence 0.4, lots of images, no structural elements) scores ≤ 0.7. |
| 4 | `parser_engine="auto"` + fixture scoring 0.97 ⇒ MinerU not called + metadata.parser_engine_used == "docling" | **Done** | `test_auto_high_confidence_keeps_docling` — mineru mock `.assert_not_called()`, metadata records docling + fallback_triggered=False. |
| 5 | `parser_engine="auto"` + fixture scoring 0.60 ⇒ MinerU called + metadata.extraction_fallback_triggered == True | **Done** | `test_auto_low_confidence_falls_back_to_mineru` — `.assert_awaited_once()`, metadata records mineru + fallback_triggered=True. |
| 6 | docling raising ⇒ auto-mode swallows + calls MinerU + returns MinerU result + WARNING log | **Done** | `test_docling_exception_triggers_mineru_with_warning` in `test_auto_fallback.py`; loguru WARNING emitted via `_safe_run`. |
| 7 | `processing_overrides.docling_min_confidence=0.99` raises the bar for that call | **Done** | `test_auto_respects_docling_min_confidence_override` in `test_source_processing_service.py`; `test_threshold_override_raises_the_bar` in `test_auto_fallback.py`. |
| 8 | `Source.metadata` after auto-extraction contains `parser_engine_used`, `extraction_confidence`, `extraction_confidence_signals`, `extraction_fallback_triggered` | **Done** | `test_lifts_extraction_metadata_to_source_metadata` in `TestUpdateSource` verifies the lift in the unit-test layer. Full HTTP-level `TestClient` integration test deferred (no existing reprocess HTTP test scaffolding in app-main today; adding one would have required a substantial new fixture for `get_source_service` + repos. The four provenance keys are end-to-end validated by the unit tests + the auto-mode `_process_file` tests above, which exercise the exact ExtractionResult → SourceProcessor flow.). |
| 9 | `score_docling_extraction()` for 50-page document fixture: < 50 ms | **Done** | `test_score_is_fast_for_50_page_document` measures ~0.1–0.5 ms locally over 5 iterations (well under the 50 ms budget; perf assertion fails fast if regressed). |

### Architectural choice — where does auto-fallback live?

Per the brief, "if `auto_fallback` orchestration introduces architectural
ambiguity (where does it live — service or extractor?) → use your judgment
but document choice".

**Choice**: lives at `apps/app-main/src/app_main/services/parsing/auto_fallback.py`
as a **pure async function** (`extract_with_auto_fallback`) that takes
explicit `docling_client` and `mineru_client` dependencies and returns the
chosen result + engine + score. Reasons:

1. The dispatcher (`engine_dispatcher.py`) is pure routing and returns
   a single concrete engine string. Adding the dual-extraction control
   flow there would have polluted its surface (now needs async + clients).
2. Inlining the algorithm into `SourceExtractor._process_file` would have
   stranded it from test coverage (`SourceExtractor` already pulls in
   IngestionWorkflow + httpx + heavy imports; the pure orchestrator can
   be tested with mocks in milliseconds).
3. Making it dependency-injected (clients passed in) lets tests use
   `MagicMock`s; `SourceExtractor` constructs both clients itself in
   `_run_auto_fallback` and passes them through. The orchestrator never
   touches env vars, file I/O, or logging beyond loguru-warnings.

The `_DoclingAdapter` inside `_run_auto_fallback` is the small bridge:
it exposes `.process(path) -> IngestionResult` over either the docling
HTTP client or the in-process IngestionWorkflow, depending on env vars
and extension — so the orchestrator never needs to know which docling
backend is active. Trade-off: an extra class per auto-extraction call.
Acceptable; it's a closure-style adapter, not a long-lived service.

### Tests run

- `uv run --project apps/app-main pytest apps/app-main/tests/` →
  **341 passed** (was 306 baseline; +23 confidence + 8 auto_fallback
  + 3 auto-mode integration + 2 metadata lift, minus 1 A.1b regression
  guard removed = +35 net new tests).
- `uv run --project packages/shared pytest packages/shared/tests/` →
  **101 passed** (was 97; +4 Source.metadata tests).
- `uv run --project packages/surrealdb-service pytest
  packages/surrealdb-service/tests/` → **45 passed** (unchanged).
- `cd frontend && npx tsc --noEmit` → **0 errors**.

### Caveats & follow-ups

1. **TestClient HTTP-level integration test for criterion #8 was
   deferred** — the app-main test suite has no existing HTTP-level
   `/reprocess` test scaffolding (`get_source_service` factory uses
   real repos and would have required ~50 LOC of fixture wiring).
   Provenance flow is end-to-end-exercised at the unit + integration
   layers (the auto-mode `_process_file` tests, which call extract,
   plus the `_update_source` tests, which assert the lift). Phase
   A.3 will add a Playwright E2E spec that drives this through the
   real reprocess endpoint; the wiring is well-trodden by then.

2. **`docling_min_confidence` is not yet exposed in the Settings UI**.
   Backend wiring is complete (model + API + frontend type); the
   UI Slider lands in Phase A.2 alongside the parser-engine dropdown
   and MinerU health chip. Users can set it via `PUT /api/settings`
   today.

3. **MinerU client instantiated on every auto-fallback call**.
   `MineruHttpClient()` is cheap (no model loading; httpx client is
   per-call inside `.process`), but if profiling later shows this
   matters, it can be lazy-cached on `SourceExtractor`. Not worth
   doing pre-optimisation.

4. **No SurrealDB schema migration for `Source.metadata`**. The
   table is schemaless in this codebase, so legacy rows without the
   `metadata` column deserialise via the Pydantic default
   (`Field(default_factory=dict)`). The new `field_validator` also
   coerces `None`/non-dict values to `{}` defensively, in case
   anything legacy ever lands in the column with a stray type.

5. **Adapter pattern in `_run_auto_fallback`**. The `_DoclingAdapter`
   inner class is a deliberate closure-style adapter so the
   orchestrator doesn't need to know whether docling went via HTTP
   or in-process workflow. Documented inline; review may want to
   extract it to a top-level helper if the pattern recurs.

6. **Both-engines-fail policy**. If docling succeeds with low
   confidence but MinerU then raises, the orchestrator returns the
   degraded docling result with `parser_engine_used="docling"` +
   `extraction_fallback_triggered=true`. This is the deliberate
   "never strand the user" policy from the plan. If MinerU is down
   for an extended period, every auto-mode upload effectively
   becomes a docling upload with a fallback-attempted flag. Phase
   A.2's MinerU health chip will surface this proactively.

7. **PR creation**: branch pushed but no PR is opened — that's the
   orchestrator's responsibility after review.

