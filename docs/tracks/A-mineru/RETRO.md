# Track A — Retrospective (MinerU integration)

> Closing date: 2026-06-04
> Branches merged: `track/a-playwright`, `track/a-mineru-service`,
> `track/a-mineru-dispatcher`, `track/a-mineru-fallback`,
> `track/a-mineru-ui`, `track/a-mineru-integration`
> Final state: see `docs/tracks/A-mineru/status.md`.

This retrospective draws on the per-phase status entries (`status.md`)
and the adversarial-review files (`reviews/phase-A.*-attempt-N.md`). It
is meant to inform sprint planning for tracks B-G; it does not double
as the project's general-purpose lessons-learned doc.

## What worked

- **Phase decomposition was correct-sized.** Six phases (A.0, A.1a,
  A.1b, A.1c, A.2, A.3) averaged ~1.5 dev-days of code per PR; large
  enough to land a self-contained capability, small enough that
  reviewers could read every diff without fatigue.
- **Adversarial review caught real defects before merge.** Three of
  six phases (A.0, A.1b, A.1c) needed an attempt-2. The blockers
  surfaced (CI redirect path, falsy-zero threshold bug, missing
  SCHEMAFULL migration) would all have caused production issues had
  they shipped silently. The first-try APPROVEDs (A.1a, A.2, A.3)
  correlate with phases where the implementer had a clear local
  validation harness — e.g. A.1a's HTTP-client mocks let the
  implementer test in isolation.
- **Pure functions for the hot path** (`score_docling_extraction`)
  paid off in tests. 23 confidence-scoring cases run in ~5s — the
  same test count against a service+IO API would have taken minutes.
- **DI through closure adapters** (`_DoclingAdapter` in
  `_run_auto_fallback`) let A.1c's orchestrator stay agnostic about
  whether docling went via HTTP or in-process workflow. Single
  pattern across both backends; no second config flag to plumb.
- **Frontend test framework decision in A.0 made A.2 cheap.** Eight
  Playwright sub-tests against route mocks in A.2 took ~0.5 day to
  author — would have been multiple days against a live stack.
- **Mock fixtures in `apps/app-main/tests/conftest.py` scaled.** All
  service tests reuse the same `AsyncMock` repo factory; new tests in
  A.3 dropped in without per-test wiring.

## What hurt

- **CI workflow PAT scope blocker (A.0).** The `e2e.yml` file is
  still pending move to `.github/workflows/` because the implementer
  lacked the `workflow` OAuth scope. Five other PRs landed and the
  workflow is still staged at `e2e-workflow.yml.pending`. This is a
  procedural debt that should be resolved before track B starts —
  otherwise the CI smoke contract Track A introduced doesn't fire.
- **No live-SurrealDB round-trip test.** A.1c attempt-1 wrongly
  declared `source` schemaless. The fix (migration #43) is right but
  the testing gap remains — the SurrealDB harness assertion only
  covers the *static SQL of the migration file*, not the live
  round-trip. Track B should land `pytest-docker` or testcontainers
  early; the work is non-trivial (~1 day) but unlocks confidence for
  every track that touches schema.
- **Confidence-score signal calibration is incomplete.** The Phase
  A.3 tuning corpus revealed that `table_success` zeros the moment
  any table has zero parsed rows — even a clean text doc with a
  single mis-parsed table tank its score by 0.15. No track A blocker
  (the bias is towards more fallback, which is the safe default), but
  worth a polish PR.
- **Frontend prod-bundle vs dev drift.** Track A.2/A.3 validated
  against `next dev`, but the bundled `open_notebook` Docker image
  served from port 8502 is from June 2 (pre-A.2). Anyone running
  `docker compose up open_notebook` won't see the parser-engine UI
  until the image rebuilds. A docker-compose `build: always` hint
  or a CHANGELOG entry would have surfaced this.
- **Three-file threshold cascade.** If A.3's tuning had concluded
  "change default to 0.85", three files would have needed
  synchronised edits: `confidence.py`, `settings.py`,
  `SettingsForm.tsx`. A single canonical constant exported from
  `packages/shared` would have been cleaner; we kept the cascade
  because the tuning landed on "keep 0.95" and the refactor isn't
  worth doing speculatively.
- **Background docling runs are slow on full corpora.** Scoring 8
  PDFs (one >30 MB) via the docling service exceeded the 20-min wall
  clock during tuning. Operators tuning thresholds need a
  pre-warm-cache flow or per-PDF parallelism — neither exists today.

## Recommendations for tracks B-G

1. **Land a testcontainers SurrealDB harness early in Track B.**
   Every track from B onward will need to test SCHEMAFULL migrations
   end-to-end. The A.1c blocker (#43) would have been caught at
   author-time. Estimate: 0.5-1 day of one-time investment.

2. **Single source of truth for thresholds/defaults.** Don't repeat
   the three-file cascade. Track B's `notebook_schema` and `pass1_results`
   tables will have similar defaults — declare them once in
   `packages/shared/src/shared/models/<feature>.py` and import from
   there both server- and client-side (via codegen if needed).

3. **Build self-validation scripts during the feature, not after.**
   Track A.3's `score_pdf_corpus.py` would have been even more
   useful if it had landed during A.1c (the implementer could have
   tuned with it before A.2 shipped the slider). Apply this to B4
   ("confidence everywhere") and B1 (two-pass schema validation).

4. **Pre-resolve at planning time** the kind of decisions A.3
   surfaced as "open questions" (ARCHITECTURE.md insertion point,
   README subsection location, raw-vs-curated tuning output,
   graceful-degradation as separate spec). The pre-resolved decisions
   in this phase's prompt cut its execution time roughly in half.

5. **Always-on telemetry hook** — even a no-op SQL `INSERT INTO
   metrics` per major action. Without it, "is the default well-tuned?"
   stays anecdotal. Track A ships blind to its own fallback rate.
   Could be added as a B-track piggy-back: every `auto_fallback`
   decision writes `(source_id, score, decision, ts)` to a `metrics`
   table.

6. **Plan for stale docker images.** Production-bundle drift hit A.3
   testing. Either: (a) make `docker-compose up` always rebuild the
   open_notebook image (slow but unambiguous), or (b) put a version
   stamp on the production index.html and surface it in the UI
   (cheap diagnostic).

7. **Reuse the route-mock pattern.** The four `_track-a-helpers.ts`
   helpers (`mockDashboardChrome`, `DEFAULT_SETTINGS`) plus the
   per-spec route mocks made each spec ~150 LOC of pure assertion.
   Tracks B-G should lift this into a shared `e2e/_helpers/` module
   the moment a second track needs the same pattern.

## Phase-by-phase attempt count

| Phase | Attempts | First-try result |
|-------|----------|------------------|
| A.0 (Playwright scaffold) | 2 | REVISIONS — smoke spec URL regex too tight; vapid body-visibility assertion. |
| A.1a (MinerU service + HTTP client) | 1 | APPROVED. |
| A.1b (dispatcher + parser_engine rename) | 2 | REVISIONS — `Optional[str]` not `Literal` on API surface; `parser_engine_used` recorded user setting not engine that ran. |
| A.1c (confidence + auto-fallback) | 2 | REVISIONS — missing SCHEMAFULL migration; falsy-zero threshold bug. |
| A.2 (Settings dropdown + slider + chip + badge + override + 4 specs) | 1 | APPROVED. |
| A.3 (integration + tuning + docs) | 1 | APPROVED. |

**Pattern**: phases with a clear, isolated "unit under test" (A.1a's
HTTP client; A.2's UI components; A.3's docs + Markdown CLI) passed
first try. Phases that crossed surfaces (A.0's CI + spec contract;
A.1b's API + dispatcher + frontend rename; A.1c's three-layer
metadata flow with a hidden SCHEMAFULL constraint) needed an
attempt-2. The implementer's per-phase local validation harness was
the strongest predictor of a first-try APPROVED.

## CI infrastructure discovery (Phase 7 → all phases)

A.0 surfaced that the implementer's GitHub PAT lacks the `workflow`
OAuth scope required to create or update files under
`.github/workflows/` via git push. The CI workflow was authored
correctly at `docs/tracks/A-mineru/e2e-workflow.yml.pending` and is
byte-identical to its canonical destination — but the move requires a
workflow-scoped token. This was the right call (commit the artifact
to the branch so it isn't lost) but it means Track A's CI smoke
contract isn't actually firing yet. **This must be resolved before
Track B opens its first PR**, otherwise B will inherit the same gap.

## Tooling that paid off

- **Loguru's structured logger** for the auto-fallback path. The
  `[auto-fallback]` prefix made post-hoc debugging fast.
- **`AsyncMock` over real repos in tests.** 349 app-main tests run in
  ~25s; the same suite with a real SurrealDB would have been minutes.
- **Playwright's `route.fulfill` route mocks.** Mocking at the
  network boundary kept the UI tests fully decoupled from backend
  shape — A.1b's `parser_engine_used` semantic change didn't break a
  single E2E.
- **Pydantic `Literal[...]` types on the API boundary.** Caught the
  attempt-1 A.1b blocker (arbitrary strings → DB) at compile time
  after the retype.

## Tooling that didn't carry its weight

- **The custom docling-host-bridge in `score_pdf_corpus.py`.**
  Necessary for the host-side script to talk to the
  bind-mounted docling container, but four env vars
  (`DOCLING_HOST_INPUT_DIR` etc.) is more configuration than the
  feature warrants. A follow-up should fold this into
  `DoclingHttpClient` itself with auto-detection.

---

**Closing**: Track A landed in 6 PRs over 2 calendar days of
adversarial execution. Five of the six PRs APPROVED on first or
second attempt; zero rollbacks needed.
