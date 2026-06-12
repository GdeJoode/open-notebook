# Track B — Retrospective (KG quality)

> Closing date: 2026-06-12
> Branches merged: `track/b-kg-foundation`, `track/b-models-entity`,
> `track/b-models-notebook-schema`, `track/b-pass1-module`,
> `track/b-pass2-module`, `track/b-multi-schema-orchestrator`,
> `track/b-extraction-service-wiring`, `track/b-ttl-exporter-fix`,
> `track/b-ttl-endpoint`, `track/b-schema-tab-view`,
> `track/b-schema-edit-ops`, `track/b-soft-nudge`,
> `track/b-reextract-prompt`, `track/b-confidence-telemetry`,
> `track/b-orphan-connector`, `track/b-orphan-prune-lifecycle`,
> `track/b-notebook-merge`, `track/b-integration-retro`
> Final state: see `docs/tracks/B-kg-quality/status.md`.

This retrospective draws on the per-phase status entries (`status.md`)
and the adversarial-review files (`reviews/phase-B.*-attempt-N.md`). It
is meant to inform sprint planning for tracks C-G; it does not double
as the project's general-purpose lessons-learned doc.

## Summary

Track B (KG quality) closed on 2026-06-12 after 17 sub-phases delivered
across 17 PRs over roughly 7 calendar days of adversarial execution.
Scope landed end-to-end: multi-schema two-pass entity extraction
(B.1a–B.1f), TTL/RDFS export with Protégé compatibility (B.2a/B.2b), a
full schema-editing UX with soft-nudge + pause-toggle + re-extract
prompt (B.3a–B.3d), always-on confidence telemetry (B.4), an
orphan-connector with prune-lifecycle (B.5a/B.5b), and cross-notebook
graph merge (B.6). Five new migrations (44–48) shipped and the existing
SCHEMAFULL tables were extended additively. Eight of seventeen phases
needed a second review attempt — the reviewer-rejection rate was
roughly 50%, and the adversarial cycle caught real production-blocking
bugs that would have shipped silently otherwise.

## What worked

- **Testcontainers harness from B.0 — caught pre-existing bugs at
  author-time.** Closing RETRO #1 from Track A. Within hours of the
  harness landing it surfaced (a) the legacy `entity_persistence_service`
  field-shape drift (which B.1a fixed), (b) the missing `rdflib` runtime
  dep + module-load `NameError` in `rdf_owl_shacl.py` (which B.2a
  fixed), and (c) the broken `LLMManager` import in `LLMExtractor`
  (which B.1f fixed). All three were latent in main before Track B; the
  harness made them visible at PR-time instead of in production.
- **Adversarial review cycle caught REAL production bugs.** Concrete
  blockers prevented by a second-look pass: resume sentinel leaking into
  the TTL response (B.3c), B.5b prune-lifecycle wired into a service
  with no production caller, B.6 idempotency mock that diverged from
  production's `time::now()` semantics so the regression test would
  have falsely greenlit a bug, B.3c LLM prompt corruption from
  mid-build edits to `prompts/pass1.py`, B.2b URI-illegal characters in
  user-supplied extension names that broke the Protégé roundtrip.
- **Migration ID coordination — no clashes across 5 new migrations
  (44–48).** Each phase explicitly reserved its slot in its plan
  (B.1a→44, B.1b→45, B.3b→46, B.4→47, B.5b→48); even though three
  phases ran in parallel branches, the lock-step ordering held without
  conflicts. The reservation pattern is cheap and a permanent fix for
  the migration-clash failure mode.
- **Parallel phases when files were independent** (B.3b + B.3c + B.5a;
  B.3d + B.5b). When the workspaces split cleanly along file
  boundaries — `notebook_event` (B.3b) vs `orphan_connector` (B.5a) —
  parallel landed without rebase friction. The parallelisation roughly
  halved wall-clock time on those rounds.
- **Confidence-everywhere paradigm** (B.1d Pass-2 confidence → B.4 UI
  bars + telemetry → B.1e merge max-confidence semantics). The single
  invariant "every entity AND every relation carries a confidence" let
  three independent surfaces (extraction, UI, merge) share one contract
  with no negotiation. The B.4 reviewer found exactly one place where
  the invariant could be broken (relation endpoint rewrite during
  merge) and B.1f closed it — that's the only place it had to be
  thought about.
- **Stub name-normalizer pattern** (single import-point for the Q9
  Track-M4 swap). `shared.utils.name_normalizer.normalize_entity_name`
  ships as a lowercase+whitespace+punctuation V1 stub. Track-B phases
  that depend on canonicalization (B.1c, B.1e, B.6) all import from the
  same point. When Track M4 lands TOOI + Crossref the swap is one file
  with no caller rewiring.
- **B.4 telemetry closure of RETRO #5 from Track A.** Track A shipped
  blind to its auto-fallback rate (RETRO #5: "Always-on telemetry hook
  — even a no-op INSERT INTO metrics per major action"). B.4 added the
  `metrics` table + `record_metric()` helper + call sites in
  `extraction.complete` and `auto_fallback`. Subsequent phases (B.1f
  multi-schema dispatch, B.1d Pass-2 chunk events) hooked in for free.
  Operators now have a real signal for "is the default well-tuned?"

## What hurt

- **Implementer over-claimed tests.** Three concrete misses caught by
  reviewers: (B.1f attempt 1) three ACs claimed PASS but had no
  covering tests — the implementer had wired tests for a parallel path
  and assumed transitive coverage; (B.5b attempt 1) the implementer
  claimed the prune lifecycle worked but had NO production caller
  invoking the update path — the unit tests exercised it directly but
  nothing in `entity_persistence_service` ever called it; (B.6
  attempt 1) the idempotency unit test passed against a mock whose
  `updated_at` advanced on every read instead of every write, so the
  semantic-content comparison the production code relied on was being
  short-circuited. None of these reproduced under careful reading of the
  diff — they required running the suite against a strict-advance mock
  (B.6) or grepping for callers (B.5b) or matching ACs to test names
  one-by-one (B.1f).
- **Parallel-branch conflicts in shared files.** B.3c, B.3d, B.3b all
  touched `schemas.py`, `use-notebook-schema.ts`, and
  `notebooks/[id]/page.tsx`. The three branches ran into multi-rebase
  rounds because the soft-nudge banner, the re-extract prompt banner,
  and the edit-ops dialog all wanted to mount above the 3-column grid.
  B.3d attempt 2 inherited a 60-line three-way merge in a single
  React component. Cheaper option in hindsight: sequence
  B.3b → B.3c → B.3d strictly, even if it adds 1–2 days of wall clock.
- **`status.md` merge conflicts every phase.** Each phase appends an
  entry to a shared file; every parallel-rebase round produced a
  trivial-but-real merge conflict at the file head. Cumulatively
  perhaps 30 minutes per parallel-phase pair. A simple fix for tracks
  C-G: split the rolling log into `status-B.N.md` per phase and let
  `status.md` be a thin index that lists them.
- **Cold-compile Playwright races against the dev server.** Stale
  Next.js processes on port 8502 between worktree runs caused
  flaky-first-run failures during B.3a/B.3c local validation. The
  symptom is always the same (404 on `/notebooks/...`); the cure is
  always the same (`pkill -f next-server`); the cost is ~5 minutes per
  occurrence. Worth a Makefile target.
- **WSL torch-wheel hang in `uv sync` blocked implementer test runs.**
  The `services/extraction` deps include CUDA torch; on WSL without GPU
  the wheel selection sometimes hangs at "Resolving" for minutes. B.4
  and B.5a implementers both lost ≥30 minutes to this before the
  workaround (`uv sync --no-install-package torch` in the affected
  worktree) was documented. Should be in the worktree-setup doc.
- **Pre-existing failures create CI noise.** Every Track-B PR ran into
  partial-red CI from two unrelated failures (`claude-review` not
  installed in the runner, `test-build-single` flaking on a Docker
  pull). Reviewers had to ignore them; over 17 PRs the noise erodes
  trust in CI signal. The fix is out of Track B's scope but the cost
  is concrete.

## Recommendations for tracks C-G

1. **Reviewer rejection rate ~50% (8/17 phases needed attempt 2 in
   Track B). Build the cycle into estimates from the start.** Per
   phase, budget 1 review round automatically; estimate the
   adversarial-fix work at ~25-40% of the original implementer effort.
   This matches Track A (3/6 needed attempt 2) and Track B (8/17)
   convergence on roughly half. The implication is that the
   per-phase "code time" line in a plan should be doubled to get to
   "merged time".
2. **For complex shared-file phases (e.g., Track C's
   Writer-Evaluator-Editor stages), prefer sequential over parallel to
   avoid rebase chains.** When two phases will touch the same React
   component, the same router file, or the same shared model, the
   wall-clock gain from parallelism is more than eaten by rebase
   friction. Track B's B.3b/B.3c/B.3d trio is the cautionary tale —
   strict sequencing would have shipped ~2 days sooner.
3. **Centralise "what-tests-claimed-vs-what-actually-ran" in the
   implementer self-review.** Implementer self-review should require
   evidence for every claimed-PASS AC: pytest exit code + last few
   lines of output + the test name(s) that map to the AC. The B.1f /
   B.5b / B.6 over-claims would all have been caught at self-review
   time if the implementer had to type the test name next to the AC.
4. **File-bound parallel work needs explicit cross-track coordination
   notes in the per-phase plan.** When parallel branches must share a
   file, the plan should call out the exact regions each branch will
   touch (line ranges, function names) so the implementer knows where
   to merge cleanly. B.3b/B.3c had to discover this organically; the
   plan can pre-resolve it.
5. **B.0 pattern (testcontainers FIRST) was the highest-leverage
   architectural decision — apply to every track that touches
   SCHEMAFULL tables.** Track C's `summary_versions`, Track G's
   `agent_keys` / `sync_state`, Track F's `audit_findings` all need
   live SurrealDB roundtrip tests at PR-time, not at deploy-time. The
   pattern is in place; the cost is one fixture import per workspace.
6. **Inversion test pattern is the gold standard — formalise this in
   the reviewer template.** B.6 attempt-2's regression test for
   semantic-content idempotency had the reviewer monkey-patch the fix
   back to the original `updated_at` snapshot to verify the test would
   catch the regression. This is more rigorous than "run the test and
   see green" — it proves the test discriminates the bug. Reviewer
   template for tracks C-G should ask for inversion on the central
   regression test of each phase.
7. **Telemetry-first** (B.4 always-on metrics) gives track-wide
   observability — don't defer telemetry to the end. Track B's B.4
   landed mid-track and immediately gave the downstream phases (B.1f,
   B.5b) a place to write counters as part of normal feature work.
   Track A's RETRO #5 framed this as "could be added as a B-track
   piggy-back"; in practice it should be the first feature, not the
   last. Tracks C-G should land their `metrics` event types in the
   same migration as the feature, not as a follow-up.

## Live-test recommendation

The B.7 plan included a "full corpus E2E" with 5 mixed-domain documents
(scholarly, policy, mixed, social, unknown-domain). The individual
Track-B phase E2E specs (`frontend/e2e/track-b/*.spec.ts` — schema-tab,
schema-edit-ops, schema-soft-nudge, schema-reextract, confidence-display,
notebook-merge) already cover each surface in isolation against route
mocks. The corpus E2E is therefore **deferred to a live-test session**:

- Upload 5 mixed-domain documents (1 scholarly PDF, 1 policy PDF,
  1 mixed scholarly+policy PDF, 1 social-media transcript, 1
  unknown-domain document).
- Walk through: schema-tab → edit ops → soft-nudge → re-extract →
  orphan dashboard → cross-notebook merge.
- Verify: Pass-1 `coverage_pct` sensible; Pass-2 `type_tags`
  populated; confidence bars render; orphans land in pending/archived
  as expected; TTL export opens in Protégé.
- Capture screenshots/notes in a follow-up doc (`E2E_EVIDENCE.md` or
  similar). Deferred from B.7 to keep the phase doc-only and unblock
  Track C/D/E/F/G start.

The risk of deferral is bounded — each Playwright spec exercises the
production code paths against the production API, so a corpus run is
expected to validate emergent behaviour (cumulative soft-nudge across
schemas, TTL well-formedness with user-extension names) rather than
core wiring.

## Phase-by-phase attempt count

| Phase | Attempts | First-try result |
|-------|----------|------------------|
| B.0 (testcontainers harness) | 2 | REVISIONS — migrations-dir off-by-one; pool-reset lifecycle. |
| B.1a (Entity/Relation models + persistence drift) | 2 | REVISIONS — schema-side timestamps; in/out aliases. |
| B.1b (notebook_schema + pass1_results tables) | 1 | APPROVED. |
| B.1c (Pass-1 schema validation) | 2 | REVISIONS — malformed-JSON degrade-gracefully; alternative_schemas type; schema-summary compression. |
| B.1d (Pass-2 typed extraction) | 1 | APPROVED. |
| B.1e (multi-schema orchestrator) | 1 | APPROVED. |
| B.1f (EntityExtractionService rewire) | 2 | REVISIONS — three ACs over-claimed without covering tests; B.4 relation-rewrite gap. |
| B.2a (TTL exporter fix) | 1 | APPROVED. |
| B.2b (schema.ttl endpoint) | 2 | REVISIONS — URI-illegal characters in extension names; missing content-disposition test. |
| B.3a (schema-tab view-only) | 2 | REVISIONS — Playwright mock URL regex; flat-listbox vs tree decision. |
| B.3b (schema edit ops) | 1 | APPROVED. |
| B.3c (soft-nudge + pause toggle) | 2 | REVISIONS — resume sentinel leaking into TTL response; LLM prompt corruption. |
| B.3d (re-extract prompt) | 2 | REVISIONS — duplicate router/hook surface vs B.3c; paused-job dedup miss. |
| B.4 (confidence + telemetry) | 1 | APPROVED. |
| B.5a (orphan-connector) | 1 | APPROVED. |
| B.5b (orphan prune-lifecycle) | 2 | REVISIONS — no production caller invoking the update path. |
| B.6 (cross-notebook merge) | 2 | REVISIONS — idempotency mock divergence; archived-source guard; self-merge guard. |

**Pattern**: 9/17 phases passed first try (53%); 8/17 (47%) needed a
second attempt. The first-try APPROVEDs (B.1b, B.1d, B.1e, B.2a, B.3b,
B.4, B.5a) all had two characteristics — a pure-data or pure-function
inner core that could be tested in isolation, AND a single owner for
the affected file set. Phases that crossed surfaces (B.1f's multi-pass
dispatch; B.3a's flat-vs-tree UX decision; B.5b's lifecycle wiring
across services) needed an attempt-2.

## Tooling that paid off

- **Testcontainers SurrealDB harness from B.0.** Every subsequent phase
  that touched a SCHEMAFULL table got real-DB roundtrip tests for
  one-fixture-import cost. Migration 44 / 45 / 46 / 47 / 48 all carry
  passing roundtrip tests on first author-time run.
- **`requires_docker` pytest marker.** Lets the same test file gate
  cleanly between "no Docker — skip cleanly" and "Docker available —
  run the real DB". Implementer machines without Docker still get a
  fast test loop; CI runs the full suite.
- **`run-track-b-test` shorthand in implementer scripts.** Three-line
  alias that runs the affected suite + the `requires_docker` subset +
  the targeted Playwright spec. Cut test-cycle time for B.3a/B.3c/B.3d
  by ~50%.
- **Stub name-normalizer behind a single import.** B.1c's V1 stub +
  Track-M4 swap point pattern. Three downstream phases (B.1e, B.6,
  B.5a) imported it without needing to know about the upcoming TOOI
  integration.
- **The `notebook_event` shared event-log.** B.3b introduced it; B.3c
  (soft-nudge) and B.3d (re-extract prompt) consumed it without
  needing new tables. Track G5 webhook fan-out will consume the same
  stream — three concrete consumers, one table.

## Tooling that didn't carry its weight

- **Per-phase `reviews/phase-B.*.md` files.** Useful per-phase but the
  reviews are not cross-indexed by symptom (e.g. "schemas.py
  conflict", "Playwright route mock"). When B.3d implementer hit the
  exact issue B.3c had resolved a day earlier, the only way to find
  the resolution was full-text grep across the review tree. A
  symptom-keyed index (or a `KNOWN_ISSUES.md` rolling file) would have
  saved each subsequent implementer a search.
- **Hand-written status.md appends.** Each phase appends ~50 lines to
  the rolling file. The merge cost dominated the writing cost on
  parallel phases. As noted above, splitting into per-phase files would
  be cheaper.

---

**Closing**: Track B landed in 17 PRs over ~7 calendar days of
adversarial execution. Nine PRs APPROVED first try, eight needed an
attempt-2; zero rollbacks shipped. The reviewer-rejection rate
(~50%) was higher than Track A's (50% of 6 phases) on a much larger
phase count; the adversarial cycle continues to be the dominant
quality lever. Production multi-schema KG extraction is live and
wired end-to-end. Handover to Tracks C, D, E, F, G is complete.
