# Review — Track B Phase B.7 attempt 1

**Branch**: `track/b-integration-retro`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-12

## Summary

Docs-only phase confirmed (5 files, 452 insertions, 1 deletion across
`ARCHITECTURE.md`, `CHANGELOG.md` [new], `docs/FEATURE_ROADMAP.md`,
`docs/tracks/B-kg-quality/RETRO.md` [new], `docs/tracks/B-kg-quality/status.md`).
Content quality is high — the RETRO captures the right autopilot
lessons for tracks C-G with concrete, evidence-backed entries; the
ARCHITECTURE additions are factually accurate against the live code;
the CHANGELOG operator notes are deploy-grade. One factual
inconsistency on phase/PR count repeated in two places (RETRO + the
FEATURE_ROADMAP Track-B banner) blocks approval at the agreed HIGH
quality bar.

## Acceptance criteria check

The plan's Phase B.7 ACs (lines 690-696) are five entries. AC1 and AC2
require a full corpus E2E run and a cross-notebook merge sanity run,
both captured in `docs/tracks/B-kg-quality/E2E_EVIDENCE.md`. The
implementer **deferred** both with explicit, well-reasoned justification
in the RETRO ("Live-test recommendation" section) and the status.md
entry; the user brief also frames this as "a docs-only phase". I am
treating that deferral as authorised. ACs 3-5 are in scope.

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | Full corpus E2E (5 mixed-domain docs) + `E2E_EVIDENCE.md` | DEFERRED | Explicit deferral in RETRO §Live-test recommendation; rationale = per-phase Playwright specs already exercise production paths. User brief sanctions doc-only. |
| 2 | Cross-notebook merge sanity check captured in evidence doc | DEFERRED | Same deferral envelope as AC1. |
| 3 | `ARCHITECTURE.md` reflects new tables + new service modules | ✅ | New §6 "Storage layer additions" lists migrations 44-48, all 4 new tables, the 3 `entity` field additions, all 6 Track-B services, and the multi-schema dispatcher pattern. Cross-checked against migration files + actual service paths (see "Spot-checks" below). |
| 4 | `FEATURE_ROADMAP.md` reflects ALL phases complete | ✅ (with arithmetic error — see Major #1) | All 18 sub-phases in the table; all marked ✅; links to RETRO + status.md resolve. |
| 5 | `RETRO.md` exists with ≥5 worked / ≥3 hurt / ≥5 recommendations | ✅ | Actual counts: 7 "what worked", 6 "what hurt", 7 "recommendations". All entries concrete and evidence-bound (no vague platitudes). |

## Spot-checks against production code

| Claim in docs | Live-code verification |
|---|---|
| Migrations 44-48 cover the listed tables | `migrations/44-48.surrealql` headers + `DEFINE TABLE/FIELD` lines all match the per-row purpose in ARCHITECTURE §6 and CHANGELOG §Migrations. |
| `metrics` table has composite index on `(event_type, created_at)` | `migrations/47.surrealql:44` — `DEFINE INDEX ... idx_metrics_event_created ON metrics FIELDS event_type, created_at;` ✓ |
| Env-flag `OPEN_NOTEBOOK_DISABLE_METRICS=1` opts out | `packages/shared/src/shared/services/metrics.py:35` declares `_DISABLE_FLAG`; line 45 short-circuits on `=="1"`; `conftest.py` and the test module both exercise it. ✓ |
| `shared.services.metrics.record_metric` is the helper | `packages/shared/src/shared/services/metrics.py:48 async def record_metric(...)` ✓ |
| `entity_filtering.resolution.orphan_connector` location | `pipelines/entity-filtering/src/entity_filtering/resolution/orphan_connector.py` exists ✓ (note: lives in `pipelines/`, not `apps/`, but ARCHITECTURE's "unless noted otherwise" caveat covers it via fully-qualified path) |
| `SchemaEditService`, `ReextractService`, `NotebookMergeService` paths | All three exist at `apps/app-main/src/app_main/services/`. ✓ |
| Orphan prune-lifecycle defaults `max_attempts=3`, age=90 days | `orphan_prune.py:83 DEFAULT_MAX_ATTEMPTS = 3`, `:86 DEFAULT_MAX_AGE_DAYS = 90`. CHANGELOG says `age_threshold=90 days` — the kwarg is actually `max_age_days`, but operator-facing name is intelligible. Logged as Minor #1. |
| `notebook_event` consumed by soft-nudge banner | `frontend/src/components/notebooks/schema/SchemaSoftNudge.tsx:12-79` uses `useNotebookEvents` hook backed by `notebook_event` table. ✓ |
| Migration 45 declares both `notebook_schema` and `pass1_results` | `migrations/45.surrealql` has both `DEFINE TABLE IF NOT EXISTS ...` ✓ |
| Docs-only — no stealth code | `git diff --stat main...origin/track/b-integration-retro` returns 5 doc files only. ✓ |

## Issues found

### 🔴 Blockers (must fix)

(none)

### 🟡 Major (must fix)

1. **Phase / PR arithmetic is wrong by one, repeated in two places.**

   - `docs/tracks/B-kg-quality/RETRO.md` (multiple sites):
     - Line 21-22: "Track B (KG quality) closed on 2026-06-12 after **17 sub-phases** delivered **across 17 PRs**"
     - "Pattern: 9/17 phases passed first try (53%); 8/17 (47%) needed a second attempt"
     - "Closing: Track B landed in **17 PRs** over ~7 calendar days"
   - `docs/FEATURE_ROADMAP.md:129-131`:
     - "All **17 sub-phases** (B.0, B.1a–B.1f, B.2a/B.2b, B.3a–B.3d, B.4, B.5a/B.5b, B.6, B.7) merged across **17 PRs**"

   The enumerated list literally contains 18 entries: `B.0` (1) + `B.1a..B.1f` (6) + `B.2a, B.2b` (2) + `B.3a..B.3d` (4) + `B.4` (1) + `B.5a, B.5b` (2) + `B.6` (1) + `B.7` (1) = **18**. The FEATURE_ROADMAP status table renders 18 rows. The RETRO header lists 18 branches merged. The RETRO's attempt-count table has 17 rows because it (correctly) excludes B.7 — but that doesn't make the total 17, it just means the attempt-table is "all production sub-phases excluding the integration phase".

   Either commit the count to 18 everywhere and have the attempt-rate sentences read "9/17 production sub-phases passed first try (53%)" (which is the meaningful denominator anyway), or commit to 17 and exclude B.7 from the FEATURE_ROADMAP banner enumeration. As-is, a reader who compares the banner enumeration to the count loses trust in the rest of the doc — exactly the failure mode the HIGH quality bar exists to prevent.

   **Recommendation**: prefer "18 PRs across 17 production sub-phases + 1 integration/retro phase" — keeps the rejection-rate denominator (17) honest and the PR count accurate.

### 🔵 Minor (optional)

1. **CHANGELOG operator note uses `age_threshold` for a kwarg actually named `max_age_days`.** `CHANGELOG.md:Migration notes` says "the orphan prune-lifecycle archive rule defaults to `max_attempts=3` and `age_threshold=90 days`". The actual Python kwarg is `max_age_days` (`orphan_prune.py:483`). Naming inconsistency for operators who grep the codebase. Suggest "and `max_age_days=90`".

2. **No explicit hyperlinks from RETRO to specific review files.** The attempt-count table is the natural place to link, e.g. `| B.1f | 2 | [REVISIONS](./reviews/phase-B.1f-attempt-1.md) — three ACs over-claimed... |`. Track A's RETRO uses the same glob-only reference (`reviews/phase-A.*-attempt-N.md`), so the omission is consistent with the precedent — but the precedent is suboptimal; an inversion-test reviewer six months from now (Track G) trying to find the B.6 idempotency-mock review fragment will have to grep. Optional but high-leverage for the cross-track audience the RETRO is explicitly targeting.

3. **Track-B service module table claims the modules "live in `apps/app-main/src/app_main/services/` unless noted otherwise".** The `orphan_connector` row carries a fully-qualified module path (`entity_filtering.resolution.orphan_connector`) but a reader without the context of the workspace layout may not realise that's a `pipelines/` module, not an `apps/` module. Consider an explicit "(in `pipelines/entity-filtering/`)" parenthetical on that one row for symmetry with the rest of the table.

4. **The FEATURE_ROADMAP status banner duplicates its one-line summary verbatim within itself.** "Multi-schema KG extraction, schema-edit UX, telemetry, orphan-lifecycle, cross-notebook merge" appears in line 127-128 AND line 135-136 (after the "One-line summary:" marker). The second instance is redundant; consider dropping the explicit "**One-line summary**" sentence or moving it elsewhere (e.g. release-notes line). Reads as copy-paste residue.

## Kudos

The RETRO captures the autopilot lessons future tracks need with a
candour that is rare. Specific callouts:

- **Honesty about implementer over-claims** is exactly the most useful
  lesson for tracks C-G. The "What hurt" entry on B.1f / B.5b / B.6
  names the bug class concretely ("transitive-coverage assumption",
  "no production caller", "mock that advanced `updated_at` on read
  not write") — these are debuggable patterns the next implementer
  can grep for, not vague "tests were thin". This is the platinum
  standard.
- **The inversion-test recommendation (Reco #6)** is the right
  generalisation of the B.6 attempt-2 reviewer pattern. Encoding it
  in the reviewer template for tracks C-G converts a one-time
  intuition into durable institutional memory.
- **The shared-file rebase friction story (Reco #2)** is operationally
  important — the next implementer who is tempted to parallelise
  Track-C Writer-Evaluator-Editor stages has a concrete cautionary
  tale to point to.
- **The "telemetry-first" inversion of Track A's RETRO #5** is
  genuinely instructive: A's RETRO framed metrics as "could be added
  as a B-track piggy-back", B's RETRO inverts it to "should be the
  first feature, not the last". This is the kind of cross-track
  learning the RETRO format was meant to surface.
- **Pre-existing-bugs-from-B.0 documented** — the entity persistence
  drift, rdflib missing dep, LLMManager broken DI are all called out
  by name in the "Testcontainers harness" worked-entry. Closes the
  loop on Track A's RETRO #1.
- **Deferred work explicitly catalogued, not lost** — corpus E2E in
  the "Live-test recommendation" section, Q9 vocab stack via the
  name-normalizer stub pointer to Track M4. Two future-work items
  that future-me will not have to re-discover.
- **CHANGELOG is deploy-grade.** A reader pulling main and reading
  this file knows (a) which migrations to run and in what order,
  (b) which env-flag toggles which behaviour, (c) what the multi-
  schema default is and how to flip it, (d) how the archived-orphan
  recovery works. That is what a CHANGELOG is for.

## Decision rationale

Track B's docs deliverables are substantively complete and the RETRO
is genuinely high-signal — it captures the autopilot lessons future
tracks need with the right blend of specificity and abstraction. There
are no blockers and the deferral of AC1/AC2 is explicitly sanctioned.
However, a factual inconsistency in the headline phase/PR count
appears in two public-facing locations (RETRO and FEATURE_ROADMAP) and
is exactly the kind of provable-against-the-table arithmetic error
that erodes trust in the rest of the document. Under the user's
explicit "HIGH quality bar for docs" direction this is a Major and
forces REVISIONS_NEEDED.

Fix is mechanical: pick a denominator (recommended: 18 PRs, 17
production sub-phases + 1 integration), make the language consistent,
re-check the rejection-rate sentences for the same denominator.
ETA: 15 minutes.

## Next steps

REVISIONS_NEEDED. Implementer should:

1. Reconcile the phase/PR count in `RETRO.md` (header, "What worked"
   wrappers, "Pattern" sentence, "Closing" paragraph) and in
   `docs/FEATURE_ROADMAP.md` Track-B status banner. Suggest "18 PRs
   across 17 production sub-phases + 1 integration/retro phase
   (B.7)".
2. Optionally address Minors #1-#4 in the same revision.
3. Re-submit for a fast second review.

This is the LAST review of Track B. Once the counting issue is
corrected, the RETRO is ready to inform Tracks C-G's planning.
