# Track F — Operations & quality (audit, librarian, resumable) — SPRINT PLAN

> **Status**: 📝 PROPOSED (2026-07-24) — **awaiting human approval**. Track-planner
> output; not yet approved for implementation. Grounded against the shipped
> `source.processing_stage` pipeline (migrations 71 + 73, `services/source_pipeline.py`)
> and the shipped `chunk_edit` audit log (migration 74, `services/audit/chunk_audit.py`).
>
> **Track ID**: `F`. Reference: `docs/FEATURE_ROADMAP.md` § "Track F — Operations &
> quality (audit, librarian, resumable)" (~line 477).

## 1. Context

**Track goal.** Give operators a per-notebook quality signal (audit), the option
to run it on a schedule (librarian), and a pipeline that resumes from the last
successful step instead of re-running from zero (resumable).

**Three sub-tracks (roadmap):**
- **F1 — Audit**: 6 LLM-free always-on checks + 2 on-demand LLM "deep audit"
  checks, persisted to an `audit_findings` table, surfaced as a dashboard widget.
- **F2 — Librarian**: cron-style periodic re-run of the F1 checks per notebook,
  opt-in. Builds on F1.
- **F3 — Resumable pipeline**: fine-grained per-step status + resume-from-last-
  successful-step. Must **reconcile with the already-shipped `processing_stage`**,
  not reinvent it.

**What already exists that F builds on (do NOT duplicate):**
- `source.processing_stage` (migration 71 add + 73 backfill). VALUES:
  `ingested | embedded | extracted | graphed | complete | awaiting_schema_review |
  failed` (an **app-level enum** stored as a bare string — new values need **no
  migration**). The pipeline is already declarative and resumable:
  `services/source_pipeline.py` (`SOURCE_PIPELINE`, `advance_source`, `StageName`)
  reads a source's current stage and dispatches the next action; every step is
  idempotent. **F3 is therefore mostly granularity + failure-provenance, not a new
  status machine.**
- `chunk_edit` audit log (migration 74) + `services/audit/chunk_audit.py`
  (`ChunkAuditService`). This is a **different concern** (chunk-level structural
  edits). F1's `audit_findings` is **notebook-level quality findings**. Naming is
  intentionally distinct; both live under `services/audit/`.
- Job/command seam: `services/command_service.py` (`_COMMAND_TO_JOB_TYPE`),
  `JobType` in `packages/shared/src/shared/types/enums.py`. F2's background job
  reuses this seam (a new command + `JobType`), mirroring V.5 `extract_references`
  / Y.3 `auto_link_note`.
- Per-notebook dashboard precedent: `frontend/src/components/notebooks/orphans/OrphansDashboard.tsx`
  + `api/routers/orphans.py` (`GET /api/notebooks/{id}/orphans`). F1's widget +
  API mirror this shape exactly (keyed by **notebook**, not source).

**Entity/graph schema the checks query** (migration 39, 50, 67, 69):
`entity{canonical_name, entity_type, confidence, community_id, status,
source_documents (FLEXIBLE array), extracted_at}`, `relation`, `mentions` edges,
`cites` edges.

**Dependencies on other tracks:** none hard. F1 reads Track B's entity graph and
Track PL's `processing_stage`; both are merged. F3 edits `source_pipeline.py`
(Track PL surface).

**Conflicts with other tracks (concurrent-file risk):**
| File | Also touched by | Mitigation |
|---|---|---|
| `services/source_pipeline.py` | any future PL phase | F3 is the only F phase that touches it; land F3 last |
| `services/command_service.py` (`_COMMAND_TO_JOB_TYPE`) | OKF.2, V.5, Y.3 already landed | F.5 appends one entry; trivial merge |
| `packages/shared/src/shared/types/enums.py` (`JobType`) | same | F.5 appends one enum member |
| `services/audit/` | I.H2 (`chunk_audit.py`) already landed | distinct filenames (`audit_service.py`); no collision |
| Migration numbering | any track adding a migration | F.1 claims **75** on merge; rebase if raced |

## 2. Decision gates (resolve before / during F.1)

Three of these **block F.1**; the rest have recommended defaults that proceed on
autopilot unless contested.

| ID | Question | Recommendation | Blocks |
|---|---|---|---|
| **F-D1** | Service location/name. Roadmap says flat `services/audit_service.py`; house convention has a `services/audit/` package (`chunk_audit.py`). | `services/audit/audit_service.py` (class `AuditService`). Keeps notebook-quality audit beside chunk audit without a flat-file clash. | **F.1** |
| **F-D2** | Findings lifecycle: snapshot-per-run (latest wins, like the mentions refresh) **vs** append-only history (like `chunk_edit`). | **Snapshot-per-run** keyed by `run_id`; dashboard reads the latest run per notebook; keep the last N runs for a trend line (prune older). Simpler dashboard semantics; trend is cheap. | **F.1** (table shape) |
| **F-D3** | Widget placement + API key. Checks are per-notebook (orphans, communities). | Per-notebook, co-located with `OrphansDashboard` in the notebook's Schema/KG tab; API `GET/POST /api/notebooks/{id}/audit` (mirrors `orphans.py`). | **F.1** (API shape) |
| F-D4 | Where check thresholds live (stale age, low-confidence band width, long-pending age, cohesion floor, fallback-ratio). | App config defaults + env override (per `config.py`), NOT per-notebook in v1. Defaults proposed per-check in F.1. | no (defaults) |
| F-D5 | Exact "citation completeness" predicate. | `entity.status IN ["active","reference"] AND array::len(source_documents) = 0` (NONE-guarded). No `mentions`/provenance join in v1. | no (default) |
| F-D6 | F3 reconciliation: extend `processing_stage` vs introduce roadmap's separate names (`chunked`/`entities_extracted`/`summarized`). | **Extend the shipped enum**: add `chunked` between `ingested` and `embedded`; map roadmap `entities_extracted`→`extracted`, `summarized`→INSIGHTS branch (not a spine stage). Keep all shipped values. | no (F.7 concern) |
| F-D7 | Deep-audit (checks 7–8) LLM model routing. | Reuse the existing model-routing seam (`services/model_routing/`, same as contradiction judge); a cheap default model, on-demand only. | no (F.3 concern) |

## 3. Phases

Convention per `docs/tracks/README.md` (Backend → UI → Integration); each phase =
one PR, own branch `track/f-<phase>`, green tests + AC before the next. Ordered by
dependency; the F1 backend + widget (F.1 + F.2) are the roadmap's flagged
"cheap, high-impact" first increment.

**Phase list (one-line goals):**
- **F.1** — `audit_findings` table (migration **75**) + `AuditService` + 6 LLM-free checks + run/read API. *(Backend — recommended first PR.)*
- **F.2** — Audit dashboard widget consuming F.1 (always-on, per-notebook).
- **F.3** — Deep audit: 2 LLM checks (conflicting facts, provenance gaps) on the F.1 engine.
- **F.4** — Deep-audit UI: "Run deep audit" trigger + surfaced findings in the widget.
- **F.5** — `LibrarianService` + opt-in periodic background job (per notebook) re-running the checks.
- **F.6** — Librarian UI: per-notebook opt-in toggle + schedule + last-run surfacing.
- **F.7** — Resumable pipeline: add `chunked` sub-stage + per-step failure provenance; resume from last successful step.
- **F.8** — Integration: end-to-end resumability + audit smoke, docs (ARCHITECTURE + roadmap), RETRO.

---

### Phase F.1 — Audit engine: findings table + 6 LLM-free checks + API (Backend)

**Goal**: A pure, LLM-free `AuditService` that runs the 6 always-on checks for a
notebook, persists results to a new `audit_findings` table, and exposes run + read
endpoints. Self-contained and shippable (no UI needed to be correct).

**Files to create**:
- `migrations/75.surrealql` + `migrations/75_down.surrealql` — `audit_findings`
  table. OVERWRITE-guarded SCHEMAFULL (per migration 74 v2-drift note):
  ```surql
  DEFINE TABLE OVERWRITE audit_findings SCHEMAFULL;
  DEFINE FIELD OVERWRITE notebook ON audit_findings TYPE record<notebook>;
  DEFINE FIELD OVERWRITE run_id   ON audit_findings TYPE string;          -- groups one run (F-D2 snapshot)
  DEFINE FIELD OVERWRITE check_id ON audit_findings TYPE string;          -- citation_completeness | stale_sources | ...
  DEFINE FIELD OVERWRITE severity ON audit_findings TYPE string ASSERT $value IN ["error","warn","info"];
  DEFINE FIELD OVERWRITE title    ON audit_findings TYPE string;
  DEFINE FIELD OVERWRITE detail   ON audit_findings TYPE option<string>;  -- JSON payload (offending ids/counts)
  DEFINE FIELD OVERWRITE subject  ON audit_findings TYPE option<string>;  -- offending record id (entity/source/community)
  DEFINE FIELD OVERWRITE count    ON audit_findings TYPE option<int>;
  DEFINE FIELD OVERWRITE created  ON audit_findings TYPE datetime DEFAULT time::now();
  DEFINE INDEX OVERWRITE af_notebook ON audit_findings FIELDS notebook;
  DEFINE INDEX OVERWRITE af_run      ON audit_findings FIELDS run_id;
  ```
  Down: `REMOVE TABLE audit_findings;` (faithful inverse — table is brand new).
- `apps/app-main/src/app_main/services/audit/audit_service.py` — `AuditService`
  with one method per check returning `List[AuditFinding]` (pure SurrealQL
  aggregates), a `run_all(notebook_id) -> run_id` that writes a snapshot, and a
  `latest(notebook_id)` reader. Each check is a small pure function so it is unit-
  testable in isolation. **The 6 checks + their default predicates/thresholds:**
  1. **citation_completeness** (`error`): `entity WHERE status IN ["active","reference"] AND array::len(source_documents) = 0` (F-D5; guard `source_documents != NONE` before `array::len`, per the SurrealDB `array::len(NONE)` rule).
  2. **stale_sources** (`info`): `source WHERE updated < time::now() - {STALE_DAYS}d` (default 180).
  3. **low_confidence_survivors** (`warn`): `entity WHERE confidence >= {FILTER_MIN} AND confidence < {FILTER_MIN}+{BAND}` (band just above the extraction filter floor).
  4. **long_pending_orphans** (`warn`): pending/orphan entities older than `{ORPHAN_DAYS}` — reuse the orphans repo query behind `api/routers/orphans.py`.
  5. **community_cohesion** (`info`): group by `entity.community_id`; flag communities whose internal-edge density < `{COHESION_FLOOR}`.
  6. **schema_drift** (`info`): ratio of `entity.entity_type` in the fallback set (`UNKNOWN` + generic) over total > `{DRIFT_RATIO}` (default 0.30).
- `apps/app-main/src/app_main/api/routers/audit.py` — `POST /api/notebooks/{id}/audit`
  (run the 6 checks now, return the run + findings) and `GET /api/notebooks/{id}/audit`
  (latest snapshot). Mirrors `orphans.py` (synchronous; work is bounded SurrealQL).
- `packages/shared/src/shared/models/audit.py` — `AuditFinding` /
  `AuditRunResponse` pydantic models (mirrors the orphans response shape).
- `apps/app-main/tests/test_audit_service.py` — per-check fixtures (see Tests).
- `apps/app-main/tests/test_audit_migration.py` — migration 75 up/down roundtrip
  (`@requires_docker`, mirrors existing migration-roundtrip tests).

**Files to modify**:
- `apps/app-main/src/app_main/api/app.py` — register the `audit` router.
- `apps/app-main/src/app_main/config.py` — the 6 threshold defaults (F-D4).
- `apps/app-main/src/app_main/services/audit/__init__.py` — export `AuditService`.

**Acceptance criteria** (falsifiable):
1. Migration 75 applies (forward) and reverts (down) cleanly on a fresh
   testcontainer AND on a drifted DB (OVERWRITE idempotent) — CI roundtrip green.
2. `POST /api/notebooks/{id}/audit` returns HTTP 200 with a body containing one
   `run_id` and a `findings` array; each finding carries `check_id`, `severity ∈
   {error,warn,info}`, `title`, and (where applicable) `subject`/`count`.
3. All 6 `check_id`s are represented in the runner output for a seeded notebook
   that triggers each condition; a clean notebook yields zero findings (not an
   error).
4. Each check is LLM-free: `run_all` issues **no** model calls (asserted by a
   no-network / mocked-model-router guard in the test).
5. `citation_completeness` flags exactly the seeded entities with empty
   `source_documents` and no others; the `array::len(NONE)` guard is present (a
   notebook with `source_documents = NONE` entities does not 500).
6. `GET /api/notebooks/{id}/audit` returns the latest run's snapshot; a second
   `POST` supersedes it (F-D2: latest `run_id` wins; older runs beyond the keep-N
   are pruned).
7. Thresholds are read from config, not hardcoded in the service body (grep: no
   numeric literals in the predicate builders).

**Tests required**:
- Unit (`test_audit_service.py`, `@requires_docker` where a seeded graph is
  needed): 1 fixture per check that (a) triggers it and (b) a negative fixture
  that does not; plus a `run_all` snapshot-write + `latest` read; plus the
  no-LLM-calls assertion.
- Integration: `POST` then `GET` on a seeded notebook; assert the 6 checks and the
  snapshot supersede semantics.
- Migration roundtrip (`test_audit_migration.py`).

**PR boundary**: ONE PR titled `feat(audit): audit_findings table + 6 LLM-free checks + API (F.1)`.
No UI, no LLM, no job-queue.

**Effort estimate**: 3–4 days (roadmap "Phase 3a" = 4–5d incl. widget; the widget
is split to F.2). Reviewer cycle budget: ×1.5.

**Risk mitigations**: pure SurrealQL — no model dependency; per-check isolation
keeps a single broken predicate from failing the whole run (wrap each check
best-effort, record a `check_error` finding rather than raising).

---

### Phase F.2 — Audit dashboard widget (UI)

**Goal**: An always-on per-notebook audit widget rendering the F.1 findings,
grouped by severity, with a manual "Re-run" action. Completes the roadmap's
"always-on dashboard widget". Mirrors `OrphansDashboard`.

**Files to create**:
- `frontend/src/components/notebooks/audit/AuditWidget.tsx` — severity-grouped
  findings list (error/warn/info), per-finding subject link, counts, last-run
  timestamp, "Re-run" button.
- `frontend/src/components/notebooks/audit/useAudit.ts` (or the repo's hook
  location) — `GET`/`POST /api/notebooks/{id}/audit` data hook, mirroring the
  orphans hook.
- `frontend/e2e/track-f/audit-widget.spec.ts` — render states + re-run flow.

**Files to modify**:
- The notebook Schema/KG tab container (co-located with `OrphansDashboard`) —
  mount `<AuditWidget notebookId=… />`.

**Acceptance criteria**:
1. Widget renders all four states: default (findings present), loading, error, and
   empty (zero findings → an explicit "No issues found" state, not a blank).
2. Findings are grouped by severity with an error/warn/info count header; each
   finding shows its `title` and, when `subject` is set, links to the entity/
   source/community.
3. "Re-run" calls `POST /api/notebooks/{id}/audit` and refreshes without a full
   page reload; button is disabled while in flight.
4. Keyboard-accessible: Tab reaches the Re-run button and each finding link;
   severity groups have `aria-label`s; the widget region has `role="region"`.
5. No regression in the adjacent `OrphansDashboard`.

**Tests required**:
- Component unit: render each of the 4 states from mocked hook data; severity
  grouping; keyboard focus order.
- E2E (`audit-widget.spec.ts`): open a seeded notebook → widget renders findings →
  click Re-run → findings refresh.

**PR boundary**: ONE PR titled `feat(frontend): per-notebook audit widget (F.2)`.
Assumes F.1 merged.

**Effort estimate**: 1.5–2 days. Reviewer cycle budget: ×1.5.

---

### Phase F.3 — Deep audit: 2 LLM checks (Backend)

**Goal**: On-demand LLM-backed checks 7 (conflicting facts) and 8 (provenance
gaps), reusing the F.1 engine + `audit_findings` table. On-demand only — never in
the always-on path.

**Files to create**:
- `apps/app-main/src/app_main/services/audit/deep_audit_service.py` —
  `DeepAuditService` with `conflicting_facts(notebook_id)` (attribute Y=A in S1 vs
  Y=B in S2 across a shared entity) and `provenance_gaps(notebook_id)` (relations
  whose `source_documents` carry no supporting evidence). Uses the model-routing
  seam (F-D7); writes findings with `check_id ∈ {conflicting_facts, provenance_gaps}`,
  severity `warn`, tagged to the same `run_id` scheme (a separate "deep" run).
- `apps/app-main/tests/test_deep_audit_service.py` — mocked-LLM fixtures.

**Files to modify**:
- `apps/app-main/src/app_main/api/routers/audit.py` — add
  `POST /api/notebooks/{id}/audit/deep` (runs checks 7–8; explicitly separate route
  so the cheap path never triggers LLM calls).

**Acceptance criteria**:
1. `POST /api/notebooks/{id}/audit/deep` returns findings for checks 7 and 8;
   `POST /api/notebooks/{id}/audit` (F.1) still issues zero model calls (regression
   guard from F.1 AC4 stays green).
2. `conflicting_facts` flags a seeded entity with contradictory attribute values
   across two sources and produces a finding naming both sources.
3. `provenance_gaps` flags a seeded relation with empty/unsupported
   `source_documents` and not a well-cited one.
4. LLM calls route through the existing model-routing seam (no bespoke client);
   the check is fail-soft (a model error yields a `check_error` finding, not a 500).
5. Deep findings persist to `audit_findings` and are readable via the F.1 `GET`
   (distinguishable by `check_id`).

**Tests required**:
- Unit: mocked LLM returning a known conflict/gap → assert finding shape; model-
  error → fail-soft.
- Integration: `POST …/audit/deep` on a seeded notebook with mocked router.

**PR boundary**: ONE PR titled `feat(audit): deep audit — conflicting facts + provenance gaps (F.3)`.
Assumes F.1 merged; independent of F.2.

**Effort estimate**: 3–4 days (roadmap "Phase 3b" ≈ 1 week incl. UI; UI split to
F.4). Reviewer cycle budget: ×1.5.

---

### Phase F.4 — Deep-audit UI trigger + surfacing (UI)

**Goal**: A "Run deep audit" action in the widget that calls F.3 and surfaces
checks 7–8 findings alongside the always-on ones, visually marked as LLM-derived.

**Files to modify**:
- `frontend/src/components/notebooks/audit/AuditWidget.tsx` — add a "Run deep
  audit" button (separate from Re-run), a loading state for it, and rendering of
  `conflicting_facts`/`provenance_gaps` findings with an "LLM" badge.
- `frontend/src/components/notebooks/audit/useAudit.ts` — add the
  `POST …/audit/deep` mutation.

**Acceptance criteria**:
1. "Run deep audit" triggers the F.3 endpoint; a distinct in-flight state (not the
   always-on Re-run spinner) is shown; button disabled while running.
2. Checks 7–8 findings render with an "LLM" badge, grouped with existing findings
   by severity.
3. The always-on widget still renders correctly when the user never runs the deep
   audit (no findings of those `check_id`s → no empty section).
4. Keyboard-accessible; the new button is in Tab order with an `aria-label`.

**Tests required**:
- Component unit: deep-audit button states; LLM-badge rendering.
- E2E: run deep audit on a seeded notebook → conflict/gap finding appears.

**PR boundary**: ONE PR titled `feat(frontend): deep-audit trigger in audit widget (F.4)`.
Assumes F.2 + F.3 merged.

**Effort estimate**: 1–1.5 days. Reviewer cycle budget: ×1.0.

---

### Phase F.5 — Librarian: opt-in periodic background job (Backend)

**Goal**: A `LibrarianService` that re-runs the F.1 checks (optionally F.3) per
notebook on a schedule, opt-in per notebook, via the existing job-queue seam.
Builds on F1.

**Files to create**:
- `apps/app-main/src/app_main/services/audit/librarian_service.py` —
  `LibrarianService`: for each opt-in notebook, invoke `AuditService.run_all` and
  record the run; emit a summary. Pure orchestration over F.1.
- A handler for the new command (in `apps/app-main/src/app_main/handlers.py`) that
  the worker dispatches, mirroring `auto_link_note` / `extract_references`.
- `apps/app-main/tests/test_librarian_service.py` — enqueue-at-the-seam +
  consumer-runs-audit tests (per the job-queue-singleton test pattern: assert
  ENQUEUE by mocking `submit_command_job`, and exercise the CONSUMER via the real
  handler with `config=live_surrealdb`).

**Files to modify**:
- `packages/shared/src/shared/types/enums.py` — add `JobType.LIBRARIAN_AUDIT = "librarian_audit"`.
- `apps/app-main/src/app_main/services/command_service.py` — add
  `"run_librarian_audit": JobType.LIBRARIAN_AUDIT` to `_COMMAND_TO_JOB_TYPE`.
- Notebook opt-in surface: a per-notebook `librarian_enabled` toggle (+ optional
  interval). Prefer a schemaless notebook setting to avoid a migration; **if** a
  strict `notebook` field is chosen, that consumes migration **76** with an S.4
  backfill (per migration 71's forward-guard rule). See F-D2/effort note.
- The scheduler entry point that enqueues `run_librarian_audit` per opt-in
  notebook on the interval (reuse the existing worker/scheduling surface; do not
  add a new always-on daemon if the repo already has a periodic hook).

**Acceptance criteria**:
1. Enabling librarian on a notebook and firing the scheduler enqueues exactly one
   `run_librarian_audit` job for that notebook (asserted at the `submit_command_job`
   seam); a disabled notebook enqueues none.
2. The consumer handler runs `AuditService.run_all` and writes a fresh
   `audit_findings` snapshot (asserted via the real handler with `live_surrealdb`).
3. Opt-in is off by default (no behaviour change for existing notebooks).
4. The job is fail-soft: an audit error on one notebook does not abort the others.
5. If migration 76 is used: it applies + reverts cleanly with a drift-only S.4
   backfill (healthy DB touches 0 rows).

**Tests required**:
- Unit: enqueue-at-seam (mock `submit_command_job`); disabled-notebook → no
  enqueue; consumer runs audit (real handler + `live_surrealdb`).
- Migration roundtrip if 76 is used.

**PR boundary**: ONE PR titled `feat(audit): opt-in librarian periodic audit job (F.5)`.
Assumes F.1 merged.

**Effort estimate**: 3–4 days (roadmap F2 ≈ 1 week incl. UI; UI split to F.6).
Reviewer cycle budget: ×1.5.

**Risk mitigations**: reuses the proven command/job seam (no new infra); opt-in
default-off means zero impact until an operator turns it on.

---

### Phase F.6 — Librarian UI: opt-in toggle + last-run (UI)

**Goal**: Per-notebook librarian controls — enable/disable, interval, last-run
timestamp — in notebook settings, beside the audit widget.

**Files to create**:
- `frontend/src/components/notebooks/audit/LibrarianSettings.tsx` — toggle +
  interval select + last-run readout.

**Files to modify**:
- Notebook settings container — mount `<LibrarianSettings />`.
- `frontend/src/components/notebooks/audit/useAudit.ts` — mutation to
  enable/disable librarian + read last-run.

**Acceptance criteria**:
1. Toggling librarian persists the opt-in and reflects it on reload.
2. Last-run timestamp displays after a run; "never run" state shown before.
3. All states render (default/loading/error/empty); keyboard-accessible toggle +
   select with `aria-label`s.
4. Disabling stops future scheduled runs (verified against the F.5 enqueue guard).

**Tests required**:
- Component unit: toggle + interval states; last-run rendering.
- E2E: enable librarian → last-run appears after a triggered run.

**PR boundary**: ONE PR titled `feat(frontend): librarian opt-in settings (F.6)`.
Assumes F.5 merged.

**Effort estimate**: 1–1.5 days. Reviewer cycle budget: ×1.0.

---

### Phase F.7 — Resumable pipeline: `chunked` sub-stage + failure provenance (Backend)

**Goal**: Close the granularity gap in the shipped `processing_stage` so the
pipeline resumes from the last *successful step* rather than restarting, and record
*which* step failed instead of a bare `failed`. **Reconcile with `processing_stage`
(F-D6) — do NOT introduce a parallel status field.**

**Reconciliation (F-D6):** shipped VALUES stay
(`ingested|embedded|extracted|graphed|complete|awaiting_schema_review|failed`).
Add **`chunked`** between `ingested` and `embedded` (the one genuinely-missing
spine step: parse→chunk→embed). Map the roadmap's proposed names:
`extracted`(roadmap) ≡ shipped `extracted`; `entities_extracted` ≡ `extracted`;
`summarized` = the INSIGHTS parallel branch (enrichment, NOT a spine stage — it
never gates `complete`, per `source_pipeline.py`). New values are an app-level enum
→ **no migration for the value itself**.

**Files to create**:
- `apps/app-main/tests/test_resumable_pipeline.py` — resume-from-each-stage +
  failure-provenance tests.

**Files to modify**:
- `apps/app-main/src/app_main/services/source_pipeline.py` — insert the `chunked`
  stage into `SOURCE_PIPELINE` + the stage→value map; `advance_source` dispatches
  embed from `chunked`, chunk from `ingested`. Preserve the chunk-count guard and
  the schema-review gate exactly.
- `apps/app-main/src/app_main/handlers.py` — write `chunked` after chunking
  succeeds; on failure, record the failing stage (see failure-provenance below).
- Failure provenance: record the stage that failed. Prefer **no new field** —
  encode as `processing_stage = "failed"` plus a best-effort `failed_stage` written
  to an existing flexible/metadata field on `source`. **If** a dedicated strict
  `source.failed_stage` field is chosen instead, it consumes migration **77** with
  an S.4 drift-only backfill. (Decision deferred into this phase.)

**Acceptance criteria**:
1. A source interrupted after chunking is at `processing_stage = "chunked"`;
   re-driving via `advance_source` resumes at embed (does **not** re-parse/re-chunk).
2. Resume from every stage is idempotent: driving `ingested→…→complete` twice
   yields identical graph/embedding output (embed deletes-then-writes; extract
   dedups) — no duplication.
3. A failure at stage X leaves the source recoverable: the failing stage is
   recorded (in `failed_stage` field or metadata), and re-driving resumes at X, not
   from `ingested`.
4. The shipped VALUES are unchanged in meaning; migration 73's backfill semantics
   still hold (a `complete` source stays `complete`).
5. INSIGHTS remains a parallel best-effort branch that does not set
   `processing_stage` and never gates `complete` (regression guard).
6. If migration 77 is used: applies + reverts with a drift-only S.4 backfill.

**Tests required**:
- Unit (`test_resumable_pipeline.py`): resume from each of `ingested/chunked/
  embedded/extracted/graphed`; failure at chunk/embed/extract → correct
  `failed_stage` + resume point; idempotent double-drive.
- Integration: full ingest with an injected mid-pipeline failure → assert resume
  completes without duplicate entities/chunks.

**PR boundary**: ONE PR titled `feat(pipeline): chunked sub-stage + resumable step recovery (F.7)`.
Land **last** (touches the shared `source_pipeline.py`).

**Effort estimate**: 3–5 days (roadmap F3 ≈ 1 week). Reviewer cycle budget: ×1.5.

**Risk mitigations**: this is additive to a *proven* resumable driver — the risk is
regressing the existing chain, mitigated by the idempotency + INSIGHTS-branch
regression guards (AC2, AC5). No parallel status field (F-D6) avoids a second drift
surface against `processing_stage`.

---

### Phase F.8 — Integration: e2e + docs + RETRO (Integration)

**Goal**: End-to-end validation across the sub-tracks and documentation close-out.

**Tasks**:
- E2E: full user flow — ingest a source, open its notebook, run the always-on
  audit, run a deep audit, enable the librarian, and confirm resume-from-failure
  behaviour on an interrupted ingest.
- Update `docs/ARCHITECTURE.md` — add the audit/librarian services + `audit_findings`
  table + the `chunked` stage to the pipeline description.
- Update `docs/FEATURE_ROADMAP.md` — mark Track F status.
- `docs/tracks/F-operations/status.md` + `RETRO.md`.
- Manual smoke checklist (audit widget states, deep-audit badge, librarian toggle,
  resume-from-`chunked`).

**Acceptance criteria**:
1. The full flow (audit → deep audit → librarian → resume) works end-to-end on a
   dev container.
2. `ARCHITECTURE.md` and `FEATURE_ROADMAP.md` reflect the shipped state; the
   `chunked` stage is documented alongside the existing `processing_stage` values.
3. Full-suite regression green; track marked CLOSED in `_status.md`.

**PR boundary**: ONE PR titled `docs(f): audit/librarian/resumable integration + RETRO (F.8)`.
Assumes all backend + UI phases merged.

**Effort estimate**: 1–2 days. Reviewer cycle budget: ×1.0.

## 4. Risk assessment

- **Risk: a single broken check predicate fails the whole always-on run.**
  Mitigation: per-check best-effort in `run_all` — a failing check writes a
  `check_error` finding rather than raising (F.1 AC + risk note).
- **Risk: `array::len(NONE)` on nullable `source_documents`/`embedding` 500s the
  audit** (recurring SurrealDB footgun). Mitigation: NONE-guard before `array::len`
  in every predicate (F.1 AC5); covered by a negative fixture.
- **Risk: F.7 regresses the shipped resumable chain** (the highest-value existing
  behaviour). Mitigation: additive stage only; idempotency + INSIGHTS-branch
  regression guards; land last; no parallel status field (F-D6).
- **Risk: deep-audit LLM cost/latency leaks into the always-on path.** Mitigation:
  separate endpoint + explicit no-model-calls regression guard on the cheap path
  (F.1 AC4 / F.3 AC1).
- **Risk: librarian scheduler double-enqueues or runs on disabled notebooks.**
  Mitigation: enqueue-at-seam test + disabled→no-enqueue test (F.5 AC1); opt-in
  default-off.
- **Risk: migration 75 raced by a concurrent track.** Mitigation: OVERWRITE-guarded
  DEFINEs (idempotent on drift) + rebase-on-merge if the number is taken.

## 5. Test strategy summary

- **Unit test files to create**: `test_audit_service.py`, `test_deep_audit_service.py`,
  `test_librarian_service.py`, `test_resumable_pipeline.py` (all with per-check /
  per-stage positive+negative fixtures; `@requires_docker` where a seeded graph is
  needed).
- **Migration roundtrip**: `test_audit_migration.py` (75); + 76/77 roundtrips if
  those fields are added (CI-gated forward/down).
- **Integration**: audit `POST`→`GET` supersede; deep-audit route; librarian
  enqueue-seam + real-handler consumer (`live_surrealdb`); resume-from-failure full
  ingest.
- **E2E (Playwright)**: `audit-widget.spec.ts` (F.2), deep-audit trigger (F.4),
  librarian settings (F.6), full-flow smoke (F.8).

## 6. Open questions (escalate to user before F.1)

- **F-D1 (blocks F.1)**: confirm `services/audit/audit_service.py` (rec) vs the
  roadmap's flat `services/audit_service.py`.
- **F-D2 (blocks F.1)**: confirm snapshot-per-run (rec) vs append-only history for
  `audit_findings`; and the keep-N for the trend line.
- **F-D3 (blocks F.1)**: confirm per-notebook widget placement + `GET/POST
  /api/notebooks/{id}/audit` API shape (rec, mirrors orphans).
- F-D4/F-D5 (defaults, non-blocking): confirm the 6 threshold defaults and the
  citation-completeness predicate, or accept the recommended values.
- F-D6 (F.7): confirm extending `processing_stage` with `chunked` (rec) over a
  separate `source.status` field.
- F-D7 (F.3): confirm reusing the existing model-routing seam + a cheap default
  model for deep audit.

## 7. Dependency map

```
F.1 (Backend: table 75 + 6 checks + API)   ← recommended FIRST PR
 ├─→ F.2 (UI widget)                        — requires F.1 API
 ├─→ F.3 (Backend: 2 LLM checks)            — requires F.1 engine
 │      └─→ F.4 (UI deep-audit trigger)     — requires F.2 + F.3
 └─→ F.5 (Backend: librarian job)           — requires F.1 engine
        └─→ F.6 (UI librarian settings)     — requires F.5

F.7 (Backend: resumable pipeline)           — independent of F1/F2; land LAST (shared source_pipeline.py)

F.8 (Integration + docs + RETRO)            — requires all above
```

**Migration budget**: F.1 → **75** (definite). F.5 → **76** (only if a strict
`notebook.librarian_enabled` field is chosen over a schemaless setting). F.7 →
**77** (only if a strict `source.failed_stage` field is chosen over metadata
encoding). Claim numbers at merge time; OVERWRITE guards make a raced rebase safe.

**Cross-track**: this track depends on **none** (Track B graph + Track PL pipeline
already merged). This track blocks **none**; F.7's `source_pipeline.py` edit should
be sequenced after any in-flight PL work to avoid a merge race.

## 8. Ordering & effort

| Phase | Title | Effort (days) | PR | Migration |
|---|---|---|---|---|
| F.1 | Audit engine: table + 6 checks + API | 3–4 | 1 | **75** |
| F.2 | Audit dashboard widget | 1.5–2 | 1 | — |
| F.3 | Deep audit: 2 LLM checks | 3–4 | 1 | — |
| F.4 | Deep-audit UI trigger | 1–1.5 | 1 | — |
| F.5 | Librarian periodic job | 3–4 | 1 | 76? |
| F.6 | Librarian UI settings | 1–1.5 | 1 | — |
| F.7 | Resumable pipeline (`chunked` + recovery) | 3–5 | 1 | 77? |
| F.8 | Integration + docs + RETRO | 1–2 | 1 | — |
| **Total (critical path)** | | **~17–24 days** | **8 PRs** | |
| **With ×1.5 reviewer budget** | | **~26–36 days** | | |

**Recommended starting PR**: **F.1** — the roadmap's flagged "cheap, high-impact"
increment (6 LLM-free checks + `audit_findings` + API), self-contained, no LLM/UI/
job dependency. Pair with **F.2** immediately after to deliver the always-on
dashboard widget the roadmap describes.
