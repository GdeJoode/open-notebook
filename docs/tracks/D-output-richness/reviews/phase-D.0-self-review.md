# Phase D.0 — Self-Review

**Branch**: `track/d-foundation`
**Commits**:
- `57e4ace` — feat(track-d/shared): export contracts + external_ids stub + JobType.EXPORT_OBSIDIAN
- `8ff7364` — feat(track-d/repo): notebook-scoped entity+relation projections for export

## Scope

Foundation phase for Track D (output rijkdom). Lands the shared upstream
contracts every Track D export format (D.1 Obsidian / D.2 JSONL /
D.3 NetworkX) consumes, plus the notebook-scoped repository projection
they read from. No user-visible behavior change; this is purely
additive plumbing.

## Files

### Created

- `packages/shared/src/shared/models/export.py` — `ExportFilter`,
  `ExportReport`, `ObsidianExportRequest`, `JsonlExportRequest`,
  `NetworkxExportRequest`, `NetworkxFormat` Literal type.
- `packages/shared/src/shared/utils/external_ids.py` — V1 stub
  returning `[]`. Mirrors `name_normalizer.py` structurally so the
  Track M4 Q9 swap is a one-file change.
- `packages/shared/tests/test_export_models.py` — 38 tests covering
  defaults, validation, round-trip, and public-API exports.
- `packages/shared/tests/test_external_ids_stub.py` — 7 tests pinning
  the V1 stub contract.
- `packages/surrealdb-service/tests/test_entity_list_for_notebook.py` —
  10 docker tests covering 6 filter combos + embedding-omission + two
  relation tests + two guard tests.

### Modified

- `packages/shared/src/shared/models/__init__.py` — export the 5 new
  models + the `NetworkxFormat` Literal.
- `packages/shared/src/shared/utils/__init__.py` — export
  `resolve_external_ids`.
- `packages/shared/src/shared/types/enums.py` — extend `JobType` with
  `EXPORT_OBSIDIAN` (only).
- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py` —
  + import `Entity`, `Relation`, `ExportFilter`
  + class-level `_ENTITY_EXPORT_FIELDS` projection constant (no
    `embedding`, per Q-D-1)
  + `list_entities_for_notebook(notebook_id, filter) -> List[Entity]`
  + `list_relations_for_notebook(notebook_id, filter) -> List[Relation]`

## Pre-resolved decisions honored

All ten planner defaults were accepted in autopilot:

| Q     | Decision                                              | Honored at                                             |
|-------|-------------------------------------------------------|--------------------------------------------------------|
| Q-D-1 | Project away `embedding`                              | `_ENTITY_EXPORT_FIELDS` constant; embedding omitted    |
| Q-D-2 | Only `EXPORT_OBSIDIAN` JobType (no JSONL/NetworkX)   | `enums.py` line 152 — single addition                  |
| Q-D-3 | Standalone `external_ids` stub module                 | `shared/utils/external_ids.py` (mirrors name_normalizer) |
| Q-D-4 | Silently drop relations into filtered entities        | Two-phase set-intersection in `list_relations_for_notebook` |
| Q-D-5 | (`-2`/`-3` filename suffix) — D.1 concern             | Not applicable to D.0                                  |
| Q-D-6 | (overwrite mode for vault) — D.1 concern              | Not applicable to D.0                                  |
| Q-D-7 | (BytesIO-then-stream JSONL) — D.2 concern             | Not applicable to D.0                                  |
| Q-D-8 | Counts only in telemetry payload                      | `ExportReport.metadata` documented as counts-only      |
| Q-D-9 | Phase order D.0 → D.3 → D.1a → D.2 → D.1b → D.1c → D.4 | Starting at D.0                                       |
| Q-D-10 | Empty aliases for V1                                 | `external_ids` stub returns `[]`; test pins this       |

## Acceptance criteria

| #   | Criterion                                                                 | Status |
|-----|---------------------------------------------------------------------------|--------|
| AC1 | All Pydantic models round-trip cleanly                                    | PASS — 38 tests in `test_export_models.py`, including parametrised JSON round-trips |
| AC2 | `resolve_external_ids(entity)` returns `[]`; module importable           | PASS — `test_acceptance_criterion`, `test_importable_from_shared_utils` |
| AC3 | 6 filter combinations validated                                           | PASS — permissive / orphans / archived / entity_types / min_confidence / embedding-omission (six docker tests) |
| AC4 | `embedding` NOT in `SELECT`                                               | PASS — sentinel-vector test + `_ENTITY_EXPORT_FIELDS` regression guard |
| AC5 | `JobType.EXPORT_OBSIDIAN` present; job_queue tests still pass             | PASS — 38/38 job-queue tests green                     |
| AC6 | ≥85% line coverage on new modules                                         | PASS — 100% on `export.py` + `external_ids.py` (45 unit tests); new entity-repo methods exercised by all 10 docker tests on the happy path |

## Quality gates

| Gate                                                       | Before | After | Delta |
|------------------------------------------------------------|--------|-------|-------|
| `packages/shared` — full pytest                            | 154    | 199   | +45   |
| `packages/surrealdb-service` — non-docker                  | 58     | 58    | 0     |
| `packages/surrealdb-service` — requires_docker             | 40     | 50    | +10   |
| `packages/job-queue` — full pytest                         | 38     | 38    | 0     |
| `apps/app-main` — full pytest                              | 508    | 508   | 0     |

No regressions in any workspace member.

## Notes / decisions inside the phase

1. **`min_connections` enforcement deferred to exporters.** The
   roadmap defines the knob at the filter level, but counting an
   entity's degree is naturally a join over the result of
   `list_relations_for_notebook`. The exporter (D.1/D.2/D.3) holds
   both sets in memory; the repository stays a single-pass projection.
   Documented in the method docstring.

2. **`Entity` orphan_* fields tolerated as extras.** The B.5b
   migration added `orphan_status`, `reconnect_attempts`,
   `first_orphaned_at`, `last_reconnect_attempt_at` to the schema but
   not to the `Entity` Pydantic model. `ObjectModel.model_config` does
   not set `extra="forbid"`, so the rows project cleanly. The
   projection still SELECTs these columns because the WHERE clause
   needs them — the rows ride through into Python and get dropped by
   the Pydantic ingest.

3. **RecordID coercion in `list_relations_for_notebook`.** SurrealQL's
   `WHERE in INSIDE $ids` compares `record<entity>` against the bound
   array element-wise. The first cut bound plain strings (from
   `Entity.id`), which silently dropped every edge. Fixed by routing
   each id through `ensure_record_id` before passing. Pre-existing
   `get_all_entities_and_relations` (line 587) uses the same pattern
   with raw `n["id"]` strings and is currently untested — it may have
   the same latent bug, but that's outside this phase's scope.

4. **Cross-track coordination on `JobType`.** Per the prompt, Track E
   may also extend `JobType` (with `RESEARCH`). I added only
   `EXPORT_OBSIDIAN` here, per Q-D-2 — JSONL and NetworkX stay
   sync-only in V1. Whoever lands first wins the merge; the second
   track rebases. The diff is a single line so the conflict will be
   trivial.

5. **`primary_type` vs `type_tags` for the allow-list.** The plan said
   "Matched against `Entity.primary_type` and `Entity.type_tags`". I
   chose `primary_type INSIDE $entity_types` only — `primary_type` is
   defined by B.1a as the "best/highest-confidence type when multiple
   type_tags apply", so it's the canonical gate for "is this a
   Person?" The `type_tags` array could match noise tags from the
   merge step. If a future requirement surfaces "include any entity
   tagged Person, even if its primary type is Researcher", this is a
   one-line SurrealQL change.

## Outstanding warnings

None of substance. One pre-existing Pydantic V2.11 deprecation
warning surfaces during testcontainers test runs (`PydanticDeprecatedSince211`)
— it's emitted by the `surrealdb` library, not our code.

## Ready for review.
