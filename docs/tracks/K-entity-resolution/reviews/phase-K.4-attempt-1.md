# Review — Track K Phase K.4 attempt 1

**Branch**: `track/k4-vocabulary`
**Decision**: REVISIONS_NEEDED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-22

## Summary

The precision guard (the central Track-K over-merge backstop) and the fail-soft
network discipline — the two things that matter most — are both implemented
correctly and well-tested. However, the headline feature value (exporters carry
reconciled `external_ids`) is unreachable end-to-end: the Track-D export
projection `_ENTITY_EXPORT_FIELDS` omits `external_ids`/`aliases`, so every
reconciled URI is silently dropped on the real export path. One MAJOR + ruff
style nits block; everything else passes.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | TOOI load → lookup_by_name(BZK) returns tooi ref w/ non-null external_uri | ✅ | Covered by docker roundtrip + unit (tooi_provider.py:179-217) |
| 2 | reconcile_entity(BZK) populates external_ids + aliases | ✅ (in-memory) | Reconciler correct; but see MAJOR-1 — DB→export read path drops it |
| 3 | Two equal candidates → no auto-link, candidate logged (precision) | ✅ | vocabulary_reconciler.py:98-114; test asserts linked=False + nothing written |
| 4 | Crossref lookup for scholarly_article returns DOI URI (mocked) | ✅ | crossref_provider.py:78-105; test_crossref_provider.py |
| 5 | resolve_external_ids(reconciled)→URIs; empty-name→[]; Track D green | ⚠️ | Function correct + Track D tests green, BUT the reconciled-entity case is only unit-tested in-memory; real export path returns [] (MAJOR-1) |
| 6 | refresh idempotent — no duplicate reference_entity rows | ✅ | upsert keyed on (canonical_name, source_vocabulary) UNIQUE idx (migration 41:20); docker test asserts len==1 after reload |
| 7 | Migration 55 idempotent; existing entities read external_ids=[] | ✅ | 55.surrealql all IF NOT EXISTS, additive, B.8 fields untouched; 55_down REMOVE FIELD IF EXISTS |

## Test status

```
test_vocabulary_reconciler.py ............ 7 passed
test_tooi_provider.py + test_crossref_provider.py ... 17 passed
test_external_ids_stub.py + test_entity_repository_roundtrip.py ... 17 passed
test_reference_entity_repository.py (requires_docker) ... 5 passed
test_entity_persistence_service.py ... 10 passed
test_obsidian_export_service.py + test_export_preview.py + test_exports_router.py ... 45 passed
create_app() OK; /api/vocabulary/refresh + /status registered
```
All targeted tests green. No live network in any committed test (all httpx.MockTransport).

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

1. **Reconciled `external_ids` never reach the exporter — export projection drops the field** — `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py:1395-1405`
   - Issue: `_ENTITY_EXPORT_FIELDS` (the SELECT projection used by `list_entities_for_notebook`, entity.py:1508) lists the entity fields but omits both `external_ids` and `aliases`. The Obsidian exporter loads entities exclusively via `list_entities_for_notebook` (`obsidian_export_service.py:843`) and then calls `resolve_external_ids(entity)` (`obsidian_export_service.py:1049`). Because the projection never SELECTs `external_ids`, the rehydrated `Entity` defaults it to `[]`, so a reconciled entity's URI is silently dropped and the YAML always emits `external_ids: []`. The whole stated purpose of K.4 ("so exported Obsidian/JSONL/NetworkX artifacts can carry them" — `external_ids.py:6-7`) is therefore unreachable through the real path.
   - Why AC5 "passes" anyway: `test_external_ids_stub.py::TestReconciledLookups` sets `external_ids` directly on an in-memory `Entity` (test lines 105-111), so it exercises `resolve_external_ids` in isolation but never the DB→projection→exporter path. The narrow AC5 wording is met; the feature value is not.
   - Recommendation: state the issue — the export projection must carry `external_ids` (and `aliases`, similarly omitted) end-to-end, with a test that loads a reconciled entity through `list_entities_for_notebook` and asserts the URI survives into the exporter output.

### 🔵 Minor (optional)

1. **Ruff I001 import-sort on 5 K.4 test files** — `apps/app-main/tests/test_vocabulary_reconciler.py:11`, `packages/shared/tests/test_crossref_provider.py:9`, `packages/shared/tests/test_external_ids_stub.py:15`, `packages/shared/tests/test_tooi_provider.py:9`, `packages/surrealdb-service/tests/test_reference_entity_repository.py:14`. The `style(resolution): ruff import-ordering` commit fixed the src modules but not the tests. All auto-fixable (`ruff check --fix`).

2. **Doc/code mismatch in `ReferenceEntityRepository.upsert`** — `reference_entity.py:11-12` vs `:70-71`. The docstring claims the merge is "done Python-side (pre-fetch + UPDATE) ... because `object::merge` is unavailable", but the code actually issues SurrealQL `UPDATE $id MERGE $data`. Functionally fine (tests pass against live SurrealDB), but the rationale comment is inaccurate and could mislead a future maintainer who copies the pattern.

3. **Crossref confidence is raw relevance score / 100, no title-match check** — `crossref_provider.py:116-117`. The docstring (line 84) says confidence considers "whether the returned title matches the query closely", but `_item_to_match` derives confidence purely from Crossref's `score/100`. Crossref relevance is not a probability and is query-length dependent; a single ≥0.85 hit on a short/generic title could auto-link the wrong DOI. Bounded by the single-candidate guard + 0.85 threshold + the scholarly-type gate, so not a blocker, but the docstring overstates the safeguard. Consider either implementing the title-similarity gate or correcting the docstring.

## Decision rationale

The two highest-priority concerns are sound:

- **Precision guard (AC3)**: `reconcile_entity` collects qualifying matches (≥0.85, a conservative threshold), collapses by `external_uri` (`_distinct_by_uri`, entity-reconciler:157-165), and auto-links only when exactly one distinct URI remains; `>1` returns early with `linked=False` BEFORE any write (`vocabulary_reconciler.py:102-114`, write at :116-117 is unreachable in the ambiguous branch). The same-URI collapse cannot manufacture false agreement: TOOI lookups are exact-equality repo reads (no fuzzy `~`/CONTAINS — `reference_entity.py:115,130`), confidence is fixed (0.99 exact / 0.9 alias), and the `(canonical_name, source_vocabulary)` UNIQUE index forbids two rows sharing a URI per source. `test_two_equal_candidates_do_not_auto_link` genuinely seeds two distinct URIs and asserts `external_ids == []` and `aliases == []`. No fuzzy-match over-link vector found on the TOOI side.
- **Fail-soft network**: `VocabularyHTTPClient.get_json` wraps the entire request in `except Exception → return None` (`http_client.py:113-121`); no path propagates. Timeout (10s), rate-limit (min_interval), cache (TTL), and Crossref polite-pool `mailto:` User-Agent are all present and tested (ConnectError + 503 → `[]`). Provider lookups and the reconciler's `_collect_candidates` add belt-and-braces `except` layers. Ingest cannot break on an unreachable authority.

Migration 55 is clean and additive (B.8 fields untouched). Idempotency (AC6) is enforced by the migration-41 UNIQUE index and verified against a live container. The TOOI seed spot-check (mnre1034=BZK, mnre1010=AZ, mnre1013=BZ, mnre1025=VWS, mnre1109=OCW) matches the real TOOI register and the documented URI pattern; no fabrication. K-D2 is documented as non-blocking with a verified seed + operator bulk-file fallback.

The single MAJOR (export projection drops `external_ids`) forces REVISIONS_NEEDED: K.4's deliverable is that reconciled identifiers flow to exports, and that path is currently inert despite a correct `resolve_external_ids`. This is a small, well-scoped fix.

## Next steps

Implementer should address MAJOR-1 (add `external_ids` + `aliases` to `_ENTITY_EXPORT_FIELDS` and add a load-through-projection test) and the ruff nits (Minor-1, trivial `--fix`), then re-submit. Minor-2/3 are doc accuracy and may be filed as follow-up.
