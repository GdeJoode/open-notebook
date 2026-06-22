# Track L — Entity typing fidelity — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |
| L.1 | Ontology→canonical bridge + preserve rich type | — | `track/l1-canonical-bridge` | 2026-06-22 | implemented — ready for review |

## Phase L.1 — implemented (2026-06-22)

**Branch**: `track/l1-canonical-bridge` (off main). Commits `5b17d4f` → `a0af91b`.

**Mechanism delivered**:
- `packages/ontology-manager/src/ontology_manager/canonical_bridge.py` — `resolve_ontology_type(label, schemas)`:
  - finds the `EntityTypeDefinition` (name/aliases, case-insensitive) across the applied schemas;
  - **prefers `schema_org_type`** (strips a `schema:` prefix → looks up `_CANONICAL_BY_SCHEMA_ORG`);
  - else **walks `parent_type`** recursively, terminating on the schema.org base NAME (works even when the base isn't a loaded definition — `AdministrativeArea`/`GovernmentOrganization` live in `policy.yaml`; `Deal`/`GovernmentService` are parent-only);
  - cycle (visited-set) + depth (max 16) guarded;
  - returns `CanonicalResolution(canonical, ontology_type, type_tags)` or `None`.
  - `_CANONICAL_BY_SCHEMA_ORG` keys are schema.org English identifiers ONLY (no Dutch). `Deal`/`GovernmentService`→`creative_work` is a documented INTERIM until L.3 adds `programme`.
- `entity_persistence_service._resolve_entity_type(raw_label, schemas)`: bridge first → stamp `entity_type`(canonical) + `primary_type`(rich) + `type_tags`; else fall through to the existing `_normalize_entity_type` alias/enum path. `raw_entity_type` still preserved.
- `persist_filtered_result` gains `applicable_schemas`; threaded from `entity_extraction_service` (stashed on the run from `detect_applicable_schemas`, no re-detection). Re-filter path passes `None` (degrades to alias, AC7).

**Per-type ACs verified against the REAL loaded ontologies** (not just fixtures):
- `Gemeente` → `administrative_area`, primary_type `Gemeente`, tags `[Gemeente, AdministrativeArea]` (AC1)
- `Ministerie` → `government_organization` (AC2)
- `RegioDeal` → `creative_work` (interim), tags include `Deal` (AC3)
- `Wet` → `legislation` (AC4)
- `BeleidsThema` → `topic` via `Concept` (AC5)
- `Quux` (not in ontology) → `None` → alias path → no crash (AC7)
- AC6 no-Dutch grep-guard + map-keys-are-schema.org guards green.

**B.8 contract**: every emitted `entity_type ∈ _ALLOWED_ENTITY_TYPES`; hash_id/upsert derive-rule unchanged for a given `(name, type)`. (A `Gemeente` that used to land as `other` now lands as `administrative_area` → different hash_id, but that is the point and applies only to new ingests; existing rows untouched until L.5.)

**Tests / validation (all green)**:
- `packages/ontology-manager/tests/test_canonical_bridge.py` (22) + extended `test_entity_persistence_service.py` (24).
- Full suite: `ontology-manager/tests` + `test_entity_persistence_service` + `test_entity_extraction_service` + `test_notebook_merge_service` + `test_entity_repository_roundtrip` → 269+29 passed, 1 skipped, 0 fail.
- K.7a relation cross-type regression: green (`type_by_name` now uses the same bridge-aware resolver).
- `from app_main.api.app import create_app` → OK. Ruff clean on changed files (2 pre-existing `F821 ExtractionResult` forward-ref warnings unrelated to L.1, untouched).

**Not done here (per plan)**: programme/technology canonical types + migration (L.3); Dutch residual aliases (L.2). The residual alias path is unchanged.

## Basis
- Analysis: `claudedocs/entity-typing-analysis.md` — 27% of entities are `other`, 44% generic concept/topic, because persistence flattens the rich ontology types onto a 20-type enum with no bridge + an English-only alias map (Dutch corpus). type_tags/primary_type empty (rich type lost).
- KEY mechanism: ontologies already declare `parent_type` chains to schema_core schema.org bases → a small fixed schema.org→canonical-enum map (language-agnostic) bridges them. Preserve the ontology type in primary_type/type_tags.
- LANGUAGE: architecture is language-agnostic (typing flows through ontology-declared types); residual free-form-label cleanup stays curated EN+NL per user (98% need). No LLM/embedding fallback for the 2%.
