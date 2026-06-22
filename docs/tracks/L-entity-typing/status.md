# Track L — Entity typing fidelity — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |
| L.1 | Ontology→canonical bridge + preserve rich type | — | `track/l1-canonical-bridge` | 2026-06-22 | implemented — ready for review |
| L.2 | Curated EN+NL residual alias map + non-silent unknown-label fallback | — | `track/l2-nl-aliases` | 2026-06-22 | implemented — ready for review |

## Phase L.2 — implemented (2026-06-22)

**Branch**: `track/l2-nl-aliases` (off main, L.1 merged). Commits `424cdb7` → `bfe57bc`.

**Mechanism delivered**:
- Extracted the alias map to `packages/shared/src/shared/utils/entity_type_aliases.py` — the **single** language-specific module (clean target for the AC5 isolation guard). Owns `ENTITY_TYPE_ALIASES`, `NON_TYPE_LABELS`, and `resolve_residual_type(raw_label) -> ResidualResolution`.
- **EN+NL aliases added** (NL set): `persoon→person, organisatie/organisation→organization, ministerie/uitvoeringsorganisatie→government_organization, gemeente/provincie/regio→administrative_area, wethouder/bestuurder/ambtenaar→person, thema/beleidsthema/sector→topic, wet→legislation, document/publicatie/publication→creative_work`. (English NER short-forms kept: `org/per/loc/gpe/law/...`.)
- **technology/programme decision = option (a)**: aliases point at the REAL targets — `technologie/technology→technology`, `programma/regiodeal/deal/project→programme`. These are NOT yet in `_ALLOWED_ENTITY_TYPES` (that is L.3), so the persistence enum-guard re-pins them to `other` **until L.3** — but the rich label is preserved in `primary_type`, so nothing is lost and **L.3 is a pure enum addition (no alias edit)**. Rejected option (b) (interim `technologie→concept`/`programma→creative_work`) to avoid a churned map + a misleading interim coarse type.
- **Non-silent fallback** (replaces the silent `other`): an unmapped label coarses to `concept` (sensible coarse default) BUT **always preserves the raw label** in `primary_type` + `type_tags`. Empty/whitespace is the only case with no `primary_type` (no signal). Extraction noise (`NON_TYPE_LABELS = {abbreviation, amount}`) is preserved AND flagged (`is_noise` → `properties.non_type_label=True`), kept distinct from genuine-unknown (`Frobnicator`: preserved, not flagged).
- `entity_persistence_service._resolve_entity_type`: order is bridge (L.1) → residual alias (L.2) → non-silent fallback (L.2); the result `entity_type` is enum-guarded at the boundary (re-pins technology/programme to `other` pre-L.3) while `primary_type`/`type_tags` carry the preserved label. `_normalize_entity_type` (coarse, enum-guarded — used by the K.7a relation type-filter) now reads the shared map.

**ACs verified**:
1. NL alias path (no ontology): `Persoon→person`, `Organisatie→organization`, `Ministerie→government_organization`, `Gemeente→administrative_area` ✅.
2. `Technologie→technology` alias EXISTS (re-pinned to `other` by the guard until L.3) + `primary_type=="Technologie"` ✅.
3. `ABBREVIATION` → not silent `other`: `primary_type` retained + `is_noise` flagged ✅.
4. `Frobnicator` → coarse `concept` BUT `primary_type=="Frobnicator"` ✅.
5. Language-surface isolation: grep guard asserts Dutch literals live ONLY in `entity_type_aliases.py`, never in `canonical_bridge.py` (both halves — absence in bridge + presence here) ✅.
6. **Measured `other`-recovery**: all **11/11** live `other`-bucket labels (`RegioDeal, Gemeente, Persoon, Organisatie, Ministerie, Programma, Sector, Wethouder, Thema, Regio, Project`) are recovered by the alias path. 8 land on a real canonical immediately (Gemeente/Persoon/Organisatie/Ministerie/Sector/Wethouder/Thema/Regio); 3 (`RegioDeal/Programma/Project→programme`) re-pin to `other` at the enum-guard **until L.3**, then light up with no further change ✅.
7. B.8 contract: every persisted `entity_type ∈ _ALLOWED_ENTITY_TYPES` (the guard re-pins unknown/pending). hash_id rule unchanged ✅.

**Tests / validation (all green)**:
- `packages/shared/tests/test_entity_type_aliases.py` (35) — new.
- `apps/app-main/tests/test_entity_persistence_service.py` (24) — 4 L.1 tests updated for the L.2 anti-flattening change (aliased/unknown labels now preserve `primary_type`; was silent `other`/`None`).
- Combined suite (`test_entity_type_aliases` + `test_entity_persistence_service` + `packages/ontology-manager/tests`): **272 passed, 1 skipped, 0 fail**. `test_resolution_metrics` (48) green.
- `from app_main.api.app import create_app` → OK. Ruff clean on the 3 changed files.

**Not done here (per plan)**: `programme`/`technology` are NOT added to the ENUM (that is L.3 — only the aliases land now; the runtime guard handles them until then).


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

| L.1 | ontology parent_type→canonical bridge + preserve rich type | — | `track/l1-canonical-bridge` | 2026-06-22 | adversarial-reviewer APPROVED (0 blockers/majors). Language-agnostic bridge (schema.org-base map; no Dutch, grep-guarded); primary_type/type_tags preserved; threaded schemas (no cross-source leak); B.8 hash_id contract byte-unchanged; canonical always in enum (+runtime guard). 60 tests. **L.6 note**: YAMLs declare `schema_org:` (URL) not the model `schema_org_type` → preference path latent, parent_type walk carries all; reconcile in L.6 (`canonical:` override). |

| L.2 | curated EN+NL residual alias map + non-silent fallback | — | `track/l2-nl-aliases` | 2026-06-22 | adversarial-reviewer APPROVED (0 blockers/majors). Dutch aliases in one isolated module (grep-guarded); unmapped labels preserve raw in primary_type (not silent other); noise flagged non_type_label; 11/11 other-recovery (3 pending L.3 enum). technology/programme = option-a (alias→real target, guard re-pins until L.3). B.8 intact, 64 tests. |
