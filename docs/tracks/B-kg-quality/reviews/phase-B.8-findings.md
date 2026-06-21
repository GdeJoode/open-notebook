# Track B.8 — KG live-validation: consolidated findings (2026-06-21)

B.8 was a follow-up to the COMPLETE Track B, triggered by a live test where
ingested documents showed **no entities / no KG**. What looked like one bug was
a chain of seven, plus a systemic schema-drift root cause. All are fixed +
adversarial-reviewed; qwen2.5:14b extraction now produces and persists entities
end-to-end (verified: bc6xa 421, four Regio Deal Convenant docs 200/333/250/274).

## The chain (why "no entities")
1. **ORDER BY name** crashed the KG query — `entity.name` is covered by a
   full-text SEARCH index; SurrealDB v2 can't ORDER BY from it. `list_entities`
   swallowed the error → empty UI. Fix: `WITH NOINDEX`. (Shipped + deployed.)
2. **No-schema fallback ran a caller-less extractor** — when a notebook has no
   schema, `_run_multi_schema` fell back to a bare `ExtractionWorkflow` whose
   single-mode path makes a caller-less `LLMExtractor` → 0 entities. Fix: shared
   `_build_single_schema_workflow` wires the caller in both paths.
3. **JSON parser strict-mode** dropped qwen's control-char batches →
   `json.loads(strict=False)`.
4. **entity_type enum violation** — qwen emits "Location"/"ABBREVIATION"; the
   live enum rejected every upsert. Fix: `_normalize_entity_type` + alias map.
5. **Invalid RELATE SurrealQL** (`$src[0].id->relation->...`) → bind ids to LET vars.
6. **`name`/`hash_id` required by the live DB but unset** → upsert sets `hash_id`
   (md5 of the exact case-sensitive dedup identity) + dual-writes `name=canonical_name`.
7. **Systemic root cause — pre-Track-B schema drift (Q-B-1)**: the live `entity`
   table still enforced the OLD schema (`name`/`hash_id` required, `entity_type`
   enum, `idx_entity_fulltext`) that migration 39 never *removed* (DEFINE TABLE
   SCHEMAFULL doesn't drop pre-existing field defs). Track B code — tested only
   against fresh testcontainers — passed CI but couldn't write to live. Fix:
   **migration 50** relaxes the legacy constraints (reconciles the drift in
   version control); 147 legacy entities backed up (`claudedocs/entity_backup_pre_mig50.json`).

## Model + provenance (B.8a / B.8a-2)
- Extraction model is now **independently configurable** from chat
  (`DefaultModels.default_extraction_model`; chain via `resolve_default_model_id`).
  Live: chat=llama3.1, extraction=qwen2.5:14b.
- `extraction_method`/model provenance now recorded; fully-failed entity batches
  raise instead of reporting silent success.

## Resolution assessment (B.8c)
897 distinct canonical entities across the 4 related docs; **107 span ≥2 docs**.
Programme-level entities (Regio Deal, Regio Envelop, ministerie van BZK, the
signing minister, key persons) resolve correctly. **Verdict: PARTIALLY MET** —
the V1 normalizer handles consistent surface forms but fragments variants
(BZK 8-way, "minister" 23-way, spelling variants). The documented Q9/M4 ceiling.
See `phase-B.8c-resolution-assessment.md`.

## Open follow-ups (tracked, not done in B.8)
- **F1 — Resolution quick-wins** (cheap, before full M4): strip leading
  articles/role-prefixes ("De ", "Minister(ie) van ") from the match key;
  spelling-variant tolerance (Koninkrij(k|ks)); govt-org abbreviation aliases
  (BZK ↔ Binnenlandse Zaken). Would collapse much of the observed fragmentation.
  Belongs in the `name_normalizer` swap-point (Q9/M4), kept out of B.8c (assess-only).
- **F2 — Full resolution = Track M4/Q9** (TOOI + Crossref + NL normalization + fuzzy).
- **F3 — idx_entity_fulltext is still live-only drift** (not in any migration).
  Recommend: add a migration defining it (so fresh + live match) and keep the
  `WITH NOINDEX` read-path; or document a deliberate drop. Decision pending.
- **F4 — Legacy 144 entities** (pre-Track-B, name-only, violate avg_confidence/etc.):
  recommend a purge (they're Dec-2025 spaCy noise) or a one-off normalize; left as-is for now.
- **F5 — Relation-write failures still swallow** (warning only) — surface like entities.
- **F6 — `_save_result` large-payload root cause**: B.8d made it non-fatal, but the
  surrealdb-ws KeyError stems from serializing hundreds of entities into one
  `extraction_result` record. Consider storing only counts+metadata (the entities
  already live in the entity table) or chunking the write.
