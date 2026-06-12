# Changelog

All notable changes to this project are documented in this file.

## [Unreleased] — Track B — KG Quality

### Migrations

- `44`: `entity.type_tags` + `entity.primary_type` — multi-schema type-tagging (FLEXIBLE arrays).
- `45`: `notebook_schema` + `pass1_results` tables — per-notebook ontology-evolution state and first-pass schema-validation log.
- `46`: `notebook_event` table + `notebook_schema.excluded_types` — shared append-only domain-event log (consumed by soft-nudge banner and future Track G5 webhooks); soft-delete list of hidden schema types.
- `47`: `metrics` table — always-on extraction telemetry with composite `(event_type, created_at)` index and FLEXIBLE payload.
- `48`: `entity.orphan_status` / `entity.reconnect_attempts` / `entity.first_orphaned_at` / `entity.last_reconnect_attempt_at` — prune-lifecycle metadata for orphan retries.

### Features

- Multi-schema two-pass entity extraction (B.1a–B.1f) — Pass-1 schema validation + Pass-2 typed extraction with per-entity / per-relation confidence; multi-schema orchestrator runs top-3 ontologies sequentially with cumulative soft-nudge and in-process merge.
- TTL / RDFS schema export with Protégé compatibility (B.2a–B.2b) — fixed module-load `NameError` in `rdf_owl_shacl.py`; new `GET /api/notebooks/{id}/schema.ttl` endpoint that merges accepted extensions as `owl:Class` declarations.
- Schema editing UI (B.3a–B.3d) — Schema-tab view with per-source coverage stats; six edit operations (accept-extension, reject-extension, rename, merge, split, delete); soft-nudge banner with per-notebook pause-toggle; schema-change → re-extract prompt.
- Always-on confidence telemetry (B.4) — `ConfidenceBar` + `ConfidenceFilter` UI components; persistent metrics for `extraction.complete` and `extraction.auto_fallback` events; closes Track A RETRO #5 (telemetry blind spot).
- Orphan-connector + prune-lifecycle (B.5a–B.5b) — co-occurrence + LLM-confirm reconnection of pending-orphan entities; status lifecycle (`pending_reconnect` → `archived`) with N-attempts and age-based thresholds; archived entities are recoverable, not hard-deleted.
- Cross-notebook graph merge (B.6) — `POST /api/notebooks/merge` with semantic-content idempotency; 422 guards for self-merge, empty source list, and archived source/target notebooks.

### Migration notes for operators

- Run `surreal sql ... < migrations/44.surrealql` through `migrations/48.surrealql` in order. All migrations are idempotent (`IF NOT EXISTS` throughout); re-runs are safe.
- Set the environment variable `OPEN_NOTEBOOK_DISABLE_METRICS=1` to opt out of the always-on telemetry writes added in B.4. The `metrics` table will be created either way (it's part of migration 47); the env-toggle only suppresses runtime `INSERT` calls.
- Multi-schema mode is **enabled by default** when extraction runs with a `notebook_id`. Pass `multi_schema_enabled=false` per-request (router → handler → service) to fall back to single-schema behaviour for that run.
- The orphan prune-lifecycle archive rule defaults to `max_attempts=3` and `max_age_days=90` (kwarg names match `pipelines/entity-filtering/.../orphan_prune.py::archive_stale_orphans`). Archived entities are kept (recoverable); the UI hides them from the active dashboard but operators can still query history via `SELECT * FROM entity WHERE orphan_status = "archived"`.
- The TTL endpoint (`GET /api/notebooks/{id}/schema.ttl`) returns `Content-Type: text/turtle` with `Content-Disposition: attachment; filename="<notebook>.ttl"`. Output is well-formed against rdflib 7+ and roundtrips through pyshacl.
