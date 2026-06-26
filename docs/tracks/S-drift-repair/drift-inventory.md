# Track S — Phase S.1: Strict-field drift inventory (staging)

> **Phase S.1, read-only discovery.** Generated 2026-06-26 against the LIVE `staging`
> DB (`ws://localhost:8000`, ns `open_notebook`, db `staging`, root/root). No code or DB
> writes were made. Counts verified as staging: **6 sources, 4870 entities, 1466 relations**.

## Method

- Enumerated all tables via `INFO FOR DB` (78 defined; 31 carry rows after excluding
  `_sbl_migrations`).
- For each table with rows, pulled field defs via `INFO FOR TABLE <t>`.
- Classified each field as **writability-blocking** iff it is strict — i.e. **not**
  `option<...>`, has **no** `VALUE` clause (those self-heal: `created`/`updated` =
  `VALUE $before OR time::now()`), and has **no** NONE-permitting `ASSERT`
  (`$value == NONE OR ...`). `FLEXIBLE` is counted only when the base type is strict
  (FLEXIBLE on a strict `array`/`object` still rejects NONE). Array element subfields
  (`field[*]`) are excluded — they constrain elements, not the row.
- For every candidate blocking field, counted NONE rows with
  `SELECT count() FROM <t> WHERE type::is::none(<field>) GROUP ALL`. (A plain
  `SELECT ... = NONE` hides NONE-valued fields, so `type::is::none()` is required.)
  Probe sanity-checked: `type::is::none(source.zotero_key)` returns 6/6 as expected.

## Headline finding

**Zero writability-blocking drift on staging right now.** Every strict, non-`option<>`,
no-self-heal field across all 31 active tables has a live-NONE-count of **0**. The two
historically-drifted tables — `entity` (migration 61) and `source` (migration 64) — are
already healed, and no other table has accrued NONE on a strict field.

This means the S.2 self-healing migration would be a **no-op coalesce on current staging
data**, but it remains worth shipping as the idempotent forward-guard the plan calls for:
it makes any future-drifted or freshly-restored environment self-heal on migrate, and the
S.2 `@requires_docker` reproduction test still validates the mechanism on synthetic drift.

## Table-by-table (strict fields and live-NONE counts)

Every table below: all listed strict fields have **NONE = 0** on staging. Tables not
listed (`model`, `notebook`, `note`, `summary`, `command`, `chat_session`,
`open_notebook`, `model`) have **no** strict-blocking fields — all their fields are
`option<>` or SCHEMALESS, or the table currently has 0 rows.

| Table | Rows | Strict fields (all NONE=0) | Blocking drift |
|-------|-----:|----------------------------|:--------------:|
| `entity` | 4870 | `entity_type`, `status`, `manual_override`, `avg_confidence`, `weight`, `mention_count`, `max_raptor_level`, `reconnect_attempts`, `sources`, `aliases`, `type_tags`, `provenance_chain`, `source_documents`, `properties`(FLEX), `external_ids`(FLEX), `raptor_level_counts`(FLEX), `created`, `created_at`, `updated`, `updated_at`, `extracted_at` | **none** (healed by mig 61) |
| `source` | 6 | `private` | **none** (healed by mig 64) |
| `relation` | 1466 | `relation_type`, `status`, `confidence`, `extraction_method`, `properties`(FLEX), `provenance_chain`(FLEX), `source_documents`(FLEX), `created_at`, `extracted_at`, `in`/`out`(edge) | none |
| `chunk` | 1448 | `text`, `element_type`, `source`, `order`, `layer`, `is_content`, `is_raptor_node`, `physical_page`, `parent_chunk_ids`, `metadata`(FLEX), `positions`(FLEX) | none |
| `source_embedding` | 1447 | `content`, `embedding`, `order`, `source` | none |
| `chat_session` | 4 | — (only `option<>`) | none |
| `claim` | 13 | `claim_type`, `statement`, `verification_status`, `created`, `updated` | none |
| `command` | 46 | — (SCHEMALESS / option) | none |
| `dead_letter` | 20 | `job_id`, `job_type`, `error_message`, `payload`(FLEX), `retry_count`, `failed_at` | none |
| `derived_from` | 1448 | `in`/`out` (edge) | none |
| `doc_node` | 1597 | `element_type`, `self_ref`, `sequence`, `source`, `created_at` | none |
| `entity_suggestion` | 30 | `name`, `entity_type`, `aliases`, `confidence`, `extraction_method`, `status`, `source_id`, `properties`(FLEX), `created`, `updated` | none |
| `episode_profile` | 3 | `name`, `default_briefing`, `num_segments`, `outline_model`, `outline_provider`, `transcript_model`, `transcript_provider`, `speaker_config`, `created`, `updated` | none |
| `extraction_result` | 4 | `source_id`, `entity_count`, `relation_count`, `entities`(FLEX), `relations`(FLEX), `metadata`(FLEX), `created` | none |
| `job` | 74 | `job_type`, `status`, `priority`, `retry_count`, `max_retries`, `payload`(FLEX) | none |
| `metrics` | 4266 | `event_type`, `payload`(FLEX), `created_at` | none |
| `model_route` | 3 | `task`, `private_chain`(FLEX), `provider_chain`(FLEX), `updated_at` | none |
| `next_node` | 1442 | `in`/`out` (edge) | none |
| `parent_of` | 1417 | `in`/`out` (edge) | none |
| `pass1_results` | 16 | `notebook`, `source`, `detected_schema`, `schema_attempted`, `coverage_pct`, `confidence_in_choice`, `alternative_schemas`(FLEX), `proposed_extensions`(FLEX), `uncovered_concepts`(FLEX), `created_at` | none |
| `preprocessing_result` | 3 | `source_id`, `total_chunks`, `naive_summary`, `filtered_chunk_ids`, `removed_chunk_ids`, `classification`(FLEX), `noise_stats`(FLEX), `structure_stats`(FLEX), `created` | none |
| `reference` | 6 | `in`/`out` (edge) | none |
| `refers_to` | 1 | `in`/`out` (edge) | none |
| `speaker_profile` | 3 | `name`, `speakers`, `tts_model`, `tts_provider`, `created`, `updated` | none |
| `status_change_log` | 1190 | `entity`, `old_status`, `new_status`, `reason`, `batch_id`, `changed_at` | none |
| `transformation` | 11 | `name`, `title`, `description`, `prompt`, `apply_default` | none |
| `triage_queue` | 787 | `entity`, `name`, `type`, `decision`, `reason`, `batch_id`, `doc_count`, `structural_degree`, `created_at`, `updated_at` | none |

`open_notebook` (3), `model` (17), `summary` (1) have rows but **no** strict-blocking
field (all `option<>` / SCHEMALESS).

## Tables that currently have writability-blocking drift on staging

**None.** No table on staging has a NONE value on a strict, non-`option<>`, no-self-heal
field. Migration S.2 has no live rows to repair on the current staging snapshot; it ships
as the idempotent forward-guard (and its reproduction test exercises synthetic drift).

## entity & source — already covered by 61 / 64

- `entity`: migration **61** coalesces `manual_override, avg_confidence, weight,
  mention_count, max_raptor_level, reconnect_attempts, status, created, created_at,
  updated, updated_at, extracted_at, provenance_chain, source_documents, type_tags,
  aliases, embedding`. All verified NONE=0 on staging. S.2 re-coalescing these is an
  idempotent overlap (no-op).
- `source`: migration **64** coalesces `private` (the load-bearing strict field) and
  `topics` (data hygiene, `option<>`). `private` verified NONE=0. S.2 re-coalescing
  `private` is an idempotent overlap (no-op).

## Out-of-band fields (in live `INFO FOR TABLE`, not defined in any `migrations/*.surrealql`)

These exist on the live schema but have no `DEFINE FIELD` in the migration set — they were
likely created out-of-band (app-side `DEFINE`, ad-hoc, or a dropped/renamed migration).
The R.0e review previously flagged the `source` ones.

| Table | Field | Live type | Strict-blocking? |
|-------|-------|-----------|:----------------:|
| `source` | `asset` | `FLEXIBLE option<object>` | no (option) |
| `source` | `cached_scores` | `FLEXIBLE option<object>` | no (option) |
| `source` | `external_ids` | `FLEXIBLE option<object>` | no (option) |
| `source` | `type_metadata` | `FLEXIBLE option<object>` | no (option) |
| `source` | `source_type` | `option<string>` (`ASSERT $value == NONE OR ... INSIDE [...]`) | no (option + NONE-permitting assert) |

All five `source` out-of-band fields are `option<>` and therefore **not** writability
risks. No out-of-band field on any other active table was found that is strict-blocking.
(Edge `in`/`out` and the standard fields are all defined in migrations.)

## "Verify" / classification caveats

- **Edge `in`/`out`** (`derived_from`, `next_node`, `parent_of`, `reference`,
  `refers_to`, `relation`): strict `record<...>` with no DEFAULT. They cannot be NONE on
  a real edge (RELATE always sets both endpoints) and have no meaningful coalesce default,
  so they are **not** drift-repairable and not a risk. Excluded from any S.2 SET list.
- **Strict fields with no DEFAULT** that are populated-at-insert (`chunk.text`,
  `source_embedding.embedding/content`, `doc_node.self_ref`, `entity.entity_type`,
  `relation.relation_type`, etc.): all NONE=0 on staging. They have no safe coalesce
  default, so S.2 should **not** invent one; they are only a theoretical risk if a future
  migration adds such a field *without* backfilling, which is exactly the S.4 prevention
  rule. Flagged as "do not coalesce" rather than "verify".
- **`refers_to.out`** parsed as `record<notebook|source>` (union FK) — confirmed
  non-NONE (1/1 row), edge field, excluded.

## Recommended S.2 coalesce list

Since staging shows zero drift, S.2's purpose is the **idempotent forward-guard**. The
recommended `UPDATE <table> SET f = f ?? <default>` list = every strict field that has a
**safe coalesce default** (an explicit migration DEFAULT, or a type-obvious zero-value).
Fields with no safe default (required FKs, `in`/`out`, content/text bodies) are excluded.

| Table | `SET f = f ?? default` |
|-------|------------------------|
| `entity` | (mirror migration 61 exactly — idempotent overlap) |
| `source` | `private = private ?? false` (mirror 64) |
| `relation` | `status ?? 'active'`, `confidence ?? 1.0`, `extraction_method ?? 'llm'`, `properties ?? {}`, `provenance_chain ?? []`, `source_documents ?? []`, `created_at ?? time::now()`, `extracted_at ?? time::now()` |
| `chunk` | `layer ?? 0`, `is_content ?? true`, `is_raptor_node ?? false`, `parent_chunk_ids ?? []`, `metadata ?? {}`, `positions ?? []` |
| `claim` | `verification_status ?? 'unverified'`, `created ?? time::now()`, `updated ?? time::now()` |
| `dead_letter` | `retry_count ?? 0`, `payload ?? {}`, `failed_at ?? time::now()` |
| `doc_node` | `created_at ?? time::now()` |
| `entity_suggestion` | `aliases ?? []`, `confidence ?? 0.8`, `extraction_method ?? 'openie'`, `status ?? 'pending'`, `properties ?? {}`, `created ?? time::now()`, `updated ?? time::now()` |
| `episode_profile` | `num_segments ?? 5`, `created ?? time::now()`, `updated ?? time::now()` |
| `extraction_result` | `entity_count ?? 0`, `relation_count ?? 0`, `entities ?? []`, `relations ?? []`, `metadata ?? {}`, `created ?? time::now()` |
| `job` | `status ?? 'queued'`, `priority ?? 'normal'`, `retry_count ?? 0`, `max_retries ?? 2`, `payload ?? {}` |
| `metrics` | `payload ?? {}`, `created_at ?? time::now()` |
| `model_route` | `private_chain ?? []`, `provider_chain ?? []`, `updated_at ?? time::now()` |
| `pass1_results` | `coverage_pct ?? 0.0`, `confidence_in_choice ?? 0.0`, `alternative_schemas ?? []`, `proposed_extensions ?? []`, `uncovered_concepts ?? []`, `created_at ?? time::now()` |
| `preprocessing_result` | `created ?? time::now()` |
| `triage_queue` | `decision ?? 'open'`, `doc_count ?? 0`, `structural_degree ?? 0`, `type ?? ''`, `created_at ?? time::now()`, `updated_at ?? time::now()` |
| `transformation` | `apply_default ?? false` |
| `status_change_log` | `changed_at ?? time::now()` |
| `speaker_profile` | `created ?? time::now()`, `updated ?? time::now()` |

Fields like `chunk.text`, `*.source`, `*.source_id`, `*.entity`, `*.name`, edge `in`/`out`
are strict-with-no-default required fields — **excluded** (no honest coalesce value; the
S.4 prevention rule is the real fix should one ever ship un-backfilled).
