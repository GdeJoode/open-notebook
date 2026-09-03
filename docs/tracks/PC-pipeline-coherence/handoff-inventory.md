# Handoff inventory — derived state and its readers

Track PC, phase PC.1b. One row per piece of derived state the extraction chain
produces. "Derived state" means what a step **measured, judged or attempted** — not
the payload (entities, relations, chunks), which survives everywhere.

**How to read a row.** `consumer` counts readers in production code, excluding
tests and excluding the model that declares the field. `NONE` means the value is
computed, often persisted, and never read back by anything in this repository.

**The cut rule**, applied in order:

1. Does a human or a downstream stage make a decision **today** that is wrong or
   absent because this is dropped? → **wire**
2. Is something else already doing this job better? → **delete**
3. Otherwise → **accept, with an owner phase and a line in that phase's AC**

Compiled 2026-09-03 by tracing `run_pass2` → `_merge_results` → `run_multi_schema`
→ `ExtractionWorkflow.extract` → `EntityExtractionService.run_extraction` →
`FilteringWorkflow.process` → `EntityPersistenceService`, and by counting readers
with a repo-wide grep. **That grep is how this file was first compiled, not how
the rule is enforced**: three review rounds established that counting `.field` by
name cannot be made honest, so `tests/test_derived_state_has_readers.py` requires
each field to DECLARE its consumer and verifies the declaration instead.

---

## Why this file exists rather than a run object

`FilteredResult` (`packages/shared/src/shared/models/extraction.py`) is
**already a typed run-state carrier**. Measured with the guard's own counter,
counting production FILES that Load each attribute:

* every **payload** field is read somewhere — `metadata` in 21 files, `entities`
  in 9, `relations` in 8, and `removed_entities`, `merged_entity_groups`,
  `match_candidates` and `predicted_edges` in 1 each;
* of the **derived-state** fields, only `concept_alignment_report` is read at all
  — in 1 file, which drops 5 of its 11 keys at the boundary.

Three review rounds went into that paragraph and each of the first two appended a
correction instead of replacing what it corrected, so the document kept recording
its own history at the cost of still stating the error. The errors were: presenting
grep OCCURRENCE counts as readers, and then "every derived-state field is read in
none", which contradicted the table below it. This is the replacement.

The original occurrence counts, kept because they are what the first draft cited:

| field | kind | occurrences (NOT files) |
|---|---|---|
| `merged_entity_groups` | payload | 4 |
| `predicted_edges` | payload | 4 |
| `match_candidates` | payload | 2 |
| `removed_entities` | payload | 2 |
| `kg_resolution_report` | derived | **0** |
| `validation_report` | derived | **0** |
| `linked_entities` | derived | **0** |
| `llm_verification_results` | derived | **0** |
| `concept_alignment_report` | derived | 1, dropping 5 of its 11 keys |

Same object, same typing. The payload is read; the derived state is not. The repo
has run both experiments already — `ExtractionResult.metadata` is the untyped bag
where state dies silently, and these typed report fields are the version where it
dies just as reliably. So the fix is not a third carrier. It is the invariant in
`tests/test_derived_state_has_readers.py`: a producer names its consumer or goes.

**Trigger for revisiting**: build an explicit run object when this table shows
**three or more boundaries needing the same field**. Today none does.

---

## WIRE — a reader is already waiting

| # | Boundary | Dropped | Producer | Waiting consumer |
|---|---|---|---|---|
| W1 | `run_multi_schema` → `ExtractionWorkflow.extract` | the `SoftNudgeDecision` | `multi_schema_orchestrator.py:667,716` | `notebook_event` table + router `notebook_events.py:68` + `SchemaSoftNudge.tsx` — **0 rows ever written** |
| W2 | `_merge_results` → persist | merged `type_tags` / `primary_type` | `multi_schema_orchestrator.py:796-798` | `entity` rows; `upsert_entity` already unions `type_tags` server-side (`entity.py:204-207`) |
| W3b | re-filter → the stored row | a prior run's `concept_alignment` sub-dict | `entity_extraction_service.py:2166-2184` overwrites | the row itself; the erase is silent |

**W3 was filed here and does not belong here.** A review pointed out that its
"waiting consumer" was given as *PC.3's own AC* — and a future phase's acceptance
criterion is not a waiting reader. By this file's own cut rule that is an ACCEPT
row, and it has moved to that table. The code stays (`summary["persisted"]` and
`extraction_result.metadata["persisted"]`), because PC.3 needs the instrument and
building it twice would be churn; what changes is the honesty of the label. This
is the phase's own rule applied to the phase's own work.

**W1 is the sharpest case in the repo**: producer, table, repository, router and
React component all exist; only the write between them was missing.

An earlier version of this file said the verdict was discarded at
`workflow.py:157` (`merged, _decision = await run_multi_schema(...)`). A review
showed that is wrong, and the correction strengthens the point: `run_multi_schema`
already writes the verdict into `merged.metadata["soft_nudge"]`, and
`_emit_soft_nudge` reads exactly that key rather than the discarded return value.
The state was **carried faithfully and nothing acted on it** — which is this
file's thesis, not an exception to it. Measured on the five sources holding Pass-1 rows, all
five would have fired a banner (coverage 0.45 / 0.48 / 0.55 / 0.58 →
`schema_mismatch`, 0.85 → `extension_suggested`).

## DELETE — superseded or unused

| # | What | Status | Why |
|---|---|---|---|
| D1 | `extraction_chunking/extraction_metrics.py` + tests | **done (PC.1b)** | Superseded by `shared/regression/extraction_gate.summarise_run`, which computes the same two rates over the same keys **with** N.5d's per-metric input discipline. `measure_extraction` had no production caller. |
| D2 | `frontend/src/lib/api/sources.ts` `getExtractionResult` | **done (PC.1b)** | Defined; no component called it. |
| D3 | `FilteredResult.linked_entities`, `.llm_verification_results` | **done (PC.1b)** | Zero readers. `linked_entities` was a second copy of URIs that already travel in each entity's `properties`; `llm_verification_results` was never written at all. |
| — | `kg_resolution_report` | kept, owner **PC.3** | Its AC needs a measured figure for how many rows cross-document resolution collapses. Listed here rather than deleted: removing a measurement one phase before it is wanted is churn. |
| — | `validation_report` | kept, owner **PC.6** | Stage 11 is inert because no production call site passes an ontology to `FilteringWorkflow`; PC.6 owns making "the flag is on and it did nothing" visible. |

## ACCEPT — recorded, with an owner

| Dropped | Boundary | Owner | Why there |
|---|---|---|---|
| `kg_resolution_report` (kept, no reader) | filtering → nothing | **PC.3** | The declaration in `tests/test_derived_state_has_readers.py` names PC.3; this row is the other half of that promise. |
| `validation_report` (kept, no reader) | filtering → nothing | **PC.6** | Same, naming PC.6. A promise made in one place is not a promise, which is why the guard requires both. |
| `persist_filtered_result`'s five counts (**W3**, code shipped) | persist → `run_extraction` | **PC.3** | The counts now reach `summary["persisted"]` and `extraction_result.metadata["persisted"]`, and `scripts/pc1b_handoff_probe.py` reads the durable copy — but no PRODUCTION consumer reads either. Filed honestly as accept-with-owner: PC.3's AC ("materially fewer than 117 rows, with a named figure") is what will read it. |
| `source_chunk_id`, `source_grounding`, `extraction_context` never persisted | filtering → persist | **PC.3** | They have live in-run consumers; only the persist boundary drops them, and they are the evidence a cross-document match needs to be explainable |
| `verdict_counts`, `method_counts`, `reason_counts`, `capped_type_fetches`, `alias_candidates` | alignment report → `filtering_stats` (7 of 11 keys copied) | **PC.2** / **PC.6** | `alias_candidates` is PC.2's containment signal; the counts are how "the flag is on and did nothing" becomes visible |
| `_save_result` stores raw pre-filter entities beside post-filter `metadata["filtering"]` | `run_extraction` → `extraction_result` | **PC.6** | Needs a decision (label both, or stop storing raw), not a patch |
| `ontology_gap` / `schema_proposal` rows unread by any route | evolution agent → nothing | **PC.5** | Already assigned by the plan |
| `metrics` rows `extraction.complete` | `record_metric` → nothing | **PC.6** | Only `routing.served` is read; decide whether extraction metrics are wanted |
| `metadata["best_coverage"]`, `["schemas_attempted"]`, `merged_from_schemas`, `schema_count` | merge → nothing | **PC.6** | Candidates for deletion once PC.6 decides what a run should report |
| `merged_duplicates_collapsed`, `per_schema` | merge → nothing yet | **PC.6** | Produced by N.5a for the gate; the gate reads the summed counters, not these |
| `relation_source` / `relation_sources` | pass 2 + persist → nothing | **PC.2** | One producer (Hearst, default-off), no reader. Either PC.2's identity work reads it or it goes |
| `incremental_report` incl. `repair` | filtering stage 10b → `filtered.metadata` | **PC.3** | Belongs with the cross-document resolution decision |
| `find_by_alias` has no `verified` filter and no `ORDER BY` | `entity_alias` → KG resolver tier 1 | **PC.2** | Two readers disagree about `verified`; one surface form can bind to two canonicals non-deterministically |

## Inverted case — a consumer with no producer

`notebook_event{extension_suggested, schema_mismatch}` is polled by the frontend
through a working router, and **nothing has ever written one**. This is the mirror
image of every row above and it is why the invariant checks both directions is left
as a follow-up: the current test catches producers without consumers, not consumers
without producers. Closing this specific one is W1.

## The residual hole, stated rather than hidden

A dead field named identically to an existing row here — `metrics`,
`alias_candidates` — satisfies the `Owned` row check, because the guard cannot
tell that the row is about the metrics *table* rather than about a new field of
that name. A review found it and judged it the intended reviewability boundary
rather than a hole; the reasoning is worth keeping. The collision set is no longer
"any substring in a document" but the ~15 rows this file contains, each already
naming a phase that owns that subject, so a false positive requires the new field
to be named identically to an existing owned handoff — at which point "that phase
owns it" is close to true anyway. It also requires a visible `Owned(...)` entry in
the diff, which is a different risk class from a value that looks like ordinary
usage. And closing it needs semantic knowledge of what a row is about, which is
the same wall as type inference.

## Found by the PC.1b review, owned elsewhere

- **`soft_nudge_dismissed` is never re-armed.** `schemas.py:1485` sets it `True`
  and a comment claims the B.1e orchestrator sets it back; nothing in the
  repository ever writes `False`. Combined with W1's duplicate suppression, one
  click on "Don't show again" permanently silences the wire this phase just
  built, while events accumulate unread and suppress each other. → **PC.5**, with
  the dismissal path it owns.
- **Suppression hides the worse document behind the better one.** The first
  unread event of a type blocks later ones, and its payload keeps the FIRST
  document's coverage. Latent today — `SchemaSoftNudge.tsx` renders a static
  per-type string and never reads the payload — and live the moment anyone
  surfaces it. → **PC.5**.
- **Deleting the four uncollectable test files cost real coverage**, even though
  their effective coverage was already zero: `clean_thinking_content` and
  `parse_thinking_content` in `shared/utils/text.py` now have no test anywhere.
  `split_text`, `token_count` and `compare_versions` are covered elsewhere. →
  unowned; named so it is a known gap rather than a silent one.
- **Four of this phase's own guards live in package test directories**, which
  `unit-guards.yml` does not run — in the phase whose header names "it runs
  nowhere" as failure mode #1. Widening the job is not free: `pytest tests
  apps/app-main/tests` in one invocation dies with `Plugin already registered
  under a different name` under `--import-mode=importlib`. → **PC.6**, with the
  configuration coherence it owns.

## Bugs found while compiling this

- `KGResolver.report["aliases_registered"]` is initialised and logged, never
  incremented — the INFO line always prints `0`.
- `run_filtering_only` builds a bare `FilteringConfig()` (fuzzy and embedding dedup
  **off**) while the main extraction path enables both. The two paths are not the
  same pipeline.
- `entity_alias` is SCHEMAFULL and declares five fields; `register_alias` writes
  four more, including `verified`. No migration declares them. → **PC.2**
- `entity.status` is a free-form `str` with four values in use, no enum.
