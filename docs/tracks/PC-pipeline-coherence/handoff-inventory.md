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
| ~~`source_chunk_id`, `source_grounding`, `extraction_context` never persisted~~ **CLOSED (PC.3)** | filtering → persist | — | All three are first-class fields on `ExtractedEntity` while `persist_filtered_result` read only `entity["properties"]`. Now carried as `properties["grounding"]`, **keyed by source id** — a canonical row is merged across documents while grounding is per mention, so a flat key would mean the last document to mention an entity overwrites where every earlier one found it, and with resolution on that is the normal case. `upsert_entity` merges this one key by source rather than overlaying it. |
| ~~`alias_candidates`~~ **CLOSED (PC.2)** | — | — | Deleted with its producer. Same population as the curator door (`find_by_type` returns graph entities) and a weaker rule — unanchored containment, measured to pair `Regio Deal` with `Regio Deal Groningen`. `_score_containment` carries the signal now, with a card and an apply path. |
| `verdict_counts`, `method_counts`, `reason_counts`, `capped_type_fetches` | alignment report → `filtering_stats` (7 of 11 keys copied) | **PC.6** | How "the flag is on and did nothing" becomes visible |
| `_save_result` stores raw pre-filter entities beside post-filter `metadata["filtering"]` | `run_extraction` → `extraction_result` | **PC.6** | Needs a decision (label both, or stop storing raw), not a patch |
| `ontology_gap` / `schema_proposal` rows unread by any route | evolution agent → nothing | **PC.5** | Already assigned by the plan |
| `metrics` rows `extraction.complete` | `record_metric` → nothing | **PC.6** | Only `routing.served` is read; decide whether extraction metrics are wanted |
| `metadata["best_coverage"]`, `["schemas_attempted"]`, `merged_from_schemas`, `schema_count` | merge → nothing | **PC.6** | Candidates for deletion once PC.6 decides what a run should report |
| `merged_duplicates_collapsed`, `per_schema` | merge → nothing yet | **PC.6** | Produced by N.5a for the gate; the gate reads the summed counters, not these |
| `relation_source` / `relation_sources` | pass 2 + persist → nothing | **PC.7** (was PC.3, was PC.2) | Reassigned: this is RELATION provenance, not entity identity, so it sits with `source_grounding` in the row above — the evidence a cross-document match needs to be explainable. Not deleted, because N.5's R2 fix exists precisely to stop it being lost at the persist collapse. **NOT closed by PC.3** — that phase spent itself on entity identity and never reached relation provenance; recorded here rather than left to look done. |
| `incremental_report` incl. `repair` | filtering stage 10b → `filtered.metadata` | **PC.7** (was PC.3) | NOT closed by PC.3, and reassigned rather than left silent. Stage 10b is off in every shipped configuration, so the report is never produced; giving it a reader before the stage has one would be the same defect PC.3 measured in stage 10. It moves with the decision about which dormant stages have a future. |
| ~~`find_by_alias` has no `verified` filter and no `ORDER BY`~~ **CLOSED (PC.2)** | `entity_alias` → KG resolver tier 1 | — | Now `ORDER BY verified DESC, similarity_score DESC, id ASC`: a human decision outranks a machine one, ranked not filtered so tier 1 does not go inert. Migration 78 was needed first — three of those columns did not exist on a fresh database. |

## Settled by PC.6

| row | disposition |
|---|---|
| `validation_report` — stage 11 inert | **Made unreachable.** The claim was "no production call site passes an ontology". Measured: with `enabled=True` and `ontology=None` the filter does not merely skip — it returns `{"valid_entities": 1, "invalid_entities": 0}`, i.e. a SUCCESS report for a validation that never ran, behind one DEBUG line. Now a BLOCK finding (`validation-without-ontology`). The report field itself stays; it is written when validation actually runs. |
| `metrics` rows nothing reads | **Measured and reassigned to PC.5.** Only `routing.served` is read (`model_routes.py`); `export.jsonl` and `export.obsidian` are written and read nowhere. That is a question about what a run should REPORT to a human, which is PC.5's remit ("a door for the curator loop"), not configuration coherence. Reassigned with the measurement rather than deferred with a shrug. |
| alignment report keys dropped at the `filtering_stats` copy | **Reassigned to PC.5**, same argument: which of the 11 keys a curator needs is a surface decision. PC.2 already closed the one key that had an owner (`alias_candidates`). |
| `_save_result` stores pre-filter entities beside post-filter stats | **Reassigned to PC.5.** Needs a decision about what a stored run means, and that decision is visible only through the surface that renders it. |

**Review pushed back on this, and the objection was checkable rather than
rhetorical**: the first version of the reassignment edited only PC.6's section, so
the rows lived in this table with a forwarding address and PC.5's scope and AC
covered none of them. All three are now in PC.5's bullet list and its AC is
widened. The `metrics` row also carries review's point that it has nothing to do
with a curator door and needs no surface to decide — delete the writes or give
them a reader.

Reassigning three rows out of the phase that owned them needs its reason on the
record: PC.6's remit is *configuration that expresses one intent* — a flag either
works or says why it cannot. All three are about what a completed run reports,
which is a different question with a different owner. The alternative was to
"decide" them here without the surface that would show whether the decision was
right, which is how `FilteredResult` acquired four fields nobody reads.

## The dead-knob class beyond PC.6's scope

Measured during PC.6's review, recorded so the next phase can scope it rather than
rediscover it. Same defect, other packages, non-test consumers only:

| module | fields with no consumer |
|---|---|
| `pipelines/summarization/.../config.py` | 11 — `critic_model`, `expansion_tokens`, `max_branch_depth`, `max_correction_rounds`, `max_skeleton_points`, `max_summary_words`, `min_entity_mentions`, `num_density_rounds`, `top_k_sentences`, `topic_threshold`, `use_pca_fallback` |
| `packages/ontology-manager/.../config.py` | `ontologies_dir` |

PC.6 swept `entity_filtering` — a field-level rescan there now finds none — and
declared the rest out of scope rather than doing it badly at the end of a phase.
The two guards it built (`tests/test_compose_env_is_consumed.py` and the AST guard
in `services/extraction/tests/`) are both scoped to what they were written for and
say so in their docstrings; widening either is the natural home for this.

## Found by PC.2's adversarial review

| finding | boundary | owner | note |
|---|---|---|---|
| `POST /apply` performs no band or type check | curator UI → `entity_resolution` router | **PC.5** | It applies whatever cluster the client echoes, so the router docstring's "only `auto_merge` candidates may be applied" is enforced only by the frontend. Pre-existing from K.5; PC.2 is what puts cross-type pairs into the review list, so the discipline now matters more. Not changed in a phase about identity. |
| `CandidatesResponse.auto_merge` + three counts fetched, never rendered | `/candidates` → resolution page | **PC.5** | The page shows `candidates.review` only. K.5-era surface and a UI decision, not forgotten wiring — but PC.2 changed what flows through it: `fold_equal` is a new AUTO producer, so a same-type case-only duplicate lands in a band no curator sees and `okf_import_service` applies unattended, with "Keep apart" unreachable for it. |
| Office-of / organ-of name pairs have no home | name shapes → nothing | **later phase** | `Minister van BZK` / `BZK`, `Burgemeester van Rotterdam` / `Rotterdam`. Real and worth surfacing, but as a RELATION — a merge proposal asserts they are one entity, which is false. PC.2 removed them from the containment rule rather than leave a leading question in front of a destructive button. |

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

- ~~`KGResolver.report["aliases_registered"]` is initialised and logged, never
  incremented — the INFO line always prints `0`.~~ **FIXED (PC.2)**, pinned in
  both directions: it counts a successful write and not a failed one.
- `run_filtering_only` builds a bare `FilteringConfig()` (fuzzy and embedding dedup
  **off**) while the main extraction path enables both. The two paths are not the
  same pipeline.
- ~~`entity_alias` is SCHEMAFULL and declares five fields; `register_alias` writes
  four more, including `verified`. No migration declares them.~~ **FIXED (PC.2)**,
  migration 78. Worse than filed: SurrealDB drops the undeclared fields silently,
  so on every fresh database an alias lost all its provenance and
  `vault_sync_service` could never export one. Invisible against `staging`, whose
  table predates the schema lock.
- `entity.status` is a free-form `str` with four values in use, no enum.
