# Track PC — Pipeline coherence

**Origin**: the pipeline review of 2026-09-02
(`claudedocs/extraction-pipeline-review.md`), run after Track N.4d closed. The
review measured a live corpus of eight documents through the production path and
found that the individual stages are correct while the **chain** is not coherent.

**Why a separate track rather than N.5.** Track N is "evidence-first extraction &
abstention" — a closed narrative from candidate anchors through abstention to the
type boundary, with its own decision log D-N4-1..14. The findings below sit in
schema review (Track B), entity resolution (K), typing (L), and the default
filtering configuration; none of them is N's to decide, and folding them into N.5
would make that track unfinishable. N.5 keeps only N's own debts.

**Baseline for every claim here**: the review's measured run — 8 documents, 70
chunks, 124 entities extracted, 117 rows in the graph, 39 relations, 0 gaps.
Raw data: `claudedocs/pipeline-review-corpus.json` (+ `-bennett`, `-achterhoek`).
Harness: `scripts/n_pipeline_review_run.py`.

---

## The through-line

Three sentences that explain every phase below.

1. **Nothing looks at the graph that is already there.** 124 entities in, 117
   rows out; every collapse happened inside one document's batch.
2. **Nothing reaches the person who is supposed to decide.** The accept queue has
   no writer; gaps and proposals have no reader; alias candidates are discarded.
3. **The defaults do not express one intent.** Half the quality stages are off, so
   the shipped pipeline is an LLM-to-graph pass-through, and each stage's flag was
   set independently of the others.

---

## PC.1 — Give the curator queue a writer — ~1d

**The finding (review G0).** Pass 1 works and persists: after eight documents
`pass1_results` holds 14 rows with coverage 0.42–0.85 and 1–3 proposed types each.
But `notebook_schema.pending_extensions` is empty, and
`NotebookSchemaRepository.add_pending_extension` has **no production caller** —
only its own roundtrip test. `notebook_schema.coverage_pct`, documented as driving
the B.3c soft-nudge, is likewise never written and sits at `0.0`.

Everything downstream operates on `pending_extensions`: accept, reject, the
`PendingExtensionsPanel`, and — through the accept step — the whole of N.4d.1–3.
**Four shipped sub-phases are unreachable from a real run because of this.**

- Write the deduped Pass-1 proposals into `pending_extensions` at the end of
  `run_multi_schema`, and roll `coverage_pct` from `best_coverage`.
- **Decided**: proposals are per-document, `pending_extensions` is per-notebook,
  so re-proposing the same type across documents must not create duplicates. Key
  on the trimmed, lowercased `type_name`.
- **Deviation from the first draft, recorded rather than silently dropped**: that
  draft said "keep the highest-coverage rationale". Proposals do not carry a
  coverage figure — coverage is per `pass1_results` ROW, across all of a pass's
  proposals — so there is nothing to rank them by without inventing it.
  First-seen wins instead. Attaching per-proposal coverage is a follow-up worth
  doing only if a curator asks for the better rationale.
- **Also decided**: `excluded_types` (the curator's explicit soft-delete) blocks
  a re-proposal, and a name that cannot survive the accept/reject route as a path
  segment is refused outright rather than queued unactionable.
- **Known race, not closed here**: `merge_pending_extensions` and
  `set_coverage_pct` are read-modify-writes over the whole row, inherited from
  `add_pending_extension` — but PC.1 is the first production caller, so the race
  becomes reachable the moment a bulk upload ingests two sources into one
  notebook concurrently. The fix is a server-side append or an optimistic version
  check, i.e. a change to the repository's write contract. Owned by **PC.6**
  (configuration and robustness).
- **Known gap, not closed here**: a REJECTED proposal returns.
  `reject_extension` drops the row and records nothing, and there is no
  `rejected_extensions` field — a durable "no" needs a new field and a migration.
  Owned by **PC.5** (the curator-surface phase). Two tests pin it, one per layer,
  because the repository's `reject_pending_extension` has no production caller —
  a curator's Reject button goes through `SchemaEditService.reject_extension`,
  keyed on `type_name`. Whichever layer PC.5 records the "no" in, one of the two
  fails. (The first version of that guard contained no rejection at all and could
  not fail; the second rejected through the dead method and would have stayed
  green if PC.5 wrote the trace in the service. Both were review findings.)
- **AC**: after two documents proposing overlapping types, the Schema tab shows
  each type once; accepting one runs the N.4d.3 placement and returns a verdict;
  `coverage_pct` is non-zero and matches the mean over sources of each source's
  best CURRENT coverage — `pass1_results` is append-only and one extraction
  writes one row per applied schema, so "current" means the rows sharing that
  source's newest `run_id`, and the best schema WITHIN that run.
- **Decided by the user (round 3)**: the schema is established **per document**;
  only when that genuinely fails does the notebook's own history decide. So the
  row is created with NO declared base ontology, and nothing is forced onto any
  extraction until a curator chooses one in the Schema tab.
  - Two attempts guessed a value for that field and both were wrong the same
    way: whatever it holds is merged into the applied set of every later run at
    confidence 0.85, ahead of the document. `config.ontology_name` ("general")
    would have broken the TTL download outright; `DEFAULT_BASE_ONTOLOGY`
    ("scholarly") is detected for ZERO of the fourteen live sources while all 17
    `pass1_results` rows ran against `policy_themes`.
  - **The forced base was compensating for a blind detector.** Detection was
    scored against the FIRST CHUNK capped at 2000 characters; in this corpus the
    first chunk of a parsed PDF is a title fragment with a median length of 66
    characters. Measured: detection fired for 2 of 14 sources. With a sample of
    40 windows spread across the document — the knee of the curve, swept — it
    fires for 13 of 14, at 4 ms for all eleven ontologies. Budget and window
    count are both swept rather than chosen; 13/14 is the ceiling. The budget is
    SHARED across the windows rather than consumed in index order — a review
    measured that the first shape stopped partway through the spread on
    long-chunk documents and scored them on their first 41%, re-introducing the
    head bias one order of magnitude larger than the cover page it replaced.
  - **The cost, stated**: 11 of 14 sources move from the legacy single-schema
    path to multi-schema. That is more LLM calls per document and a different
    result shape (`type_tags` and `primary_type` are now set where they were
    not). It is the point — the legacy path is where Pass 1 never runs — but it
    is a real change in what every ingest does, not a free improvement.
  - **Known consequence of the empty base**: on a notebook's FIRST accept, the
    N.4d.3 placement composes an empty forced vocabulary (no base, no accepted
    extensions yet) and so places against nothing. It still returns a verdict and
    is advisory, so the AC holds, but the first accept is less informed than
    later ones. Owned by PC.5 alongside the curator surface.
  - The remaining document falls back to the notebook's most attempted schema
    (ties broken by mean coverage), at most one, never overriding a document that
    detected for itself, and returning nothing when the notebook has no history —
    in which case the legacy path is taken exactly as before.
- **Also decided (review round 2)**: the orchestrator stamps a `run_id` per
  extraction into the FLEXIBLE `pass1_metadata`. Two grouping rules were tried
  before it and both could only let coverage RISE; only a run boundary expresses
  "these rows are the current measurement, those are history", which is what the
  soft-nudge's own flow (low coverage -> edit the schema -> re-extract) needs.
- **Guard**: the test drives `run_extraction`, not the repository — this whole
  finding is that the repository method works and nobody calls it.

> **PC.1 SHIPPED** (review-approved attempt 5, merged `0786e0a6`; full report in
> `reviews/phase-PC.1-attempts-1-5.md`) — `ensure_row`, `merge_pending_extensions` and
> `set_coverage_pct` on the repository; the extraction path creates the row
> BEFORE schema detection and records the Pass-1 outcome after it. Verified
> against the project's own database: row created, `coverage_pct` 0.0 -> 0.508,
> 111 stranded proposals collapsed to 79 queued types, a repeat merge adding 0.
>
> **What the live run changed about the phase.** The first implementation was
> fully tested, fully green and completely inert on real data: it created the row
> in `_record_pass1_outcome`, which sits AFTER `_run_multi_schema`'s early return
> to the legacy path — and that early return is exactly what happens when no
> schema row exists. The cycle (no row -> nothing forced -> detection finds
> nothing -> legacy path -> no Pass 1 -> no row) can only be broken before
> detection. **Binding for the rest of this track**: a fix for a "nothing ever
> writes X" finding must be proven on data where X is genuinely absent, not only
> on a fixture that says it is.
>
> **Round 2 found the same class of thing one layer up.** The row was created,
> but with `config.ontology_name` — a per-request parameter defaulting to
> "general" that nothing sets — as the notebook's declared base ontology. The
> live row proved it was not hypothetical: `base_ontology: general`, written by
> PC.1's own verification run, and the schema TTL download for that notebook
> raised `AttributeError` because `general.yaml` cannot be parsed by
> `load_yaml_ontology`. Repaired in place (`scholarly`, queue of 79 and coverage
> 0.508 intact) and re-verified on a notebook where the row was genuinely absent.
> **Second binding lesson**: per-request state must not set per-notebook state.
> The two look alike at a call site and diverge everywhere else.
>
> **Carried out of PC.1**: a durable "no" for rejected proposals (PC.5); the
> read-modify-write race, now reachable because PC.1 is the first production
> caller (PC.6); and the Ollama context truncation measured during verification
> (PC.6, review finding R4b).

## PC.1b — Derived state that reaches a reader — ~1.5d

**The finding, named by the user after Track N closed**: it is as though state is
not preserved between pipeline steps — per document, per notebook, and across the
graph. Substantially right, and promoted to its own phase ahead of PC.2 because
two of its fixes are instruments PC.2 and PC.3 need.

The PAYLOAD always survives. What is repeatedly lost is the DERIVED state — what a
step measured, judged or attempted — and it is lost at HANDOFFS, never inside a
step. A full trace found ~20 instances, not the six first recorded. The full table
lives in `handoff-inventory.md`; this entry records the decision.

**The user asked for the most structural fix and expected a run object. Measured,
it is not.** `FilteredResult` is already one, measured with the guard's own counter,
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

Both experiments have already been run — the untyped bag
(`ExtractionResult.metadata`) and the typed carrier — and state died in each. The
fields are not orphaned because nothing carried them; most are carried faithfully
to a place where nothing reads them. A third carrier moves the corpse.

So the structural fix is the **invariant**: a producer names its consumer or it
goes — the generalisation of what N.5b decided about the Hearst miner, made able to
fail. The run object keeps a falsifiable trigger: build it when the inventory shows
three or more boundaries needing the same field. Today none does.

- **The inventory** — every boundary classified wire / delete / accept-with-owner.
- **The invariant** — `tests/test_derived_state_has_readers.py`, with a walker
  control, a detector control, a reader-counter control and a mutant control that
  runs the same detector over a planted dead field in an in-test source string.
- **CI** — `unit-guards.yml`. Before this, NO CI job ran any Python unit suite, so
  a guard placed in a package test directory ran only on its author's machine.
- **W1** the soft-nudge decision reaches `notebook_event`; **W2** the merge's
  `type_tags` reach persistence; **W3** the persist counters reach the summary;
  **W3b** the re-filter stops erasing alignment state.
- **AC**: the inventory is checked in and every row has a disposition and an owner;
  the invariant fails when a derived-state field has no reader and no owner; the
  wired boundaries each have a test that fails when the state stops crossing; W1 is
  proven on a database where `notebook_event` is genuinely empty.

> **PC.1b SHIPPED 2026-09-03** (review-approved attempt 4; report in
> `reviews/phase-PC.1b-attempts-1-4.md`) — see `handoff-inventory.md` and the phase report.
> Turning the CI job on required making `tests/` green first, and it was not: four
> files have been UNCOLLECTABLE since the monolith→workspace cutover of 2026-04-24,
> and four more failed because they patched `execute_query` in a module the service
> under test does not use. Both were invisible because nothing ran the suite.
>
> **Four review rounds, and the finding is the shape of the rounds themselves.**
> Rounds 1 and 2 closed the specific dead fields the reviewer planted — a denylist
> entry, one more parsed file — and round 2's review named why that was not
> progress: the counter attributed a read by bare attribute name with no
> information about what object it belonged to, and no denylist fixes that. Round 3
> inverted the guard — every derived-state field DECLARES its consumer and the
> guard verifies the declaration — and round 4 closed the declaration's own escape
> hatches.
>
> **Binding for the rest of the track**, in order of how much they cost:
>
> 1. A carrier is not a consumer. Before adding a field to a result model, name
>    what reads it — the invariant now asks.
> 2. Closing planted instances is not closing the property. The tell is that each
>    fix names a field rather than a rule.
> 3. A guard's escape hatch is most dangerous when it looks like ordinary usage:
>    the bare-string declaration was a blocker because it was the shape that same
>    table had one commit earlier.
> 4. A correction that leaves the corrected text in place is not a correction, and
>    reporting one as done converts a review round into a round that verifies
>    nothing.

## PC.2 — One identity — ~1.5d

**The finding (review I2, R2, G2).** `KGResolver` registers an alias
automatically on a fuzzy match (`register_aliases=True` by default) while concept
alignment refuses to on the explicit grounds that merging identities must be a
deliberate act (D-N4-9). One decision, two opposite policies, in one pass.
Separately, the same six-line normalisation is copied four times
(`EntityDeduplicator._normalize_key`, `FuzzyResolver._normalize`,
`KGResolver._normalize`, `concept_alignment._normalize`), and `Brede Welvaart` /
`brede welvaart` are two rows in the measured graph.

- One shared normalisation, imported by all four call sites; the duplicates go.
- One alias policy, stated once: decide whether a fuzzy match may auto-register,
  and make both stages obey it.
- Surface the long-form/short-form candidates the fuzzy tier structurally misses
  (25 pairs in the measured graph, e.g. `Binnenlandse Zaken en Koninkrijksrelaties`
  beside `Minister van Binnenlandse Zaken en Koninkrijksrelaties`).
- **AC**: the normalisation exists once; a test fails if a fifth copy appears;
  the measured 25 pairs are reachable by a curator.

**Status: DONE** — `feature/track-pc2-one-identity`, 12 commits, **APPROVED** after
three review rounds. Report: [`phase-PC.2-report.md`](./phase-PC.2-report.md).
Review: [`reviews/phase-PC.2-attempts-1-3.md`](./reviews/phase-PC.2-attempts-1-3.md).

All four AC items met. `fold_for_comparison` replaces the four copies (verified
byte-identical by execution over 18 adversarial inputs first) and an AST guard
fails on a fifth. `register_aliases` now defaults to False in both the config and
the constructor, matching D-N4-9. The long-form/short-form pairs reach the curator
through `_score_containment`.

Three things the plan did not anticipate, each with its measurement:

- **The containment rule needed three attempts.** Unanchored containment — the
  rule `concept_alignment` already had — manufactures exactly the merge the dedup
  config refuses. Head-anchored with a free length guard yields 315 candidates and
  pairs a place with every organisation named after it. Head-anchored **and**
  curated yields 82, noise gone. The measurements are in the report.
- **`entity_alias` loses its provenance on every fresh database.** SCHEMAFULL,
  five fields declared, nine written, the rest dropped silently. Invisible against
  `staging`, which predates the schema lock. Migration 78. This is also why the
  determinism fix could not have been verified on `staging` alone.
- **Two labelling bugs in the apply path.** The frontend labelled the survivor
  `name_a` regardless of who won, contradicting the server-side rule it mirrors.

Three review rounds, and every blocker plus two of three round-2 majors were the
same defect: **a value produced for a surface that never consumes it** — the card
helpers imported and never called, the containment head run computed and
discarded, and a `.test.ts`/`.test.tsx` name collision that made `tsc` skip the
file guarding the first blocker. PC.1b's producer-must-name-its-consumer invariant
is Python-only and all three sat on the Python/TypeScript boundary. **A
cross-boundary guard is the track-level follow-up this phase argues for.** Round
3's sweep confirmed no fourth instance: 40 response-model fields against every
identifier in `frontend/src`, 0 orphaned.

Round 1 returned REVISIONS_NEEDED and changed two things materially, both of which
the first report had claimed as done:

- **The curator card was never wired.** `candidateTypeLabel` was written, tested
  and imported — and called nowhere, so a cross-type candidate still rendered one
  name twice with the deciding fact hidden. Now covered by a jsdom test that
  mounts the real component.
- **Migration 78 was a bare DEFINE.** A SurrealDB DEFAULT does not backfill, and a
  strict type then blocks every UPDATE to a pre-existing row — the class
  migrations 61, 64 and 65 already fixed twice here. It now carries the coalescing
  repair, proven against a forged legacy row.

And the correction that matters most for later phases: **an organ OF X is not X**.
The curated affix list was built from corpus frequency and carried the governance
affixes, so it proposed `Burgemeester van Rotterdam` ~ `Rotterdam` and
`Gemeenteraad van Amsterdam` ~ `Amsterdam` as merges — and the tests pinned one of
them as correct. Cut from 40 affixes to 11 (`waterschap` went in round 2, once
review showed it re-created the very class the cut removed). This gives up the
plan's own
`Minister van BZK` example: that class is real but belongs in a later phase as an
**organ-of relation**, not a merge.

Handed to **PC.4**: 8 names the graph held twice under two types, six of them
`programme` against `topic`. The working corpus was emptied after the measurement,
so PC.4 should re-derive the list from the cross-type band on real data — what
carries forward is the shape of the finding, not the rows. The report records
three figures that do not reconcile rather than picking one.

## PC.3 — Look at the graph that is already there — ~1.5d

**The finding.** KG resolution (stage 10) is what matches a new mention against
existing entities, and the app's default `FilteringConfig` does not enable it. So
every document's entities are written fresh. Three convenanten naming the same
ministers produced 58 entities with no consolidation.

- Decide whether cross-document resolution belongs in the default path. If yes,
  enable it and measure the cost; if no, say so in the config docstring and accept
  that the graph is per-document.
- **From PC.1b's inventory**: `kg_resolution_report` (kept with no reader — this phase's AC needs its measured figure), `source_chunk_id`/`source_grounding`/`extraction_context` never persisted, `incremental_report`, and the raw write key (`upsert_entity` keys on the unfolded `canonical_name`, which is why case-variants are two rows).
- **AC**: re-running the review corpus produces materially fewer than 117 rows,
  with a named, measured figure; `M.C.G. Keijzer` is one entity.
- **Watch**: stage 10 is also where the automatic alias registration lives, so
  PC.2's policy decision lands before this is switched on.

## PC.4 — Stable typing — ~1d

**The finding (review I1).** `resolve_ontology_type("Person", …)` returns `person`
under `general` and `concept` under `deals`/`government`. Which ontologies apply
is decided by a content score over the document, so the same label lands in a
different canonical bucket depending on the document. In the measured corpus,
`Provincie Drenthe` is `programme/RegioDeal` while `Provincie Overijssel` is
`administrative_area/Provincie`, and 38% of all entities are `concept` or `other`.

- Make the canonical answer for a label independent of which applied set happens
  to contain it — either by always including the base vocabulary in the applied
  set, or by resolving against the union rather than the selection.
- **From PC.1b's inventory**: PC.2's cross-type fold-equal candidates are this phase's evidence — labels holding two canonical answers on real data rather than a sweep over shipped ontologies.
- **AC**: a label's canonical type is identical across the eight review documents;
  a sweep over the shipped ontologies shows no label with two canonical answers.

## PC.5 — A door for the curator loop — ~1.5d

**Also owned here (from PC.1's review)**: a durable "no". `reject_extension`
records nothing, so a rejected proposal is re-queued by the next document that
proposes it. Needs a `rejected_extensions` field, a migration, and the
`merge_pending_extensions` check — at which point PC.1's
`test_a_rejected_type_does_come_back` must be inverted.


**The finding (review G1).** `ontology_gap` and `schema_proposal` are written by
`OntologyEvolutionAgent`, whose only production caller is N.4d.4's recorder. No
API route, MCP tool, CLI or frontend reads either table. Meanwhile
`auto_propose=True` with `frequency_threshold=5` writes proposals unasked.

- Either surface both tables (list, accept, reject) or turn `auto_propose` off
  until something reads them. Writing into a room with no door is the one option
  to rule out.
- **From PC.1b's inventory**: `ontology_gap` / `schema_proposal` have readers but no route; and `notebook_event` is now written by PC.1b, so a dismissal path is this phase's to design.
- **AC**: a proposal created by the threshold is visible and actionable, or it is
  not created.

## PC.6 — Configuration that expresses one intent — ~1d

**Status: DONE (review pending)** — `feature/track-pc6-config-coherence`.
Report: [`phase-PC.6-report.md`](./phase-PC.6-report.md).

**The finding (review R1).** `ENABLE_CONCEPT_ALIGNMENT=true` did nothing in the
measured run because alignment classifies only entities KG resolution marked
`is_new`, and KG resolution is off by default — a feature reachable only by
changing a second, unrelated default.

- One named profile per intent, or an explicit dependency check that refuses an
  incoherent combination loudly.
- A startup or first-use check that the routed extraction model actually answers.
- **From PC.1b's inventory**: `validation_report` (kept with no reader — stage 11
  is inert because no production call site passes an ontology), the alignment
  report keys dropped at the `filtering_stats` copy, `metrics` rows nothing reads,
  and `_save_result` storing pre-filter entities beside post-filter stats.
- **AC**: enabling a feature either works or says why it cannot; the measured
  "flag on, zero effect, only a warning" state is unreachable.

### Two claims in the original plan that measurement disproved

Both were carried from review R1/R4 and are corrected here rather than left to be
inherited as fact.

**The Ollama `num_ctx` paragraph (R4b) was wrong in its specifics.** It said
esperanto never sends `num_ctx`, so Ollama truncates every prompt over its runtime
default and the JSON instructions in the prompt's tail are what get cut. Measured
(2026-09-04, adversarial review of a branch that was reverted in full):

* `num_ctx` **is** sent, unconditionally, by both callers —
  `shared/model_routing.py::_call_ollama` and esperanto's own Ollama provider. A
  request-level value also *overrides* a Modelfile `PARAMETER`, so baking context
  into a model variant is inert on this path.
* Truncation discards the **head**, not the tail — verified with markers at both
  ends of an over-long prompt. Instructions at the end survive; document content
  is what is lost.
* Extraction is **not** truncating: `EXTRACTION_CHUNK_SIZE=4000` chars plus the
  regiodeal ontology gives a worst-case prompt of ~4,530 tokens against the 8,192
  already sent.

The genuine finding underneath is narrower and belongs to PC.6: the context lives
in `model_routing.yaml` per step, and nothing checks that a step's `num_ctx` is
large enough for the prompts that step builds. The `-ctx16k` model-variant
approach is **not** the answer here; it was tried and reverted.

**The `gemma2` fallback claim has no basis in the code.** The plan said that with
no `default_models` row the router falls back to Ollama `gemma2`, which is not
installed. `gemma` appears nowhere in the repository. The real state, measured: all
eleven routed local models are installed, `default_models` holds zero rows, and
that empty row is a legitimate "not configured yet" rather than a fault — models
for summaries and transformations are added later.

### What the phase found instead

**Two model-configuration systems with opposite privacy defaults.**
`model_routing.yaml` (pipeline steps × privacy) defaults to `internal`, local
only; `default_models` + `route_resolver` (LLMTask × PrivacyMode) defaults to
`CLOUD`, prefer cloud and fall back to local. Each is defensible alone; together
they mean the same document is treated as local by one path and cloud-eligible by
the other. **Decided (user): keep both, one authoritative per domain, with a
bridge check that fails when they contradict** — merging touches both code paths,
the `/api/models/defaults` surface and the J.4 telemetry for no behaviour anyone
asked for.

**Startup seeded model rows for a provider the config had retired.** `seed_nim_routes`
runs on every boot and is where the single NVIDIA `model` row came from — a
configuration contradicting itself once per startup, for a vendor nothing is
allowed to call. Now skipped while the provider is declared unavailable.

**NVIDIA is declared unavailable rather than deleted** (user's decision): the
`public` routes stay as the record of what the cloud path was, and resolving one
raises `ProviderUnavailableError` with the reason instead of reaching for a key.

---

## Sequence

PC.1 first — it unblocks four already-shipped sub-phases and is the smallest.
Then **PC.1b** (added 2026-09-03, ahead of PC.2 at the user's direction: it ships
the instruments PC.2 and PC.3 measure with, and the invariant that stops the next
orphan arriving while they work). Then PC.2 (the policy decision PC.3 depends on),
then PC.3 (the largest effect on the data), then PC.4, PC.5, PC.6 in any order.

Rough total ~8.5 days. PC.1 alone was worth doing immediately regardless of the
rest, and PC.1b turned out to be worth doing before PC.2 for a reason the plan did
not anticipate: turning on a CI job that runs the guards revealed that `tests/`
had been red on main — four files uncollectable since April, four more failing —
because nothing ran it.

## Standing rules inherited from Track N (D-N4-14)

1. A test double reproduces the real method's failure **RETURN**, not a raise it
   never performs.
2. At least one fixture builds the **production** argument set, not a superset.
3. A guard that reads a collaborator's value is exercised against the **real**
   collaborator at least once.
4. Every phase is gated by adversarial review before merge.

These were paid for three times over in N.4d; both blockers there were findable by
no other means.
