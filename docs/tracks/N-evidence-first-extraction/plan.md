# Track N — Evidence-first extraction & abstention — Sprint Plan

> Status: 🔨 IN PROGRESS (2026-09-01). **N.1 fully SHIPPED** across PRs #78
> (candidate anchors: TF-IDF + noun-chunks + EntityRuler stub + budget-safe prompt
> threading), #79 (N.1-deps: `spacy` + `en_core_web_sm` first-class), #80
> (language-aware NL/EN model selection + complementary spaCy+regex merge +
> edge-word cleanup) — all review-APPROVED and live-validated on real Dutch (Regio
> Deal, lang→nl) AND English (academic paper, lang→en) PDFs. Model decision
> resolved: **NL + EN side-by-side, picked per document.**
>
> **N.2 SHIPPED** (merge 2d78953, branch `feature/track-n2-hearst-isa`) — the
> Hearst is-a miner is **spaCy noun-chunk based**, NOT regex. A first regex draft
> produced clause/verb garbage on live prose (`… need to --is_a--> public sector`)
> and leaned on a `_clean_list_item` verb-truncation band-aid the user rejected as
> a monkey-patch; rebuilt on `doc.noun_chunks` + POS-bounded coordination lists,
> sentence-bounded hypernym anchor (`_sentence_span`/`_gap_is_clean`), EN+NL cues,
> seeded only between already-extracted entities (precision gate, provenance
> `relation_source="hearst"`, conf 0.5). Corpus language detected once + spaCy
> model loaded once, shared by N.1 candidates and N.2 mining. Adversarial-review
> APPROVED after a MAJOR fix (anchor was crossing sentence/clause boundaries) +
> two minors (relation_type-aware dedup, lang-scope alignment). Live-validated on
> real EN/NL PDFs (clean `→ policy areas` / `→ partners` clusters, zero verb
> garbage). **spaCy runs fine on this repo's WSL /mnt venv** — the earlier
> "/mnt blocks spaCy" note was disk-full, not WSL; corrected in the tests.
>
> N.2 follow-ups (non-blocking, from review — do in N.3/N.5, not now):
> (1) residual spaCy chunk noise (`help`/`Governance matters` mis-chunked into a
> list/hypernym) is inert through the precision gate but worth an **audit-side
> note if Hearst precision is measured on live corpora**;
> (2) Hearst seeds have no live precision/recall number yet — fold a small
> Hearst-precision sample into the N.5 regression metrics.
>
> **N.3 SHIPPED** (merge 6132110, branch `feature/track-n3-abstention`) — the
> article's core lesson (selection/abstention over generation) at extraction time:
> (1) an `INSUFFICIENT_EVIDENCE` abstention clause in the Pass-2 prompt (a
> furniture-only chunk returns empty entities, framed as the single exception to
> exhaustive recall); (2) `not_a_concept.py`, a stricter extraction-time cousin of
> entity-filtering's `noise_filter` — a deterministic pre-pass (high-precision
> reject of UI/nav/reference/boilerplate; fast-accept of specific-label or proper-
> name entities) + a BATCHED LLM-judge (default ON, D4) for the ambiguous middle,
> KEEP-on-doubt everywhere; (3) `extraction_metrics.py` (app-main, mirrors M.5)
> deriving `over_generation_rate` + `abstain_rate` from raw counts run_pass2 now
> records in metadata. Env flags `EXTRACTION_NOT_A_CONCEPT` / `_JUDGE` (both ON).
> Adversarial-review APPROVED after a MAJOR fix (a SPECIFIC schema label must
> override the homograph field-word reject so a specifically-typed "Total"/"Page"
> is never dropped) + two minors (explicit-only judge verdicts; stripped relation
> drop-set).
>
> N.3 follow-up (non-blocking, from review): `_REJECT_ALWAYS` hard-rejects the
> single-token action verbs `download`/`print`/`share`/`subscribe` regardless of
> label — marginal real-entity readings ("Share", "Print"); revisit only if a live
> corpus surfaces them.
>
> **N.4 attempt 1 PARKED — chapter RE-PLANNED (v2).** Branch
> `feature/track-n4-concept-alignment` @ `da0bda6` is **not merged**. Adversarial
> review returned REVISIONS_NEEDED with 2 blockers + 5 majors
> (`reviews/phase-N.4-attempt-1.md`); both blockers are DESIGN errors, so the
> chapter was rewritten instead of patched (user decision 2026-09-01: park and
> re-plan). Headline causes: (1) lexical containment was treated as `is_a` — on real
> Dutch names it means *alias / part_of / named_after* at least as often as
> *subtype*, and `KGResolver`'s fuzzy tier feeds the aligner exactly those alias
> pairs; (2) the output was structurally unreachable — seeded edges were emitted
> before the ontology filter that drops off-batch endpoints, and the candidate fetch
> queried the canonical `entity_type` column with a rich Track-L label, returning
> `[]` by construction.
>
> **Planner correction**: the `evolution` agent DOES exist —
> `packages/ontology-manager/src/ontology_manager/evolution.py`
> (`OntologyEvolutionAgent`: `record_gap` → frequency threshold → `SchemaProposal`).
> Attempt 1 searched only `apps/`+`pipelines/` for a *directory* and wrongly recorded
> it as missing. v2 wires it: a NOVEL verdict records an ontology gap.
>
> **N.4a SHIPPED** (merge `fad6833`, branch `feature/track-n4a-ontology-subsumption`)
> — ontology-grounded verdicts + falsifiable evidence, no seeding and no workflow
> stage. Attempt 1 of N.4a returned REVISIONS_NEEDED (4 blockers + 4 majors, all one
> class: evidence that reported an OBSERVATION as the inference it would license);
> attempt 2 APPROVED after a restructure — `_Fetch.ok` separates a raised fetch from
> an empty result, `NeighbourProbe` separates the three "nothing was compared"
> causes, a local `_cosine` returns `None` for incomparable vectors instead of
> inheriting `_cosine_similarity`'s out-of-band `0.0`, judge items are keyed by index
> rather than surface form, and `resolve_types` is asserted against the REAL
> `canonical_bridge` instead of a stub of itself. Six non-blocking minors are carried
> forward as binding line items C1–C6 (see the N.4 chapter); **C1 must be closed
> before N.4c filters gap-recording on `reason_code`.**
> Report: `reviews/phase-N.4a-attempts-1-2.md`.
>
> **N.4b SHIPPED** (merge `19add9c`, branch `feature/track-n4b-placement-seeding`)
> — Stage 15 runs AFTER ontology validation and centrality, which is what makes the
> alignment's output actually reach the graph without perturbing what came before.
> Both failure modes were MEASURED rather than argued: the real ontology filter
> yields `surviving relations: []` for such a seed, and the real graph analyser
> scores the same entities 0.5 vs 0.25974 with and without one. Seeding is a pure
> function of the recorded verdict and refuses self-referential, untyped, dangling
> and duplicate edges while stamping both endpoint types (D-N4-5). Attempt 1 hit a
> BLOCKER — a self-referential `is_a` (`Deal is_a Deal`), reachable because an
> entity whose surface form equals an ancestor type name matched its own row;
> fixed at the verdict AND the seeding level. Attempt 2 APPROVED, verified by
> MUTATION TESTING: disabling each new guard makes a specific test fail.
> Report: `reviews/phase-N.4b-attempts-1-2.md`; residuals R2–R5 carried to N.4c.
>
> **N.4c NOT MERGED — the instance-level premise is abandoned (D-N4-12).** Branch
> `feature/track-n4c-verifiable-subsumption` @ `68c544f` widened `find_by_type`'s
> projection to return the rich type, then used it to make the subsumption tier
> "checkable" and enable it. Review returned REVISIONS_NEEDED (3 blockers, 6
> majors) and the decisive finding was structural, not a defect list: **nothing in
> this codebase ever creates a node representing an ontology type** — every
> `entity` row is written from a text mention — so "this entity is narrower than
> that entity" is not a claim the data can support. Two supporting measurements:
> `type_tags` is NOT an ancestor trail (the no-schemas persist path writes
> `[raw_label]`, and the repository unions it across upserts), so the veto that was
> meant to reject siblings does not; and `ConceptAlignmentConfig.enabled` is still
> `False`, so the commit's claim that the pipeline now "seeds under the shipped
> configuration" was itself an over-claim — the track's recurring failure mode, this
> time in a commit message.
>
> That was the THIRD instance-level attempt to fail identically (lexical
> containment → name matching → declared type). **D-N4-12** moves subsumption to
> the TYPE boundary, where both sides of the question are types and the parent slot
> is currently filled with an unvalidated guess; D-N4-11's descendant sweep is
> superseded, because an accepted BROADER_THAN becomes a SCHEMA re-parent whose
> descendants are inherited rather than enumerated.
>
> **▶ RESUME AT N.4d** (see its chapter): N.4d.1 type placement verdicts → N.4d.2
> the judge over the sibling set → N.4d.3 apply as a schema edit → N.4d.4 gap loop +
> reachability (C1 first). Then N.5 (regression gate + docs; also lands the N.2 +
> N.3 follow-ups above). The **evidence-packet clustering stays DEFERRED — measure
> first** (user decision, §5).
>
> Unrelated pre-existing breakage noticed during N.4 review (NOT from Track N):
> `pipelines/entity-filtering/tests/test_llm_matcher.py::TestMatchPair::
> test_calls_ollama_for_unknown_pair` fails on `main` with `'LLMMatcher' object has
> no attribute '_agentic_enabled'` — `__init__` does set it, so the test's
> construction path likely bypasses `__init__`. Needs its own fix.
>
> Live-testing lessons folded in: en_core_web_sm on Dutch produced verbal-phrase
> junk → per-document model selection; both spaCy models fragment long compound
> proper names → merge spaCy WITH the regex source; academic prose over-generates
> generic noun-phrases → N.3 abstention matters.
> Origin: assessment of a Medium article ("Which parts of ontology discovery
> actually need an LLM?") against open-notebook's KG stack.
> Scope: the two applicable, high-value wins the article surfaces — a **pre-LLM
> evidence/candidate layer** and a **first-class abstention gate** — plus a
> **concept-level alignment taxonomy** phase.
>
> **Resolved decisions (2026-08-31):** (1) spaCy is a first-class dependency for
> **noun-chunk** candidates, with a rule-based **`EntityRuler` NER stub** (domain
> gazetteer wired but empty, populated later); (2) the **evidence-packet /
> cluster-before-LLM inversion is DEFERRED** with a measurement trigger; (3) the
> **LLM-judge is default ON** as the arbiter for the semantic gates (not-a-concept,
> RELATED/NOVEL), behind a cheap deterministic pre-pass.

## 1. Context — where open-notebook already stands

open-notebook already embodies the article's core thesis ("surround the LLM with
inspectable deterministic components; the system decides") — largely on the
**post-LLM** side, and more maturely than the article:

- **Semantic grouping**: `entity_filtering/deduplication/semantic_blocker.py`
  (UMAP + HDBSCAN), `resolution/contextual_clusterer.py` (co-occurrence
  Union-Find), `resolution/incremental_resolver.py` (centroid clusters).
- **Alignment ("does it exist?")**: `resolution/kg_resolver.py` 3-tier cascade
  (alias→fuzzy(Levenshtein)→semantic(cosine)), enriched with `kg_match_type` +
  `kg_similarity_score` + `is_new` (= the article's NOVEL).
- **Deterministic validation / label filters**: `filters/noise_filter.py`
  (citations, single chars, pure numbers, URLs, emails), `filters/reclassifier.py`,
  `validation/ontology_constraint_filter.py`, plus the **13-stage** filtering
  workflow (`entity_filtering/workflow.py`) and the Track F audit's 6 LLM-free
  checks (`audit_service.py`).
- **Provenance**: `ExtractedEntity` carries `source_chunk_id` + `source_grounding`;
  `citation_completeness` is an enforced audit invariant; cites/mentions
  materialization (Tracks U/X/Y).
- **Ontology hierarchy + evolution**: `ontology_manager/canonical_bridge.py`
  walks a `parent_type` chain to schema.org base types (→ `type_tags`);
  `ontology_manager/evolution.py` already proposes new types from unmapped-entity
  frequency (the article's "should this become a concept?" — at the TYPE level).

**The genuine gap is the PRE-LLM side.** Extraction (`pass2_typed_extraction.py`)
sends Track-M context-packed chunk *text* straight to the LLM for direct
entity+relation extraction. There is **no cheap candidate/evidence layer** before
it (no TF-IDF salience, noun-phrase/NER candidates, or Hearst is-a patterns), and
**no first-class abstention** — the LLM extracts, then over-generation is cleaned
up post-hoc by the 13-stage filter. The article's biggest lesson (selection >
generation; the model over-produces) argues for moving some of that governance
*upstream*.

## 2. Design decisions

- **D1 — Incremental (pre-LLM candidates), NOT a pipeline inversion.** Keep the
  per-chunk extraction + the 13-stage post-filter. ADD a pre-LLM candidate/evidence
  pass that (a) distils each chunk into salient terms + noun-phrase/NER candidates +
  deterministic is-a candidates, and (b) threads them into the Pass-2 prompt as a
  "candidate anchor" list + seeds high-precision is_a relations directly. The full
  "cluster evidence into packets, ask the LLM per-packet" inversion (article §"Don't
  Send Everything to the LLM") is DEFERRED (§5) — it re-architects the extraction
  loop and trades granularity/provenance for call-count.
- **D2 — spaCy noun-chunks first-class; domain-gazetteer NER stub (user decision
  2026-08-31).** Add spaCy (+ `en_core_web_sm`, ~12MB) as a first-class dependency:
  its dependency-parse **noun-chunk** extraction is the core, domain-agnostic
  candidate source (default ON), alongside a TF-IDF salience pass over the source's
  own chunks and regex Hearst patterns (N.2). **NER ships as a wired-but-OFF stub
  built on spaCy's RULE-BASED `EntityRuler`** (a `domain_patterns` config hook) —
  NOT the general-domain statistical NER, which under-fits domain concepts. The
  EntityRuler injects domain term lists deterministically (gazetteer), so it is the
  right home for domain-specific entities. **Populating that domain gazetteer** is a
  deferred follow-up (§5) — the stub is scaffolded now so the wiring exists.
- **D3 — Evidence is transient, no migration.** Candidate/evidence records are
  computed per-chunk at extraction time and threaded into the prompt / seeded as
  relations; they are NOT a new persisted table in N.1–N.3 (rides the existing
  chunk + extraction flow). A persisted `evidence` store (for the full article
  traceability chain) is part of the deferred inversion, not this track.
- **D4 — Abstention = confidence + explicit "not-a-concept" gate; LLM-judge is the
  DEFAULT arbiter (user decision 2026-08-31).** Pass-2 already has per-element
  `confidence`. N.3 adds an explicit `INSUFFICIENT_EVIDENCE` skip path + a
  "real domain concept vs UI/table/field artifact" classifier: a cheap
  DETERMINISTIC pre-pass (a stricter cousin of `noise_filter`) fast-paths the
  obvious accept/reject cases, and the **LLM-judge — reusing the `llm_matcher` /
  `contradiction_judge` pattern — is the DEFAULT arbiter for the ambiguous middle**
  (not gated off). Same for N.4's RELATED/NOVEL split. A **regression metric**
  (over-generation rate: extracted vs survived-the-filter) keeps the win
  falsifiable — the M.5 pattern.

## 3. Phases (each ONE PR, Backend-first)

| Phase | Title | Effort | Depends |
|---|---|---|---|
| **N.1** | Pre-LLM candidate layer (TF-IDF salience + noun-phrase candidates) threaded into the Pass-2 prompt | 2–3d | — |
| **N.2** ✅ | Deterministic Hearst is-a miner (spaCy noun-chunk based) → high-precision `is_a` relation seeds | 1.5–2d | N.1 |
| **N.3** ✅ | Abstention gate: `INSUFFICIENT_EVIDENCE` + not-a-concept pre-filter (determin. + LLM-judge) + over-generation metric | 2–2.5d | — |
| **N.4** 🔁 | Concept-level alignment taxonomy (BROADER/NARROWER/RELATED/NOVEL). **Re-planned v2** after attempt 1 was parked: **N.4a** ontology-grounded subsumption (verdicts only) → **N.4b** placement + surviving seeds → **N.4c** evolution gap loop + reachability | 2.5–3d | N.1–N.3 |
| **N.5** | Integration: regression gate (call-count / recall / over-generation) + docs + RETRO | 1.5–2d | N.1–N.4 |

### N.1 — Pre-LLM candidate layer
- **New** `pipelines/ontology-extraction/src/ontology_extraction/candidates.py`:
  `extract_candidates(chunk_text, corpus_stats) -> list[Candidate]` — combining
  (1) **TF-IDF salience** over the source's own chunks (cheap, dependency-free),
  (2) **spaCy `en_core_web_sm` noun-chunk extraction** (default ON — the core
  linguistic candidate source), and (3) an **`EntityRuler` NER stub**: the spaCy
  pipeline is wired with a rule-based EntityRuler reading a `domain_patterns`
  gazetteer (empty for now, `EXTRACTION_DOMAIN_NER_ENABLED` default OFF) so a
  future domain term list plugs straight in (D2). spaCy loading is lazy + guarded
  (a missing model degrades to TF-IDF + a regex noun-phrase fallback, never
  crashes extraction).
- **Modify** `prompts/pass2.py` + `pass2_typed_extraction.py`: thread the top-K
  candidates into the user prompt as a "candidate anchors (extract these if
  present, ignore if spurious)" block — a precision nudge, NOT a hard constraint
  (the LLM may still find more, and abstain on a spurious anchor).
- **AC**: candidates are deterministic + reproducible; the prompt carries them;
  extraction still runs with candidates=∅ (graceful); a token-budget guard caps
  the anchor block (reuse the Track-M budget math).
- **Tests**: `test_candidates.py` (TF-IDF ranking, noun-phrase heuristic, empty
  input); a prompt-assembly test that the anchor block is present + budget-capped.

### N.2 — Hearst is-a miner ✅ SHIPPED (merge 2d78953)
- `candidates.py::mine_hearst_isa(text, *, nlp=None, lang=None) -> list[(narrow,
  broad)]`: **spaCy noun-chunk based, NOT regex.** A regex draft grabbed
  clause/verb garbage on live prose and needed a `_clean_list_item` band-aid the
  user rejected as a monkey-patch — so the miner now finds the cue phrase
  ("such as"/"zoals"/"including"/"waaronder"/… broad-first; "and other"/"en
  andere" broad-last, EN+NL), takes the noun-chunk before/after the cue as the
  hypernym, and collects the following/preceding noun-chunks as the hyponym list.
  Both the anchor and the list are bounded to the cue's OWN sentence
  (`_sentence_span` via `doc.sents`) and by POS gap discipline (`_gap_is_clean`:
  a VERB/ADP between chunks ends the span) so it never crosses a clause. No spaCy
  / model → `[]` (no garbage regex fallback).
- `pass2_typed_extraction.py::_seed_hearst_relations` seeds mined pairs as `is_a`
  relations ONLY between entities the LLM already extracted for the chunk
  (precision gate, exact normalized-lowercase on both endpoints), tagged
  `relation_source="hearst"` + conf 0.5, so `ontology_constraint_filter`/dedup
  still govern them. dedup key includes `relation_type`. `run_pass2` detects the
  corpus language once + loads the spaCy model once, shared by N.1 candidates and
  N.2 mining.
- **AC** (met): precision-first (no cue → no relation); provenance + chunk id on
  every seed; LLM path unchanged (additive); degrades to `[]` without spaCy.
- **Tests**: `test_hearst.py` — token-level spaCy STUB for the boundary/POS edge
  cases (verb-gap stop, other-drop, dedup, edge-strip, no-cue, no-spacy,
  cross-sentence both directions, within-sentence verb-block, comma-still-anchors)
  + gated real-model regressions. Adversarial-review APPROVED after the
  anchor-sentence-bounding MAJOR fix. Live-validated on real EN/NL PDFs.

### N.3 — Abstention gate + over-generation metric ✅ SHIPPED (merge 6132110)
- `prompts/pass2.py`: an `INSUFFICIENT_EVIDENCE` abstention clause — a
  furniture-only chunk (nav, TOC line, header/footer, page number, caption
  reference, boilerplate) returns an empty entities array instead of manufacturing
  vague concepts. Scoped as the SINGLE exception to exhaustive recall (recall wins
  ties). Abstention is OBSERVED as a non-parse-error chunk with zero entities and
  counted (`abstained_chunks`) — no machine-readable reason token added to the
  JSON contract.
- `not_a_concept.py`: two-tier gate. `classify_deterministic(text, label)` →
  `True` (reject) / `False` (keep) / `None` (ambiguous). Precedence:
  structural + `_REJECT_ALWAYS` (UI/nav/boilerplate, any label) → SPECIFIC-label
  fast-accept → `_FIELD_WORDS` reject (generic label only — homographs like
  "Total"/"Page" kept when specifically typed) → proper-name accept → ambiguous.
  Plausible homographs (Next/Home/Index/…) defer to the judge, never hard-drop.
  The **batched LLM-judge (default ON, D4)** arbitrates the ambiguous middle in one
  call per chunk; no judge / judge failure / silent verdict → KEEP. Pure classify +
  judge prompt/parse (unit-tested without an LLM).
- `pass2_typed_extraction.py`: the gate runs per chunk BEFORE append + Hearst
  seeding, so only survivors reach the graph + precision gate. Relations
  referencing a REMOVED entity are dropped (stripped compare); a relation to a
  never-extracted endpoint passes through (pre-N.3 behaviour). Counts land in
  metadata. Env flags `EXTRACTION_NOT_A_CONCEPT` / `_JUDGE` (both ON).
- `apps/app-main/.../extraction_chunking/extraction_metrics.py` (mirrors M.5's
  `chunking_metrics`): pure `over_generation_rate = 1 − survivors/extracted` +
  `abstain_rate = abstained_chunks/total_chunks`, derived from the metadata counts.
- **AC** (met): abstain counted; not-a-concept filter deterministic +
  config-tunable (`extra_reject_exact`); a specifically-typed real entity is never
  dropped (review MAJOR fix). **NOTE**: the deterministic tier drops a REAL entity
  only when its whole surface form is a generic field word UNDER a generic label —
  the accepted precision/recall trade; the golden-corpus recall floor is the N.5
  gate's job (no golden fixtures exist yet).
- **Tests**: `test_not_a_concept.py` (13→ classify/partition/judge, incl.
  specific-vs-generic homograph + UI-homograph→judge), `test_abstention.py` (8→
  prompt clause + run_pass2 filter/judge/abstain counts), `test_extraction_metrics.py`
  (6→ pure rates). Adversarial-review APPROVED.

### N.4 — Concept-level alignment taxonomy — 🔁 RE-PLANNED v2 (attempt 1 PARKED)

> Attempt 1 (`feature/track-n4-concept-alignment` @ `da0bda6`) is **parked, not
> merged**. Adversarial review: **REVISIONS_NEEDED — 2 blockers, 5 majors**; full
> report in `reviews/phase-N.4-attempt-1.md`. The blockers are DESIGN errors, not
> code defects, so the chapter is rewritten rather than patched. The branch is kept
> for the parts that survived (see "Salvage" below); do NOT resume from it blindly.

**Why v1 failed** (the two design errors that drive this rewrite):

1. **Lexical containment was treated as subsumption.** "A contains B" was emitted
   as `NARROWER_THAN` at the highest confidence. On real Dutch names that is wrong
   more often than right — `Tweede Kamer der Staten-Generaal` ⊃ `Tweede Kamer` is
   *same_as*, `Den Haag Zuidwest` ⊃ `Den Haag` is *part_of*, `Van Gogh Museum` ⊃
   `Van Gogh` is *named_after*. And the input population is biased toward exactly
   that: `KGResolver`'s fuzzy tier rejects long/short ALIAS pairs (the length delta
   tanks Levenshtein), so aliases and meronyms are the aligner's main diet.
2. **The output was structurally unreachable.** Seeded `is_a` edges were emitted
   before the ontology filter, which drops any relation with an off-batch endpoint —
   and a seeded target is *by construction* an existing KG node. 100% of the output
   was silently discarded while the report still counted it. Compounding: the
   candidate fetch queried `find_by_type(entity["label"])`, but that repo method
   filters the canonical `entity_type` column while `label` is the rich Track-L
   extraction label — so under rich typing the fetch returns `[]` by construction.

**Planner correction**: the original chapter cited the `evolution` agent's gap
analysis and attempt 1 recorded that "no such agent exists". That was wrong —
`packages/ontology-manager/src/ontology_manager/evolution.py` holds a working
`OntologyEvolutionAgent` (`record_gap` → frequency threshold → `SchemaProposal` →
approve/reject/implement, plus `list_gaps` / `get_gap_statistics`), wired into
`ontology_manager/manager.py` but NOT into the filtering pipeline. v2 uses it.

#### Decisions (binding for v2)

| # | Decision | Rationale |
|---|---|---|
| **D-N4-1** | **Lexical containment is NOT subsumption evidence.** It may never seed an `is_a`. | It signals *alias / part-of / named-after* at least as often as *subtype* (review B1). |
| **D-N4-2** | The **only** deterministic subsumption source is the ontology's declared `parent_type` chain (`canonical_bridge`). Embeddings inform RELATED/NOVEL **only**. | Subsumption is an ontological claim; only the ontology can ground it. |
| **D-N4-3** | Candidate fetch queries the **canonical `entity_type`** (via `canonical_bridge`), never the raw rich label. | `find_by_type` filters the canonical column; passing a rich label returns `[]` (review M5a). |
| **D-N4-4** | The stage runs **after** ontology validation **and** after graph centrality. | Otherwise seeds are stripped (B2) or perturb PageRank and can cause entity REMOVAL (M4). |
| **D-N4-5** | A seeded edge carries `source_type`, `target_type` and the target record id. | Restores the Track-O.1 endpoint disambiguation the v1 seed threw away (M3). |
| **D-N4-6** | A `NOVEL` verdict **records an ontology gap** via `OntologyEvolutionAgent.record_gap`. | Closes the loop the chapter always intended: novel concept → gap → frequency → `SchemaProposal`. |
| **D-N4-7** | Evidence must be **falsifiable**: state what was OBSERVED, never the inference it would license. Each reason code names an observation with exactly **one** cause. As shipped in N.4a: `no_repo`, `empty_surface_form`, `no_resolvable_type`, `candidate_fetch_failed`, `type_query_returned_no_rows`, `entity_has_no_embedding`, `no_candidate_embeddings`, `vectors_incomparable`, `compared_none_close`, `classification_error`. | An evidence-first track cannot stamp confident falsehoods (attempt-1 M5; attempt-2 B1–B4). N.4c filters gap-recording on these codes, so they must never be a guess. |
| **D-N4-8** | Reachability is part of the phase, not a follow-up: an env flag **and** DI wiring in `entity_extraction_service`, with a WARNING on enabled-but-unwired (the orphan-connector pattern). | v1's judge was unreachable in every real run (M7). |
| **D-N4-10** | **`BROADER_THAN` must be reachable**, delivered in N.4c via inverse chain lookup and/or type-level alignment of `SchemaProposal`s — still ontological evidence only, never lexical or cosine. | N.4a leaves it unreachable (the chain only walks upward), which would silently reduce the taxonomy to three values. Full statement in the N.4c section. |
| **D-N4-12** | **Subsumption is decided where a TYPE enters the system** (extension proposal / evolution proposal / curator acceptance), never per entity — the entity table stores mentions, and no node ever represents a type. An accepted BROADER_THAN is applied as a SCHEMA re-parent, so descendants are inherited rather than enumerated. | Three attempts failed identically at instance level (N.4 parked, N.4a/b, N.4c). Full statement in the N.4d chapter. |
| **D-N4-11** ↩ | *Superseded by D-N4-12*: an accepted `BROADER_THAN` triggers a descendant sweep: every other concept already in the graph that falls under the new broader concept is attached too. Exhaustive-or-declared-partial, idempotent, reversible, and sharing one rule with the per-entity path. | BROADER_THAN is one-to-many; attaching only the triggering child leaves a half-built hierarchy whose shape depends on extraction order. Full statement in the N.4c section. |

**D-N4-9 (RESOLVED 2026-09-01, user)** — the lexical signal becomes an **alias
candidate**, not `is_a` and not a verdict. `lexical_alias_candidates` emits
direction-agnostic long-form/short-form pairs into the alignment report for
review; they never influence the verdict and are **never auto-registered** —
writing an alias merges two identities in the graph, which must be an explicit
decision, not a side effect of classification. This recovers real value: those
pairs are exactly what `KGResolver`'s fuzzy tier structurally misses (a large
length delta tanks Levenshtein). Auto-registration behind an explicit opt-in
remains possible later.

#### Sub-phases

**N.4a — Subsumption from the ontology (verdicts only, no seeding)** — ~1d
- **New** `resolution/concept_alignment.py`: the four-verdict taxonomy +
  `Alignment` evidence record (SALVAGE v1). Subsumption exclusively from the
  `parent_type` chain (D-N4-2); embedding band decides RELATED vs NOVEL; the
  batched LLM-judge arbitrates only the ambiguous band, with v1's fencing kept
  verbatim (may not answer subsumption; a target outside the concept's own
  neighbour list is downgraded to NOVEL; garbage/silence/failure → NOVEL).
- Candidate fetch on the canonical type (D-N4-3), `properties: None` guarded,
  per-batch candidate cache.
- **AC**: every verdict carries falsifiable evidence (D-N4-7); `method_counts` and
  `judged_count` are mutually consistent; no entity is mutated beyond `properties`;
  a rich-label corpus actually fetches candidates (regression against M5a).
- **Tests**: unit — all four verdicts; type-chain with and without a materialised
  target; embedding band edges; judge fencing; evidence-string honesty for the
  empty-fetch vs nothing-close paths.

**N.4b — Placement + seeding that survives the pipeline** — ~1d
- **Modify** `workflow.py`: run the stage after ontology validation AND centrality
  (D-N4-4); seed `is_a` with `source_type`/`target_type` + target id (D-N4-5); only
  for a NARROWER verdict with a materialised target (no dangling edges).
- **AC**: a seeded edge is present in `result.relations` with
  `ontology_validation.enabled=True`; centrality scores are byte-identical with the
  stage on vs off; `seeded_is_a` in the report equals what actually survives.
- **Tests**: **workflow-level integration** (mandatory — its absence hid both
  blockers): seeds survive validation, do not shift centrality, do not change which
  entities stage 12 removes, and `ExtractedRelation(**seed)` validates.

#### Carried-forward line items from the N.4a review (binding, not optional)

The N.4a approval listed six non-blocking minors, all landing on code N.4b/N.4c
will touch. They are line items, not "nice to have":

| # | Item | Owner |
|---|---|---|
| **C1** | `EV_NONE_CLOSE` currently has THREE causes (below-floor; judged-and-rejected; unadjudicated band), violating D-N4-7's own "exactly one cause" contract. The below-floor and *unadjudicated band* cases share both the code AND `method=none`, separable only by comparing `similarity` to a floor the consumer must know out of band. **Split out an unadjudicated-band code (or correct the definition) BEFORE N.4c filters gap-recording on it** — otherwise a concept nobody actually adjudicated is recorded as a confirmed ontology gap. | **N.4c** (its consumer) |
| **C2** | The LIMIT-cap disclosure (`_sample_note`) is appended in 2 of 5 negative paths — missing from `_probe_evidence` ("100 rows fetched, none had an embedding" has the same "may not have seen the closer ones" problem) and from all three judge-path evidences. | N.4b |
| **C3** | `type_chain_subsumption` builds its `Alignment` **without** `canonical_type` — and `NARROWER_THAN` is the one verdict N.4b actually seeds, so the field added for D-N4-5 is missing exactly where it is needed. Latent only while the tier is off. | N.4b (fix when D-N4-10 rebuilds the tier) |
| **C4** | `EV_NO_REPO` drops `canonical_type` although `resolve_types` already ran successfully. | N.4b |
| **C5** | `assert nearest is not None` is a production-path control guard that vanishes under `python -O`. Invariant is sound; convert to an explicit branch. | N.4b |
| **C6** | ~~Plan drift: the D-N4-7 row named retired constants~~ — **fixed** in this commit. | done |

> **Standing consequence of D-N4-10, recorded so it is not discovered again:**
> under the shipped `ConceptAlignmentConfig()` defaults the pipeline seeds **zero**
> `is_a` edges. `type_chain_enabled` is OFF (it cannot verify a name-matched node's
> type), and it is the only producer of `NARROWER_THAN` — so N.4b's seeding path is
> exercised today only by explicitly opting into a tier the module documents as
> unverifiable. N.4c's verifiable subsumption is what turns seeding on for real.

### N.4d — Subsumption at the TYPE boundary (re-scoped after N.4c)

> **Why this chapter exists.** Three times now the subsumption premise has failed
> — lexical containment (parked attempt), name matching (N.4a/b), and the declared
> type (N.4c). Each failure had the same shape, and the N.4c review finally made it
> unmistakable: **subsumption is a relation between TYPES, and the entity table
> only stores MENTIONS.** Nothing in this codebase ever creates a node representing
> an ontology type — every `entity` row is written from a text mention by
> `EntityPersistenceService`. So "this entity is narrower than that entity" is not
> a claim the data can support, however it is dressed up. The answer is not a
> better heuristic in the same place; it is a different place.

#### D-N4-12 — subsumption is decided where a TYPE enters the system, not per entity

There are exactly three such moments, and **all three currently fill the parent
slot with an unvalidated guess**:

| # | Moment | Who sets the parent today |
|---|---|---|
| 1 | `pass1_schema_validation` proposes an extension | the extraction LLM guesses `parent_type` |
| 2 | `OntologyEvolutionAgent.create_proposal_from_gap` | `definition["parent_type"] = gap.entity_type_guess` — the gap's raw type guess |
| 3 | `SchemaEditService.accept_extension` | carries whichever guess arrived, unchecked, into `accepted_extensions` |

That unchecked guess IS the gap this work should fill. At this boundary both sides
of the question are types, so the question is well-posed for the first time.

#### The deterministic constraint that makes BROADER_THAN tractable

For a proposed type `P` with (guessed or known) parent `G`:

* **NARROWER_THAN is a declaration, not an inference.** If `G` resolves to an
  existing type, `P is_a G` is simply what the proposal says. The job is to
  VALIDATE it (does `G` exist? is the chain acyclic? does `P` duplicate an
  existing type or alias?), not to derive it.
* **BROADER_THAN has a bounded, sound candidate set: `P`'s SIBLINGS.** `P` can
  only become the parent of an existing type `T` by being inserted between `T` and
  `T`'s current parent. That is structurally valid only when `T` currently hangs
  from `G` too — i.e. `T` is a sibling of `P`. Any `T` whose chain already passes
  through something narrower than `P` cannot be re-parented under it.

  This is what was missing at instance level: a bounded candidate set derived from
  declarations rather than guessed from names or vectors. The LLM-judge then
  decides WHICH siblings actually fall under `P`, over a handful of type
  definitions rather than a graph-sized set of mentions.

#### D-N4-11 dissolves: express subsumption in the SCHEMA, not as N edges

An accepted `BROADER_THAN` re-parents the affected types: `T.parent_type := P`.
Because `canonical_bridge` walks the declared chain, **every existing and future
entity of type `T` inherits `P` automatically.** The descendant sweep D-N4-11
demanded — find every other concept beneath the new broader concept and attach it
— is therefore not implemented at all; it is obtained. One schema edit does what
N per-entity `is_a` edges would have done, stays correct for entities ingested
afterwards, and is reversible as a single op with an event trail.

This also retires the instance-level seeding question: there is nothing to seed.

#### Sub-phases

**N.4d.0 — Retire the instance-level subsumption tier** — ~0.5d
The tier is already ON MAIN (shipped in N.4a/N.4b with `type_chain_enabled=False`),
so D-N4-12 leaves dead machinery in place unless it is removed. Remove
`type_chain_subsumption`, the `NARROWER_THAN`/`BROADER_THAN` producers, and — since
`NARROWER_THAN` was its only trigger — `build_is_a_seeds`, the `seed_is_a` config
and Stage 15's relation output. `concept_alignment` keeps `RELATED_TO`/`NOVEL`,
the alias candidates, and the report, which is what N.4d.4's gap loop consumes.

- **AC**: the alignment stage emits no relations at all; `FilteredResult.relations`
  is byte-identical with the stage on and off.
- **AC CORRECTED after review**: the original AC also claimed "the N.4b placement
  guarantees stay asserted … so a future producer cannot silently reintroduce the
  two blockers that phase fixed". The reviewer **disproved that by mutation**: a
  producer emitting the shape that actually mattered — an edge into an existing,
  **off-batch** graph node, which is exactly what the retired tier seeded — placed
  before Stages 11 and 12 leaves all thirteen workflow tests green. The tests only
  fire for ON-batch endpoints, i.e. they catch the safe reintroduction and miss the
  dangerous one. With no producer left there is nothing to place, so the claim was
  DROPPED rather than propped up with an artificial guard. **Binding for whoever
  reintroduces a producer**: re-establish the guarantee with a test whose target is
  off-batch, because both blockers are still real.
- **Removal, not deprecation**, deliberately: with D-N4-12 there is no story in
  which the tier becomes correct, and a disabled-but-present tier invites a fourth
  attempt. If a future path ever marks concept nodes explicitly (the vault-sync
  importer creates `entity` rows from notes and could flag a concept page), the
  identification problem this tier could never solve would be solved *there* — and
  the tier would be rebuilt against that flag, not resurrected.

**The N.4c branch is PARKED, not partially merged.** Its one reusable piece — the
widened `find_by_type` projection returning `primary_type`/`type_tags` — existed
solely to make the instance tier verifiable. With the tier retired it has no
consumer, and the review noted it has no test and has never been executed against
a live DB. Shipping an untested widening of a shared package for a caller that no
longer exists is dead weight; it can be recovered from `68c544f` if a real consumer
appears.

> **N.4d.0 SHIPPED** (merge `d347474`). The tier is gone: ~400 net lines removed,
> the taxonomy here is two-valued, and a test asserts the retired symbols are
> absent so a fifth attempt has to change that test rather than slip past it.
> Review APPROVED at attempt 2 — the removal was mechanically clean throughout;
> every finding was a claim the commit made about itself, including a placement
> guarantee the reviewer **disproved by mutation** and which was withdrawn rather
> than defended. Report: `reviews/phase-N.4d0-attempts-1-2.md`.

**N.4d.1 — Type placement, verdicts only** — ~1d
- **New** `packages/ontology-manager/.../type_placement.py`: pure functions over
  `(proposed type, applied ontologies)` → a placement with evidence.
  Deterministic layer: name/alias collision against existing types (a "new" type
  that already exists is a DUPLICATE, not a placement); parent existence; cycle
  detection; sibling enumeration via the declared chains.
- **AC**: every verdict carries falsifiable evidence with a reason code, following
  the N.4a discipline verbatim — state what was observed, never the inference it
  would license. A proposal whose parent does not resolve reports exactly that,
  and never "it is top-level".
- **Tests**: against the REAL shipped ontologies, not a stub of the bridge — the
  N.4a M2 lesson.

> **N.4d.1 SHIPPED** (merge `ab526fc`). `type_placement.py` validates a proposal's
> DECLARED parent and enumerates the bounded candidate set. APPROVED at attempt 3;
> the two intermediate rejections were both fixes that introduced a new instance of
> the track's failure mode, because the defect was measured on the real vocabulary
> and the fix was not. The guard that closed it asserts a safety property over all
> eleven shipped ontologies (262 candidates, zero violations) rather than pinning a
> case. Report: `reviews/phase-N.4d1-attempts-1-3.md`.
>
> Two facts N.4d.2 inherits: a parent is valid either as a defined type OR as a
> mapped schema.org base, and `roots_at` gives an explicit `parent_type` precedence
> over `schema_org_type` (the opposite of the bridge's canonicalisation order, for
> placement-specific reasons documented in the function).

**N.4d.2 — The judge over the sibling set** — ~0.5d
- The batched LLM-judge decides which of `P`'s siblings belong under `P`. Fenced
  as in N.4a: it may only choose from the sibling list it was given, a fabricated
  or borrowed name is refused, and silence means "leave where it is".
- **AC**: the judge can never widen the candidate set, only select within it.

> **N.4d.2 SHIPPED** (merge `3b02d76`). Pure prompt/parse; the judge selects
> within the bounded set and cannot widen it, and silence leaves a type where it
> is. APPROVED at attempt 3 — the three rejections were three shapes of one
> failure: a fence claimed but removed, a guard measuring the direction that was
> never at risk, and a guard asserted where it could not fail. Report:
> `reviews/phase-N.4d2-attempts-1-3.md`.
>
> **Binding for N.4d.3 and every later sweep**: assert in the shape production
> assembles. `detect_applicable_schemas(top_k=3)` means an applied set holds
> THREE ontologies; on a single-ontology load `entity_types` is a name-keyed dict
> and cross-ontology properties hold by the dict rather than by the mechanism.
> `candidates_from_ontologies` is exported for N.4d.3, which owns the LLM call.

**N.4d.3 — Apply as a schema edit** — ~1d
- **New** `SchemaEditService.reparent_type`, alongside the existing
  `rename/merge/split/delete` ops, with the same persist + event plumbing.
- Placement runs at `accept_extension` time and surfaces its verdict to the
  curator; the re-parent is applied on the curator's decision, never silently.
- **AC**: accepting a `BROADER_THAN` over N qualifying siblings re-parents all N;
  re-running changes nothing; an entity of a re-parented type resolves through the
  new ancestor via `canonical_bridge` with no per-entity write.

> **D-N4-13 — which vocabulary a placement is judged against (N.4d.3).**
> A placement is only meaningful relative to an applied set, and at
> `accept_extension` time the per-document set does not exist: `detect_applicable_schemas`
> scores a document, and no document is in hand. The first attempt used that to
> argue the whole AC bullet away; the review was right that this proves less than
> it claims. What DOES exist at acceptance is the notebook-level FORCED set —
> `base_ontology` + its affinity bundle + the schemas named on accepted extensions
> — which `_apply_notebook_schema_default` applies to every extraction in the
> notebook. That is the set `TypePlacementService` uses, and the report names it
> in `vocabulary`.
>
> **A placement CAN disagree with the verdict the runtime set would give**, and
> the second attempt's claim that it cannot was wrong twice over. Both were
> measured on the shipped vocabulary:
>
> 1. **Verdicts are not monotone in the applied set.** Adding a schema can turn
>    `PLACED` into `DUPLICATE`, which is the verdict N.4d.1 exists to produce.
>    Proposing `ScholarlyArticle` under `Deal` in a `deals` notebook reports
>    `PLACED` against `(deals, policy_themes)` and `DUPLICATE` once auto-detection
>    adds `scholarly`. So "everything in the forced set is in the runtime set"
>    licenses nothing: a superset premise only carries monotone conclusions, and
>    `PARENT_UNKNOWN` and missing siblings — the two consequences the second
>    attempt listed — happen to be the monotone ones.
> 2. **The forced set is not always a subset.** `_apply_notebook_schema_default`
>    is gated on a truthy `base_ontology`, and the Regio-Deal corpus's notebooks
>    have it empty today. There the runtime forces NOTHING, while the report still
>    composes a set from the schemas named on accepted extensions and places
>    against it. Guarded on BOTH sides, after a review measured that the runtime
>    half was first asserted in a form that could not fail: the gate itself in
>    `test_an_empty_base_ontology_forces_no_schema` (with
>    `test_a_configured_base_ontology_does_force_its_schemas` as its vacuity
>    guard), and the report in `test_an_empty_base_ontology_still_produces_a_report`.
>
> The placement is therefore **advisory**: it is a report a curator reads, it
> writes nothing, and the re-parent it may suggest is applied only by an explicit
> `POST /schema/reparent`. That is what makes the disagreement tolerable rather
> than a defect — and it is the reason the phase does not act on a placement
> automatically.

> **N.4d.3 SHIPPED** (merge `801f84ae`). `reparent_type` + `POST /schema/reparent`
> record the curator's decision; `ontology_manager.schema_projection` applies
> accepted edits to DEEP COPIES of the applied ontologies (the registry hands out
> shared objects); the bridge's parent walk is now symmetric with its own step 1;
> and `TypePlacementService` runs the placement and the N.4d.2 judge at accept
> time, reporting and writing nothing. APPROVED at attempt 4 after 3 blockers and
> 8 majors. Report: `reviews/phase-N.4d3-attempts-1-4.md`.
>
> **Binding for N.4d.4** — the gap loop gates on adjudication (C1), and
> `parse_judge_response` never returns None: a REFUSED reply carries an empty
> selection exactly like a judge that moved nothing. Read
> `JudgeSelection.decided` / `PlacementReport.judge_status`, never the emptiness
> of `selected`. Three of the four judge states carry an empty one.
>
> **Binding generally** — a guard that cannot fail is worse than no guard, because
> it reports as one. It occurred three times here: a sweep that killed zero
> mutants, a call site whose deletion was invisible, and the test written to stop
> this very decision text from drifting. Assert at the seam where the mechanism
> runs, and add the vacuity guard that makes a negative assertion mean something.

**N.4d.4 — Gap loop + reachability** (unchanged from the previous N.4c scope) — ~1d
- `record_gap` on a NOVEL verdict (D-N4-6), gated on the reason code — which is
  why **C1 must be closed first**: a concept nobody adjudicated must not be
  recorded as a confirmed gap.
- Env flag + DI wiring (D-N4-8), with the honest DEGRADED warning N.4b landed on.
- Note `record_gap` swallows its own exceptions and returns a gap with `id=None`
  on failure; treat a null id as "not recorded", never as success.

> **D-N4-14 — what a gap row is named after, and what recording one sets in
> motion (N.4d.4).**
> Gaps are keyed on `(entity_text, ontology_name)`, so the name decides whether a
> concept ACCUMULATES. The first attempt used `applicable_schemas[0]`, which
> `detect_applicable_schemas` ranks by per-document content overlap — so the same
> concept in two documents of one notebook landed in two rows at frequency 1
> instead of one at frequency 2, defeating the cross-document accumulation the
> `source_id` plumbing exists for. The name is now the notebook's **declared**
> `base_ontology`, falling back to the applied schema's own name only when a
> notebook has none configured.
>
> Relatedly, alignment receives **all** applied schemas, not the first:
> `detect_applicable_schemas(top_k=3)` means a type declared in the second or
> third would otherwise fail `resolve_canonical_type`, produce a code that
> licenses no gap, and make the loop under-fire for two thirds of the vocabulary.
>
> **What the flag switches on.** `OntologyEvolutionAgent` ships with
> `frequency_threshold=5` and `auto_propose=True`, so the fifth recording of one
> concept writes a `schema_proposal` row unasked, with `parent_type` taken from
> the rich extraction label. Two things bound what reaches a curator's queue: the
> reason-code gate (C1), and per-run de-duplication by concept, since the
> threshold counts DOCUMENTS.
>
> The de-duplication is **belt-and-braces under the shipped pipeline**, stated
> that way after a review measured it rather than as the live scenario an earlier
> draft claimed: Stage 4's `EntityDeduplicator._normalize_key` applies
> character-for-character the same normalisation eleven stages earlier, whenever
> `dedup_enabled` is set — which it is on both the app default config and the
> re-filter router. A duplicate reaches the gap loop only when the aligner is
> driven directly (the configuration both of its tests use) or dedup is off. It
> is kept because a gap is a claim about the graph and must not depend on an
> unrelated stage's configuration, and what it suppresses is counted
> (`gap_duplicates_suppressed`) because its key is the normalised form while
> `record_gap` matches `entity_text` exactly.
>
> **Two collaborators, one failure shape.** `record_gap` returns a gap with
> `id=None` on failure and `get_gap_statistics` returns a truthy
> `{"ontology_name": …, "error": …}` — both swallow their own exceptions and
> report through the RETURN. A caller watching only for a raise reports success.
> The first was handled from the start; the second was not, and reported
> `gap_statistics_status="ok"` with the error payload as the standing totals,
> persisted into `extraction_result.metadata`. Both are now read from the
> returned value — by MEMBERSHIP, not truthiness: `str(e)` is `""` for any
> exception raised without arguments, so a truthiness check still reported `ok`
> for a bare `TimeoutError`, which is what a slow gap store under load produces.
>
> **Binding, three rules, each of which caught something here.** A test double
> for a collaborator must reproduce the real method's failure RETURN, not a raise
> it never performs. At least one fixture must build the PRODUCTION argument set
> rather than a superset — a helper that always supplies an argument the app has
> stopped supplying makes the discriminating configuration unreachable. And when
> a guard reads a value from a collaborator, exercise it against the REAL
> collaborator at least once: both blockers in this sub-phase were found that way
> and by no other means, the second only by sweeping several exception types.
>
> **Where the loop is inert, stated so it is not rediscovered**: the
> single-schema path and `run_filtering_only` detect no schemas, so no canonical
> type resolves and no verdict is gap-licensing. Both wire the same collaborators
> anyway, so enabling the stage there degrades nothing silently.

> **N.4d.4 SHIPPED** (merge `948416bf`). C1 is closed — `EV_NONE_CLOSE` split
> three ways — and the gap loop gates on the reason code, so only a NOVEL verdict
> that ESTABLISHED something can become an ontology gap. `ENABLE_CONCEPT_ALIGNMENT`
> plus DI for all four collaborators makes the stage reachable at all (D-N4-8).
> APPROVED at attempt 5 after 3 blockers and 7 majors. Report:
> `reviews/phase-N.4d4-attempts-1-5.md`.
>
> **Binding for N.5**, from D-N4-14 and paid for three times over: a test double
> must reproduce the real method's failure RETURN, not a raise it never performs;
> at least one fixture must build the PRODUCTION argument set rather than a
> superset; and a guard that reads a collaborator's value must be exercised
> against the REAL collaborator at least once. Both blockers in this sub-phase
> were findable by no other means.
>
> **Operational note**: the entity-filtering suite runs against a REACHABLE live
> SurrealDB, so an unpatched repository call in a test silently exercises the
> real database and can pass for the wrong reason.

#### What happens to the entity-level tier

`concept_alignment` keeps `RELATED_TO` / `NOVEL` — which is where its value always
was, and what feeds the gap loop. The subsumption tier is **removed**, not merely
disabled: with D-N4-12 there is no longer a story in which it becomes correct, and
leaving dead machinery behind invites a fourth attempt at it.

#### Residuals carried from the N.4b review (binding)

| # | Item |
|---|---|
| **R2** | **Cross-pass duplicate `is_a`.** N.4b de-duplicates within its own pass only. An `is_a` on the same ordered pair already contributed by N.2's Hearst miner is not suppressed; at persist the two collapse on `(in, out, relation_type)` and the later write re-tags `relation_source`, weakening the "one `WHERE relation_source = …` drops the pass" reversibility claim. Fix alongside the sweep, which multiplies edges and so multiplies this. |
| **R3** | C2's two ruled-judge evidence branches and `EV_INCOMPARABLE_VECTORS` are unasserted (all consume the identical `sampled` element — negligible). |
| **R4** | `float(props.get("alignment_confidence") or 0.0)` raises on a non-numeric value and forwards an out-of-range one to Pydantic. Unreachable from `_enrich`; recorded so it is not rediscovered. |
| **R5** | Under the SHIPPED defaults **zero** edges are seeded — `type_chain_enabled` is off and is the only `NARROWER_THAN` producer. D-N4-10's verifiable subsumption is what turns seeding on for real, so N.4c is where this phase starts producing anything by default. |

**N.4c — Gap loop + reachability + BROADER_THAN + descendant sweep** — ~1.5–2d
- **Wire** `OntologyEvolutionAgent.record_gap` on every NOVEL verdict (D-N4-6),
  carrying the entity text, the rich label as `entity_type_guess`, and the chunk
  context; surface `get_gap_statistics` in the alignment report.
- **Make `BROADER_THAN` reachable (D-N4-10)** — see below. N.4a leaves it
  unreachable as an honest consequence of D-N4-2 (the declared `parent_type` chain
  only ever walks UPWARD, so it can say "novel is narrower than an ancestor" but
  never "novel is broader than an existing concept"). N.4c must close that, so the
  taxonomy is four-valued in practice and not three.
- **Wire** the env flag + DI (`entity_repo`, `ontology`, `alignment_llm_caller`) in
  `entity_extraction_service`, with a WARNING when enabled but unwired (D-N4-8).
- **AC**: a NOVEL concept produces a gap row; N occurrences cross the threshold and
  produce a `SchemaProposal`; enabling the stage without a repo logs a WARNING and
  is a no-op; the judge is exercised in a real run; **a concept that generalises
  existing graph concepts is classified `BROADER_THAN` with ontological evidence,
  never from a similarity or string signal**.
- **Tests**: gap recorded once per novel concept; proposal appears at the threshold;
  unwired-but-enabled warns and no-ops; an inverse-chain case yields BROADER_THAN
  and its mirror yields NARROWER_THAN (the two must not both fire for one pair).

> **D-N4-10 — `BROADER_THAN` must be reachable (user, 2026-09-01).**
> It stays bound by D-N4-2: the evidence must be ontological, never lexical or
> cosine-based. Two sound routes, to be chosen/combined during N.4c:
>
> 1. **Inverse chain lookup** — the mirror of the N.4a tier. For each entity type
>    `T` in the applied ontology whose declared ancestor chain CONTAINS the novel
>    concept's type/label `L`, every existing entity of type `T` is narrower than
>    `L`, hence `L` is `BROADER_THAN` them. Pure ontology reasoning plus a fetch on
>    `T`'s canonical type.
>    *Enabling constraint*: `find_by_type` projects `id, name, embedding, weight`
>    and **no type column**, so the direction cannot be recovered from a candidate
>    row — the lookup must be driven from the ontology side (enumerate the types
>    whose chain contains `L`, then fetch per type), or the repo projection must be
>    widened. Decide this explicitly; do not infer a candidate's type from its name.
> 2. **Type-level alignment of `SchemaProposal`s** — the evolution agent's natural
>    unit. A proposed NEW TYPE can genuinely become the parent of existing types,
>    which is exactly `BROADER_THAN`. This is the case where the verdict is most
>    informative, and it feeds the proposal's own `parent_type` field back.
>
> Note the framing this exposes: subsumption is a TYPE-level notion. For an
> ordinary novel INSTANCE, `NARROWER/BROADER` will remain rare and `RELATED/NOVEL`
> will carry the phase's value — which is why the gap loop (D-N4-6) is the part
> that pays off for instances, and BROADER_THAN pays off for proposed types.

> **D-N4-11 — an accepted `BROADER_THAN` triggers a descendant SWEEP (user,
> 2026-09-01).**
> `BROADER_THAN` is inherently **one-to-many**, unlike the other three verdicts.
> The existing concept that triggered the discovery is just the *first* child
> found; the same new broader concept very likely subsumes others already in the
> graph. Emitting only the triggering edge leaves a half-built hierarchy — one
> sibling gets an `is_a`, its equals do not — which is both wrong and unstable
> (the outcome would depend on which entity happened to be extracted first).
>
> So: **once a `BROADER_THAN` is ACCEPTED, the pipeline must sweep for every other
> concept that falls under the new broader concept and attach them too.**
> Requirements for that sweep:
>
> - **Trigger = acceptance, not classification.** A verdict is a proposal; the
>   sweep is a graph-wide write, so it hangs off the acceptance step (an operator,
>   or `OntologyEvolutionAgent.approve_proposal` when the broader concept arrives
>   as a `SchemaProposal`), never off per-batch classification.
> - **Same evidence rule (D-N4-2).** The sweep reuses the inverse chain lookup that
>   established the verdict — enumerate every ontology type whose declared chain
>   contains the new concept, then fetch those types' entities — simply without
>   stopping at the first hit. No lexical or cosine expansion: a sweep multiplies
>   whatever error the evidence rule allows.
> - **Completeness must be honest.** The M4 sampling problem bites hardest here: a
>   sweep that sees only a `LIMIT`-capped, unordered page builds a partial
>   hierarchy and then looks complete. The sweep must paginate to exhaustion, or
>   mark its result explicitly partial and re-runnable.
> - **Idempotent + reversible.** Re-running must not duplicate edges; every edge
>   carries the same `relation_source` provenance so the whole sweep can be
>   dropped as one unit.
> - **Also applies on later arrivals.** A concept extracted *after* the sweep that
>   falls under the broader concept must still be attached — i.e. the normal
>   per-entity NARROWER path must find it, so the sweep and the per-entity tier
>   have to agree on the same rule rather than being two implementations.
>
> **AC additions**: accepting a BROADER_THAN over a graph holding N qualifying
> concepts attaches all N (not just the trigger); re-running the sweep changes
> nothing; a capped sweep reports itself partial; a qualifying concept extracted
> after the sweep is still attached by the per-entity path.

#### Salvage from attempt 1 (`da0bda6`)
Carry forward, do not rebuild: the four-verdict taxonomy + `Alignment` record; the
judge fencing (`parse_judge_response` rejecting subsumption verdicts and fabricated
targets, sync/async caller handling); the token-boundary matcher
`_is_token_subsequence` (correct — only the inference from it was wrong);
`find_by_type` as the ONLY repo method (no new repo surface, no migration); the
three plan-mandated behavioural tests. Also carry the `ConceptAlignmentConfig` and
the `FilteredResult.concept_alignment_report` field (both backward compatible and
already regression-verified against ontology-extraction's 316 tests).

### N.5 — Close Track N's own debts + regression gate + docs

> **RE-PLANNED 2026-09-02** after the pipeline review
> (`claudedocs/extraction-pipeline-review.md`). The review measured a live corpus
> of eight documents and found problems that are real but mostly **not Track N's**:
> they span schema review (Track B), entity resolution (K), typing (L), and the
> default filtering configuration. Those move to **Track PC — pipeline coherence**.
> N.5 keeps only what this track genuinely owes, so it can close.
>
> The one finding that changes N's own footing: **the accept-extension queue has
> no writer**, so N.4d.1–N.4d.3 (placement, judge, re-parent) cannot be reached
> from a real run. That is PC.1, and until it lands the N.4d work is shipped but
> unreachable. N.5 does not wait for it — the debts below are independent.

**N.5a — N.3's observability survives the production path** (review R3)
`_merge_results` builds a fresh `ExtractionResult`, so `not_a_concept_removed`,
`not_a_concept_judged` and `abstained_chunks` are discarded on the multi-schema
path — which is the path production uses. Measured cost: `Bennett_test.pdf`
produced ten chunks and zero entities, and its stored record cannot say whether
the model found nothing or the gate removed everything.
- **AC**: the three counters survive the merge, summed across passes, and reach
  `extraction_result.metadata`. A document that yields nothing is explainable from
  its own record.
- **Guard**: assert on a MERGED result, not a single-pass one — the single-schema
  path already keeps them, so a test there cannot fail.

**N.5b — Decide what `is_a` is now** (review I3)
N.2 mines `is_a` from text; D-N4-12 moved subsumption to the type boundary and
retired the instance tier. Both ship. `is_a` is declared in no ontology, so the
mined edges survive only because `OntologyValidator` downgrades an unknown
predicate to a WARNING outside strict mode and stage 11 is off by default. Turning
on `strict_mode` deletes every mined hierarchy edge, silently.
- **Decision required, not a fix**: either declare `is_a` in the shipped
  ontologies (it becomes a first-class predicate that survives strict mode), or
  retire the Hearst miner as D-N4-12 retired its instance-level sibling. Shipping
  a producer whose output survives by accident is the option to rule out.
- **AC**: whichever is chosen, a test fails if `strict_mode` changes the outcome
  unnoticed.

**N.5c — The carried residuals** — R2 (cross-pass duplicate `is_a`), C2 (the
LIMIT-cap disclosure missing from three evidence paths), C3/C4 (`canonical_type`
dropped on two alignment paths), C5 (an `assert` used as a production control
guard). All small; all on code this track owns.

**N.5d — The regression gate + docs**
The original scope: a heterogeneous gate on a golden corpus asserting the pre-LLM
layer + abstention cut LLM calls and/or over-generation WITHOUT dropping golden
entities (recall floor). Update `ARCHITECTURE.md`, `status.md`, `RETRO.md`.
- **Now measurable**: `scripts/n_pipeline_review_run.py` is the harness the review
  used and the corpus JSONs are its baseline (124 entities over 70 chunks, in 8
  records covering 7 distinct PDFs — one was run twice). The gate compares against
  those, so "did we regress" has a number instead of an opinion.
- **Correction (review, attempt 1)**: "now measurable" was true of recall and NOT
  of the two cost dimensions. `run_extraction` returned only entity/relation
  counts plus filtering stats, so N.5a's counters — freshly rescued from the merge
  — never crossed out of the service, and the gate read a key nothing wrote. Half
  the gate could not fail while three documents said re-measuring would fill it.
  The seam is closed in `_observability_counters`; the lesson is that a producer
  and a consumer can each be correct while the thing between them was never
  built.
- **Binding** (D-N4-14): a double reproduces the real method's failure RETURN; a
  fixture builds the PRODUCTION argument set; a guard that reads a collaborator's
  value is exercised against the real collaborator at least once.

> **N.5 SHIPPED 2026-09-03** (review-approved attempt 3, merged `a4bb9d1b`;
> report in `reviews/phase-N.5-attempts-1-3.md`) — `e8ae249c` (N.5a), `2942cb5b` (N.5b),
> `8d9937fc` (N.5c), `47f424dd` (N.5d). Status and retrospective in
> `status.md` / `RETRO.md`.
>
> Three of N.5c's five residuals were already closed by earlier phases and a
> fourth (C3) became moot when D-N4-12 deleted the tier carrying it — verified in
> the code, not assumed from the plan. The two live findings had the same shape as
> each other: **a measurement that reads as a statement.** N.5a's merge reported
> `over_generation_rate` 0.00 for a run that culled 14 entities to 5, and N.5b's
> miner shipped enabled while contributing zero edges to the graph.
>
> N.5b was a user decision on measured evidence (220 raw pairs, 0 graph edges, 15
> survivors of mixed quality): declare `is_a` in the root ontologies AND ship the
> miner explicitly off. Neither half works alone.
>
> **The gate's central rule** (N.5d): a dimension with no baseline value is
> SKIPPED, never PASSED, and an all-skipped comparison is inconclusive rather than
> green. The two cost dimensions are exactly the ones the baseline cannot contain,
> because the merge was discarding their inputs when it was measured — so a gate
> that read "no baseline" as "not worse" could not have failed on anything this
> track added.

**Out of N.5, moved to Track PC**: the curator-queue writer, cross-document
identity, canonicalisation stability, the alias-policy contradiction, the gap/
proposal read path, and default-configuration coherence.

## 4. Dependencies, test & migration strategy
- **New dependency**: `spacy` + `en_core_web_sm` (D2), added to the
  `ontology-extraction` pipeline. Lazy-loaded + guarded (missing model → TF-IDF +
  regex noun-phrase fallback), so it never hard-fails extraction. Check the model
  is fetched in the container build / CI.
- Pure-function-first: candidates (TF-IDF + noun-chunks), Hearst, the deterministic
  not-a-concept pre-pass, and the metrics are all deterministic → fast unit tests
  (the M.5 pattern). The **LLM-judge is mocked at its seam** (like `llm_matcher`
  tests) so N.3/N.4 stay fast + deterministic; one `@requires_docker` integration
  test in N.5 drives the real extraction seam.
- **No migration** for N.1–N.3 (transient evidence, D3). N.4 may stamp a
  `concept_alignment` tag on existing FLEXIBLE metadata — no new table.
- Every phase gated by the standard **adversarial-review-as-merge-gate**.

## 5. Deferred (decisions resolved 2026-08-31)
- **Full evidence-packet inversion** (article §"Don't Send Everything") — DEFERRED
  with a measurement trigger: cluster evidence into packets + ask the LLM per-packet
  instead of per-chunk. A real re-architecture of the extraction loop that trades
  per-chunk provenance/granularity for fewer calls. Revisit only if the N.3/N.5
  metrics show per-chunk + candidates leaves a large call-count/over-generation win
  on the table.
- **Populate the spaCy `EntityRuler` domain gazetteer** — the NER stub ships wired
  but empty in N.1 (D2); loading real domain term lists (products, standards,
  domain-specific entities) is a follow-up the user wants "at some point". A small
  standalone PR once the wiring lands: a `domain_patterns` file/table + the
  `EXTRACTION_DOMAIN_NER_ENABLED` flip.

RESOLVED this round (no longer open): spaCy = first-class noun-chunks + EntityRuler
NER stub; LLM-judge = default ON for the semantic arbitration; packet-inversion =
deferred-measure-first.

## 6. Effort
~9.5–12.5 days across N.1–N.5 (N.4 is the harder phase). N.1+N.3 alone (~4.5–5.5d)
deliver the article's two headline wins and are independently shippable.
