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
> **N.4a IN REVIEW** on `feature/track-n4a-ontology-subsumption` — ontology-grounded
> verdicts + evidence, no seeding and no workflow stage. Attempt 1 of N.4a returned
> REVISIONS_NEEDED (4 blockers + 4 majors, all one class: evidence that reported an
> OBSERVATION as the inference it would license); revised in `27e5c64` — reason
> codes now name observations with exactly one cause each, judge items are keyed by
> index rather than surface form, and `resolve_types` is finally asserted against
> the REAL `canonical_bridge` instead of a stub of itself.
>
> **▶ THEN N.4b** (placement + seeding that survives ontology validation and
> centrality, with the mandatory workflow-level integration test) → **N.4c** (gap
> loop + `BROADER_THAN` reachability + the descendant sweep). All N.4 decisions are
> resolved: D-N4-9 (the lexical signal becomes an alias candidate), D-N4-10
> (BROADER_THAN must be reachable), D-N4-11 (an accepted BROADER_THAN sweeps for
> all its other descendants). Then N.5 (regression gate + docs; also lands the N.2 +
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
| **D-N4-7** | Evidence must be **falsifiable**: never assert "nothing comparable exists" when no candidates were *fetched*. Distinguish `no_candidates_fetched` from `candidates_fetched_none_close`. | An evidence-first track cannot stamp confident falsehoods (M5). |
| **D-N4-8** | Reachability is part of the phase, not a follow-up: an env flag **and** DI wiring in `entity_extraction_service`, with a WARNING on enabled-but-unwired (the orphan-connector pattern). | v1's judge was unreachable in every real run (M7). |
| **D-N4-10** | **`BROADER_THAN` must be reachable**, delivered in N.4c via inverse chain lookup and/or type-level alignment of `SchemaProposal`s — still ontological evidence only, never lexical or cosine. | N.4a leaves it unreachable (the chain only walks upward), which would silently reduce the taxonomy to three values. Full statement in the N.4c section. |
| **D-N4-11** | An **accepted `BROADER_THAN` triggers a descendant sweep**: every other concept already in the graph that falls under the new broader concept is attached too. Exhaustive-or-declared-partial, idempotent, reversible, and sharing one rule with the per-entity path. | BROADER_THAN is one-to-many; attaching only the triggering child leaves a half-built hierarchy whose shape depends on extraction order. Full statement in the N.4c section. |

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

### N.5 — Integration + regression gate + docs
- A heterogeneous regression gate (M.5-style): on a golden corpus, assert the
  pre-LLM layer + abstention **cut LLM extraction calls and/or over-generation**
  WITHOUT dropping golden entities/relations (recall floor). Update `ARCHITECTURE.md`
  extraction subsection; `status.md` + `RETRO.md`.

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
