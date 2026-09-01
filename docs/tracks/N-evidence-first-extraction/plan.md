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
> **▶ RESUME AT N.4 — Concept-level alignment taxonomy** (BROADER/NARROWER/
> RELATED/NOVEL over `kg_resolver` + `evolution`; the harder phase). Then N.5
> (regression gate + docs; also lands the N.2 + N.3 follow-ups above). The
> **evidence-packet clustering stays DEFERRED — measure first** (user decision, §5).
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
| **N.4** | Concept-level alignment taxonomy (BROADER/NARROWER/RELATED/NOVEL) over `kg_resolver` + `evolution` | 2.5–3d | N.1–N.3 |
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

### N.4 — Concept-level alignment taxonomy (harder)
- **Extend** `kg_resolver` beyond exists/new to classify a NOVEL concept relative
  to the existing graph: `BROADER_THAN` / `NARROWER_THAN` / `RELATED_TO` / `NOVEL`,
  using the existing embedding + `parent_type` chain (`canonical_bridge`) + the
  `evolution` agent's gap analysis. A NARROWER novel concept is attached under its
  broader existing type (an is_a edge) instead of floating.
- **AC**: subsumption is derived deterministically where possible (via `parent_type`
  + embedding neighbourhood) as a cheap pre-pass; the **LLM-judge (default ON, D4)**
  arbitrates the ambiguous RELATED/NOVEL split; every classification carries its
  evidence + is reversible.
- **Tests**: `test_concept_alignment.py` — a novel narrower concept lands under the
  right broader type; a true-novel one stays NOVEL; a related one links, not merges.

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
