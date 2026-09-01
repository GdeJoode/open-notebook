# Track N — Evidence-first extraction & abstention — Sprint Plan

> Status: 🔨 IN PROGRESS (2026-09-01). **N.1 fully SHIPPED** across PRs #78
> (candidate anchors: TF-IDF + noun-chunks + EntityRuler stub + budget-safe prompt
> threading), #79 (N.1-deps: `spacy` + `en_core_web_sm` first-class), #80
> (language-aware NL/EN model selection + complementary spaCy+regex merge +
> edge-word cleanup) — all review-APPROVED and live-validated on real Dutch (Regio
> Deal, lang→nl) AND English (academic paper, lang→en) PDFs. Model decision
> resolved: **NL + EN side-by-side, picked per document.**
>
> **▶ RESUME AT N.2 — Hearst is-a miner** (deterministic high-precision `is_a`
> relation seeds). Then N.3 (abstention gate — the English-paper test showed the
> candidate layer over-generates generic phrases on academic prose, so N.3's
> not-a-concept gate is the measured next win), N.4 (concept alignment), N.5
> (regression gate + docs). The **evidence-packet clustering stays DEFERRED —
> measure first** (user decision, §5).
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
| **N.2** | Deterministic Hearst is-a miner → high-precision `is_a` relation seeds | 1.5–2d | N.1 |
| **N.3** | Abstention gate: `INSUFFICIENT_EVIDENCE` + not-a-concept pre-filter + over-generation metric | 2–2.5d | — |
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

### N.2 — Hearst is-a miner
- **New** `candidates.py::mine_hearst_isa(chunk_text) -> list[(narrow, broad)]`:
  regex Hearst patterns ("X such as A, B", "A and other X", "X including A",
  "X, especially A") → high-precision `is_a` candidate pairs.
- **Modify** the extraction merge to SEED these as `is_a` relations with a
  provenance tag `relation_source="hearst"` and a conservative confidence, so the
  post-filter (`ontology_constraint_filter`, dedup) still governs them.
- **AC**: precision-first (no pattern → no relation); every mined relation carries
  its source pattern + chunk provenance; the LLM path is unchanged (additive).
- **Tests**: `test_hearst.py` — each pattern fires on a positive, abstains on a
  near-miss; mined relations carry provenance.

### N.3 — Abstention gate + over-generation metric
- **Modify** `prompts/pass2.py`: an explicit instruction that a chunk with no
  genuine domain entity returns an empty result WITH a reason (`INSUFFICIENT_EVIDENCE`),
  not a plausible-but-empty concept — the article's core lesson.
- **New** `pipelines/ontology-extraction/.../not_a_concept.py`: a two-tier filter —
  a deterministic pre-pass ("obvious UI label / table field / boilerplate") fast-
  paths accept/reject, and the **LLM-judge (default ON, reusing `llm_matcher`)**
  arbitrates the ambiguous middle (D4). A stricter, extraction-time cousin of
  `noise_filter`.
- **New** `apps/app-main/src/app_main/services/.../extraction_metrics.py` (mirrors
  M.5's `chunking_metrics`): `over_generation_rate = 1 − survivors/extracted` +
  `abstain_rate` — a falsifiable gate (does the pre-filter reduce downstream
  noise-filter workload without dropping real entities?).
- **AC**: abstain path emits a reason + metric; the not-a-concept filter is
  deterministic + config-tunable; no real domain entity in the golden fixtures is
  dropped.
- **Tests**: `test_abstention.py` (empty-on-junk with reason), `test_not_a_concept.py`
  (UI/table artifacts rejected, real concepts kept), `test_extraction_metrics.py`.

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
