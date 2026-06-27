# Phase T.1 — Extraction-economy baseline (decision gate)

> **Read-only discovery.** No code changed, no DB writes. All numbers measured on
> **staging** (`ws://localhost:8000`, ns `open_notebook`, db `staging`, root/root)
> on 2026-06-27. Corpus sanity check passed: **6 sources, 4,870 entities, 1,466
> relations**. Code citations are to the paths read during this measurement.

## TL;DR — the decision-gate verdict

**R.6 already captured nearly all of the available *search* win.** The R.2 KG
scorer takes only `status='active'` entities (423 of 4,870). R.6's
`normalize_entities_for_signal` then collapses those 423 into **25 emitted
concepts** and drops the other 398 (94.1%) as post-grouping singletons. So of the
LLM's blanket output, **<0.6% (25 / 4,870 rows) actually reaches the search
ranking** — and 23 of those 25 are still generic `topic`/`concept` that the
scorer down-weights anyway.

There is therefore **no material *search-projection* win left for Track T** — the
search waste is already gone in memory. The remaining win is a **cost / write-
amplification** win: extraction makes ~6–17 LLM calls per document to persist
~75% generic, ~91% singleton rows that nothing load-bearing reads. That is real
but it is an **extraction-core** target (T.2b prompt change / T.3 cascade), not a
search-projection one — and per the locked "search-projection-only, careful"
course those steps are higher-risk and gated.

**Recommendation:** proceed to a **thin T.2a** (the reversible search-facing
filter — it formalizes/owns what R.6 already does and costs nothing), but **do
NOT greenlight T.2b/T.3/T.4 on search grounds.** They are only justified if the
product explicitly wants the *LLM-cost / write-volume* reduction (Decision-2's
"prompt appetite") — and that is a separate, deferred call. Track T can
**correctly stop thin** on the search axis.

---

## Measurement 1 — Cost per document

The cost is **derived from the code path** (no token counters are persisted —
`extraction_result.metadata` records only `chunk_count` / `entity_count` /
`relation_count` / `filtering`, no token or call totals). The estimate below is
defensible from the structure; the absolute token figures are estimates and
labelled as such.

**Call structure** (multi-schema path, the production path when `notebook_id` is
set — `entity_extraction_service.py:1099-1115`, `_run_multi_schema:802-980`):
- **Pass-1**: one LLM call **per applicable schema** on a *sample* (first ~2 KB),
  not per chunk (`multi_schema_orchestrator.py:575-616`, `_sample_text_for_pass1`).
- **Pass-2**: `run_pass2` loops **one LLM call per packed window per schema**
  (`pass2_typed_extraction.py:452-460` — `for chunk in chunks: … build_pass2_prompt`).
- Track M packs the fixed ingestion chunks into model-sized windows first
  (`entity_extraction_service.py:1091-1096` → `context_packer.pack_chunks_for_model`).
  Effective budget = `min(input_budget, max_window_cap=6000)`; with the default
  `context_window=8192`, `max_output=2000`, `overhead=2000`, `safety=0.85` the
  measured budget is **3,563 tokens/window**.

**Windows per source** (measured, from `is_content` char totals on staging,
packed with the real `pack_chunks_for_model`):

| source | content chunks | content chars | est. tokens | **packed windows** |
|---|---:|---:|---:|---:|
| A | 284 | 66,306 | 16,577 | 5 |
| B | 209 | 47,398 | 11,850 | 4 |
| C | 135 | 36,878 | 9,220 | 3 |
| D | 260 | 125,595 | 31,399 | 10 |
| E | 280 | 67,062 | 16,766 | 5 |
| F | 279 | 70,366 | 17,592 | 6 |

**Average ≈ 5.5 Pass-2 windows/document** (range 3–10).

**LLM calls per document** ≈ `n_schemas` (Pass-1, on a sample) + `n_schemas × ~5.5`
(Pass-2). For the Regio-Deal corpus auto-detection typically applies **1–2
schemas** (the notebook has no configured `base_ontology` — plan note + L.4 path),
so:

> **≈ 6–8 LLM calls/document for 1 schema, ≈ 12–17 for 2 schemas.**

Per Pass-2 call the input ≈ window (≤6,000 tok, here 3,563-capped) + ontology block
(~600 tok for the small Regio ontology) ≈ **~4–6.5K input tokens/call**, plus
output. So a 2-schema 10-window document is on the order of **~120–200K input
tokens** of LLM work — to produce the rows analysed below.

**Honest caveat:** schema count per document was not directly logged on staging;
1–2 is inferred from the corpus + L.4 path. The window counts ARE measured.

---

## Measurement 2 — Output composition

`status` values are `active | reference | merged | archived`
(`shared/models/entity.py:103`). "Generic" = `entity_type ∈ {topic, concept,
other}` (the LOW/FLOOR salience buckets in `kg_source_scorer.TYPE_SALIENCE`);
"named" = everything else. `df` = number of distinct `source_documents`;
singleton = `df ≤ 1`. (`source_documents` was `array::distinct`-counted in
Python — `type::is::none()`-safe, nulls treated as df 0.)

### Status split (all 4,870)
| status | count | % |
|---|---:|---:|
| archived | 3,313 | 68.0% |
| reference | 1,055 | 21.7% |
| **active** | **423** | **8.7%** |
| merged | 79 | 1.6% |

Only **8.7%** of produced entities survive triage as `active` — the rest are
archived/reference/merged. (Track Q triage status.)

### Named vs generic
| set | named | generic | generic % |
|---|---:|---:|---:|
| ALL (4,870) | 1,233 | 3,637 | **74.7%** |
| **ACTIVE (423)** | 76 | 347 | **82.0%** |
| archived (3,313) | 802 | 2,511 | 75.8% |
| reference (1,055) | 335 | 720 | 68.2% |

Active type breakdown: `topic` 344, `administrative_area` 35, `programme` 24,
`government_organization` 17, `concept` 3. So among the entities that DO survive
triage, **82% are generic buckets**.

### Singleton vs linking
| set | linking (df≥2) | singleton (df≤1) | singleton % |
|---|---:|---:|---:|
| ALL | 445 | 4,425 | **90.9%** |
| **ACTIVE** | 29 | 394 | **93.1%** |

df distribution (active): df1=394, df2=16, df3=12, df4=1. **Generic singletons
alone = 321 of 423 active (75.9%)** — produced, kept active, but link nothing.

### Relations (1,466 total)
- **37 distinct predicates.** Top: `RELATED` 433 (30%), `IS_PIJLER_VAN` 317,
  `LEIDT_TOT` 274, `VERSTERKT` 258, `VERMINDERT` 52 — then a long tail.
- **17 of 37 predicates (46%) appear exactly once** — duplicate/idiosyncratic
  predicate noise.
- The single largest predicate is the **generic, untyped `RELATED`** (433, ~30%
  of all edges). Note `RELATED`/`RELATED_TO` are **not** in R.6's
  `PREDICATE_CANON` map (`kg_signal_normalizer.py:113-146`), so they pass through
  uncanonicalized — but relation expansion is gated OFF by default
  (`kg_source_scorer.score_related_sources(expand_relations=False)`), so today
  relations contribute **nothing** to the live search signal regardless.

---

## Measurement 3 — Origin of the generic output

Generic-topic emission is **instructed, in two places, and is separable** from the
named-entity instruction:

1. **The combined prompt has a dedicated "Thematic Concepts" section + guideline**
   (`ontology-manager/prompts.py`):
   - `generate_combined_extraction_prompt` adds a whole `## Thematic Concepts`
     block (`prompts.py:282-292`) and a `"concepts": [...]` array in the output
     schema (`prompts.py:327-329`), driven by `include_concepts=True`
     (default, `prompts.py:206`).
   - Guideline 3 explicitly tells the LLM to "**Extract thematic concept
     mentions**" (`prompts.py:364-371`). This is an *additive instruction* — it
     could be dropped (`include_concepts=False`) without touching the entity
     instruction.

2. **The `"other"` fallback is baked into the core entity instruction** — the
   harder one to cut:
   - Combined prompt: *"if an entity genuinely fits NO defined type, set
     `entity_type` to \"other\" — never drop it"* (`prompts.py:347-348`).
   - Pass-2 prompt repeats it: *"ONLY if an entity genuinely fits no defined
     type, set its type to \"other\" — never drop it"*
     (`pass2.py:278-282`), under a `## CRITICAL — Exhaustive extraction` header
     that demands **complete recall** including "theme, indicator, … and named
     concept" (`pass2.py:261-270`).

So: **the `topic`/`concept` mass comes from the optional Concepts section
(droppable via a flag) + the exhaustive-recall mandate; the `other` bucket is the
core entity fallback.** T.2b's prompt lever is real (drop the Concepts section,
soften "exhaustive recall of every theme/indicator") but it lives in the
load-bearing entity prompt, which is exactly the risk Decision-2 flagged.

---

## Measurement 4 — What R.6 already removes vs what extraction still wastes

R.6's `normalize_entities_for_signal` (drop_singletons=True) was run against the
**actual 423 active rows** from staging (the same input the R.2 scorer takes):

```
NormalizationStats(input_entities=423, grouped_concepts=413, merged_groups=10,
                   singletons_dropped=388, emitted_concepts=25, empty_skipped=0)
```

- **Input to signal: 423** active entities.
- After case/type grouping: 413 concepts (only **10 merged groups** — case/type
  de-fragmentation is small on this corpus).
- After singleton drop: **388 dropped**, **25 concepts emitted** to the scorer.
- **R.6 removes 398 / 423 (94.1%) of active entities from the search signal.**
- Of the 25 survivors: **23 are still generic** `topic`/`concept`, only **2 are
  named** (`administrative_area "Regio"`, `programme "Regio Deal"`). Salience-
  weighted, generic still carries **63%** of the surviving signal mass — but it's
  a tiny 25-concept signal to begin with.

The 25 survivors (df ≥ 2) are mostly cross-document themes the LLM happened to
emit consistently: *brede welvaart, energietransitie, gezondheid, jongeren, Regio
Deal, samenredzaamheid, vergrijzing, wonen, …* — i.e. exactly the
embedding-clusterable themes Track T's design wants to move OFF the LLM.

### The "wasted for BOTH" fraction (Track T's real target)

| layer | count | % of 4,870 |
|---|---:|---:|
| produced (all entity rows) | 4,870 | 100% |
| reach `active` (triage keeps) | 423 | 8.7% |
| reach the **search signal** (R.6 emits) | **25** | **0.5%** |
| of those, **named** | 2 | 0.04% |

- **4,447 rows (91.3%) never even reach `active`** — archived/reference/merged by
  triage. They are not in the search signal and (for archived/merged) not in
  exports' active view either.
- Of the 423 active, **398 (94.1%) are dropped by R.6** from search.
- **Generic + singleton that is dead for BOTH search (R.6 drops it) AND not a
  named export anchor:** the **321 generic active singletons** are the clearest
  "wasted for both" set inside the active tier; extrapolated across all statuses,
  **generic singletons = 3,288 of 4,870 rows (67.5%)** are produced rows that
  link nothing and that the search projection discards.

**So extraction spends ~6–17 LLM calls/doc to write ~3,300 generic-singleton rows
that the search signal never uses.** That is the *write-amplification / LLM-cost*
waste. It is NOT search-signal waste — R.6 already neutralised that.

**Caveat (Decision-1):** those generic rows are NOT pure waste for the **export**
artifact — the Obsidian/NetworkX/JSONL exports read the canonical rows including
topics (plan Risk 2). Cutting them at the source trades export richness for lean.
The locked decision keeps exports rich and prunes only the in-memory search
projection — under which R.6 has already done the pruning.

---

## What to cut (concrete)

Safe-to-stop-*search-feeding* (already effectively cut by R.6; T.2a just owns it):
1. **Generic singletons** (`topic`/`concept`/`other` with df ≤ 1): **321 active /
   3,288 total**. R.6 already drops every df≤1 concept; T.2a can formalize this as
   an explicit, reversible search-projection filter with zero behaviour change.
2. **`other` bucket** (739 rows, salience floor 0.05): contributes essentially
   nothing to ranking; safe to exclude from the signal projection.

Cut only if the **LLM-cost** win is explicitly wanted (T.2b/T.3 — extraction-core,
higher risk, gated):
3. **The `## Thematic Concepts` prompt section** (`prompts.py:282-292,364-371`,
   `include_concepts=False`) — stops the LLM inventing per-document `topic`/
   `concept`. This is the **3,288 generic-singleton write** at its source. Moves
   the theme job to the embedding/clustering cascade (T.3), where the 25 surviving
   themes belong.
4. **Soften the "exhaustive recall of every theme/indicator/named concept"
   mandate** (`pass2.py:261-282`) toward named-entity + relation recall — the
   direct LLM-call/token saving.

Do **not** cut (load-bearing):
- The 2 named survivors and the named active set (76 rows) — that IS the signal.
- The `other` *fallback instruction* if it risks dropping genuinely-named
  entities the schema doesn't cover yet (recall-protective).

---

## Decision-gate verdict (explicit)

| question | answer |
|---|---|
| Material remaining **search** win beyond R.6? | **No.** R.6 drops 94.1% of active and the signal is already a 25-concept projection. |
| Material remaining **cost / write-volume** win? | **Yes** — ~6–17 LLM calls/doc producing ~3,300 generic-singleton rows nothing in search reads. |
| Is that win on the **safe (search-projection)** axis or the **risky (extraction-core)** axis? | The risky axis (T.2b prompt / T.3 cascade). T.2a (search filter) yields ~0 incremental search win because R.6 got there first. |
| Proceed to T.2a? | **Yes, thin** — reversible, formalizes R.6, no risk. |
| Proceed to T.2b/T.3/T.4? | **Only if the product explicitly wants the LLM-cost/leaner-write reduction** (Decision-2 "prompt appetite"). Not justified on search grounds. Defer per locked course. |

**Honest headline: R.6 already captured the search win. Track T's remaining value
is purely cost/volume, lives in the extraction core, and is gated — so Track T can
legitimately stop at a thin T.2a unless the cost reduction is independently
prioritised.**
