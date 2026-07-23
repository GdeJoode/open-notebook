# Track V — Reference / footnote extraction (SPRINT PLAN — PROPOSED)

> **Status**: 📝 PROPOSED (2026-07-23) — **awaiting human approval**. Upgraded from
> the original agenda note into an actionable track-planner sprint plan, grounded
> against the live U.3 contract (`ParsedReference` in
> `packages/shared/src/shared/retrieval/cites_matching.py:78`) and the existing
> reuse anchors. Track ID `V`.

## Vision

Turn documents' **reference sections + footnotes** into a structured
`List[ParsedReference]` per source, so Track U.3's already-built `cites`
mechanism can materialize real source→source citation edges. V is the **producer**;
U.3 is the **consumer**. Today U.3 is complete but yields **0 live `cites` edges
because nothing feeds it** — V closes that gap.

## Why it's needed (staging evidence, Track U.1 + 2026-06-27 probe)

- Docling emits **no** `footnote`/`reference`/`bibliography` element types — only
  generic `list_item`/`text`/`heading`/`section_header`.
- BUT the reference *regions are locatable*: `section_header`/`heading` nodes
  literally titled "References" (9 found across the corpus, incl. both papers),
  and the papers carry real bibliographies (~20 + ~12 DOIs in `full_text`).
- So the data is **present but unstructured** — nothing parses it into citable
  references today.

## What already exists (reuse anchors — this track is producer-only)

| Anchor | Path | Reused for |
|---|---|---|
| `ParsedReference` (the contract) | `packages/shared/.../retrieval/cites_matching.py:78` | V.3 output type — keep stable |
| `normalize_doi`, `_surnames`, matcher | same file | V.3 DOI normalization; U.3 does the matching |
| U.3 cites materialization | `apps/app-main/.../services/cites_materialization_service.py` | V.5 hand-off target |
| `CrossrefProvider` (K.4) | `packages/shared/.../vocabulary/crossref_provider.py` | V.4 DOI resolution + precision-guard pattern to mirror |
| Vocabulary provider interface | `packages/shared/.../vocabulary/provider.py` (`VocabMatch`), `http_client.py` | V.4 pattern to mirror for the new `WorkResolver` cascade |
| Chunk structure (`section_path`, `physical_page`, `element_type`) | `surrealdb-service/.../repositories/source.py` (X.1/I.F) | V.1 region location |
| cheap-first→LLM-on-margin cascade | `design-thematic-classification.md` pattern | V.2 form classifier |

> **Interface note**: the existing provider interface is entity-vocabulary shaped
> (`lookup(name, entity_type) -> List[VocabMatch]`, Track K). Reference resolution
> is a **different shape** (`resolve(ParsedReference) -> ResolvedWork`). V.4 adds a
> sibling **`WorkResolver`** interface + one provider per source, reusing K.4's
> HTTP client + precision-guard *pattern*, not the entity `VocabProvider` directly.

## Scope & cut-line

**In scope (v1)**: bibliography-**section** references (the locatable ~90%): locate
region → segment → parse → `ParsedReference` → optional **multi-source resolution**
(OpenAlex + Crossref + RePEc + overheid.nl, see V.4) → feed U.3 as a **post-ingest
pass**.

**Out of scope / best-effort (documented)**:
- Inline footnotes/endnotes: v1 recovers only what appears as `full_text` markers;
  layout/superscript recovery is **deferred** (Docling doesn't tag them).
- No change to the U.3 matcher or the `cites` schema — V only produces input.

## Decision gates (resolve the agenda note's 4 open questions)

- **V-D1** (region source): **structure-first** (`section_header` titled
  References/Referenties/Bibliography via chunk `section_path`) **+ `full_text`
  regex fallback** for the bibliography block. Rec: both, structure preferred.
- **V-D2** (footnotes): v1 = `full_text` marker recovery only; layout/superscript
  deferred. Rec: accept the gap, log what was skipped.
- **V-D3** (external lookup): a **multi-source resolver cascade** routed by
  reference *shape*, behind one `WorkResolver` interface + the K.4 precision guard
  (single high-confidence match). Rec order: DOI → **Crossref** then **DataCite**;
  econ working-paper → **RePEc/CitEc**; arXiv-id → **arXiv**; NL policy/Kamerstuk →
  **overheid.nl (KOOP SRU)**; title/author fallback → **OpenAlex** (broadest, open);
  then identity-enrich authors→**ORCID**, institutions→**ROR**. Phase order: ship
  **OpenAlex first** (largest open recall) + keep Crossref, add RePEc + overheid.nl
  as the domain-specific second layer; ORCID/ROR/DataCite/arXiv are Q9-anticipated
  follow-ons. Title/author search stays opt-in per source (noisier). ⚠ **NARCIS**
  (old NL research portal) was **discontinued 2023 — do not use**; OpenAIRE covers
  the NL/EU research + funding graph instead.
- **V-D4** (build site): **post-ingest pass** (decoupled from Docling), not inline
  enrichment. Rec: post-ingest — keeps the load-bearing ingest path untouched (T).

## Phases (Backend → Integration; one PR per phase, `track/v<n>-…`)

### V.1 — Reference-region location (Backend, pure) · reuse-heavy
**Deliverables**: `pipelines/.../references/region_locator.py` — per source, find the
reference region: chunk `section_path`/`section_header` titled
References/Referenties/Bibliography (structure), else a `full_text` bibliography-block
regex (fallback). Returns region text + span + `located_via` provenance.
**ACs**: (1) locates the References section on both known papers; (2) absent region →
empty result, no crash; (3) records `located_via` (structure|full_text|none);
(4) pure/deterministic.
**Tests**: fixtures from the two probe papers + a no-reference doc.
**PR boundary**: location only; no parsing, no DB.

### V.2 — Segmentation + reference-form classifier (Backend, pure) · new
**Deliverables**: `references/segmenter.py` — split the region into individual
entries via cheap-first heuristics (numbered list, author-year, DOI line, footnote
marker); an LLM classifier fires **only on ambiguous forms** (the margin), per the
thematic-classification cascade.
**ACs**: (1) segments the ~20 and ~12-DOI bibliographies into individual entries;
(2) heuristic-only on clean forms (no LLM call); (3) LLM invoked only above an
ambiguity threshold; (4) over-segmentation bounded (precision guard).
**Tests**: segmentation counts on the fixtures; assert no-LLM on the clean path.
**PR boundary**: segmentation only.

### V.3 — Parse to `ParsedReference` (Backend, pure) · reuse-heavy
**Deliverables**: `references/parser.py` — each entry → `ParsedReference{raw_text,
title, authors, year, doi, venue}`. DOI via regex + `cites_matching.normalize_doi`;
authors→surname-friendly; `raw_text` always populated.
**ACs**: (1) emits valid `ParsedReference` passing its `__post_init__`; (2) DOI-bearing
entries carry a normalized DOI; (3) author-year entries carry surnames + year;
(4) title-only entries still valid (best-effort fields empty, `raw_text` set).
**Tests**: field-level parse assertions across the fixture entries incl. DOI-only,
author-year, and title-only forms.
**PR boundary**: parsing only; output not yet persisted.

### V.4 — External resolution: multi-source `WorkResolver` cascade (Backend, opt-in)
**Deliverables**:
- `references/work_resolver.py` — a `WorkResolver` interface
  (`resolve(ParsedReference) -> ResolvedWork | None`) + a cascade that routes by
  reference *shape* and stops at the first high-confidence hit (K.4 precision guard).
- One provider per source, mirroring the K.4 HTTP-client + precision-guard pattern:
  | Provider | Source | Routes when | Access |
  |---|---|---|---|
  | `CrossrefResolver` | Crossref | has DOI (journal/book) | open |
  | `DataCiteResolver` | DataCite | has DOI (dataset/thesis/preprint/software) | open |
  | `OpenAlexResolver` | OpenAlex | title/author fallback (broadest recall) | open, no key |
  | `RePEcResolver` | RePEc/CitEc | econ working-paper shape | **access code (email)** |
  | `ArxivResolver` | arXiv | arXiv id present | open |
  | `OverheidResolver` | overheid.nl (KOOP SRU) | NL Kamerstuk/policy citation | open |
- Identity enrichment (post-match, opt-in): authors→**ORCID**, institutions→**ROR**.
- **Phasing**: v1 ships **OpenAlex + Crossref** (biggest open recall) and the
  **RePEc + overheid.nl** domain layer; DataCite/arXiv/ORCID/ROR are Q9-anticipated
  follow-ons behind the same interface.

**ACs**: (1) DOI refs resolve via Crossref/DataCite to confirmed metadata;
(2) a title/author-only econ working-paper resolves via OpenAlex or RePEc where a
DOI is absent; (3) an NL policy citation resolves via overheid.nl; (4) ambiguous →
**unresolved** (no false enrichment — precision guard, single high-confidence match);
(5) each source's title/author search is opt-in (noisier) and per-source rate-limited;
(6) any provider network failure → graceful skip to the next in the cascade, logged;
(7) RePEc runs only when its access code is configured, else that leg is skipped
(no hard dependency).
**Tests**: mocked HTTP per provider (no live network in CI); the shape-router picks
the right provider per fixture; precision-guard rejection case; RePEc-absent skip.
**PR boundary**: resolution only; assumes V.3 merged.

### V.5 — Orchestration: feed U.3 + post-ingest hook (Integration) · reuse-heavy
**Deliverables**: `references/reference_extraction_service.py` — run V.1→V.4 per
source and hand the `List[ParsedReference]` to
`cites_materialization_service` (U.3). Post-ingest pass; batch via the job-queue.
**ACs**: (1) end-to-end on the two papers → references parsed → `cites` edges
materialized *(LIVE-DB AC — deferred to a live run, per repo pattern)*;
(2) idempotent re-run (no dup edges — U.3 clear-before-relate); (3) a source with no
references → no-op; (4) job enqueued at the seam (assert per job-queue-singleton
pattern).
**Tests**: seam-level enqueue + consumer with a live-DB fixture where available;
the true end-to-end edge count is the deferred live AC.
**PR boundary**: orchestration; assumes V.1–V.4 merged.

### V.6 — Docs + RETRO + close (Integration)
**Deliverables**: ARCHITECTURE references section; roadmap Track V entry (currently
plan-only); `RETRO.md`; `_status.md` update.
**ACs**: docs updated; full-suite regression green; track marked CLOSED.

## Risks & mitigations

1. **Docling's generic node types** → region location may miss non-standard layouts
   → structure + full_text dual path (V-D1); log `located_via=none` misses.
2. **LLM cost on segmentation** → cheap-first, LLM strictly on the ambiguous margin.
3. **External-lookup noise** (across the V.4 cascade) → DOI-first (Crossref/DataCite)
   before title/author fallback (OpenAlex); K.4 single-high-confidence precision
   guard; per-source title search opt-in + rate-limited.
5. **Source availability/coupling** → RePEc needs an emailed access code (skip-leg
   if absent, never a hard dep); providers are independent behind `WorkResolver` so
   one being down/unconfigured degrades gracefully to the rest. NARCIS is retired
   (2023) — excluded; OpenAIRE is the NL/EU alternative if broader recall is wanted.
4. **Live verification** → the end-to-end edge-count AC (V.5.1) needs a live
   SurrealDB; explicitly a deferred live-run AC, consistent with the tracked
   ~54 live-gap items in `_status.md`.

## Effort estimate

Small–medium. V.3/V.4/V.5 are reuse-heavy (contract + Crossref + U.3 already exist);
the genuinely new work is region location (V.1) + the form segmenter (V.2). No new
schema — V produces input to an existing mechanism.

## Open questions for the operator

- [ ] Approve the track / schedule it as the next build?
- [ ] V-D3: confirm the v1 resolver set — **OpenAlex + Crossref + RePEc +
      overheid.nl** (rec), or start DOI-only (Crossref/DataCite) and add the rest later?
- [ ] RePEc access-code request worth it for v1, or defer until the econ-working-paper
      miss-rate from OpenAlex/Crossref is measured?
- [ ] Footnote recovery: accept the v1 `full_text`-marker-only gap, or invest in
      layout/superscript recovery now?
