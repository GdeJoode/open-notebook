# Track V — status

## Phases V.1 + V.2 + V.3 — reference-EXTRACTION producer (Backend) — READY FOR REVIEW

**Branch**: `track/v123-reference-producer` (off `track/v4-work-resolver` @ `ebfc64d`; stacked PR)
**Date**: 2026-07-24
**Scope**: pure text processing only — the producer side that turns a source's
document structure into a `List[ParsedReference]` (the V → U.3/V.4 boundary type,
unchanged). NO DB, NO external APIs, NO LLM, NO U.3 / `cites` materialization
wiring (that is V.5). Complements V.4 (the resolver side) in the same
`packages/shared/src/shared/references/` subpackage.

### What was built
The three pure producer stages + a thin chain, sibling to the V.4 resolver files.

1. **V.1 — `references/region_locator.py`** — locate the bibliography region.
   - `ReferenceChunk`: DB-free projection of a persisted `chunk`
     (`text`/`section_path`/`element_type`/`order`/`page`), mirroring the fields
     V needs (docling_document_json is transient — structure is the chunks).
   - `locate_reference_region(chunks, full_text="") -> LocatedRegion`:
     structure-first (a chunk whose `section_path` tail / heading matches an
     EN+NL reference vocabulary — References/Referenties/Bibliography/Literatuur/
     Bronnen/Works Cited/…), else a `full_text` heading-only regex fallback with
     an appendix/acknowledgements terminator so a trailing appendix is not
     swallowed. `located_via` ∈ {`structure`,`full_text`,`none`}; span is the
     char offset into `full_text` when known. Absent region → empty, never raises.
2. **V.2 — `references/segmenter.py`** — split the region into entries.
   - `segment_region(region_text, *, ambiguity_resolver=None) -> List[str]`:
     cheap-first deterministic heuristics — numbered lists (`[1]`/`1.`/`(1)`),
     blank-line paragraph blocks, or author-year start-detection. Continuation
     lines are merged so a wrapped multi-line entry stays ONE entry (bounded
     over-segmentation). `ambiguity_resolver` is the LLM-on-the-margin seam
     (per `design-thematic-classification`); the default path never calls it.
3. **V.3 — `references/reference_parser.py`** — parse one entry.
   - `parse_reference(entry) -> ParsedReference`: DOI (regex + shared
     `normalize_doi`), year (parenthesized-slot-first; Dutch vergaderjaar
     `2023/24`→2023), authors (APA `Surname, I.` comma-form AND IEEE
     initials-first `I. Surname`; matcher normalizes to surnames), best-effort
     quoted/APA title+venue. `raw_text` ALWAYS set → a partially-parseable entry
     still yields a valid `ParsedReference`.
4. **Chain — `references/reference_extractor.py`** —
   `extract_references(chunks, full_text="", *, ambiguity_resolver=None) ->
   List[ParsedReference]`: locate → segment → parse; empty region → `[]`. Pure;
   V.5 orchestration/U.3/DB wiring is deliberately NOT here.

New producer surface is exported from `references/__init__.py` (extends the V.4
exports cleanly; V.4 resolver files untouched).

### Tests / checks
- `uv run pytest packages/shared/tests/ -q` → **602 passed** (42 new; 560 prior,
  no regressions). New files: `test_reference_region_locator.py` (V.1),
  `test_reference_segmenter.py` (V.2), `test_reference_parser.py` (V.3),
  `test_reference_extractor.py` (chain). All offline/deterministic — no DB, no
  network, no LLM. Fixtures: `tests/fixtures/references/*.txt` (synthetic,
  committed; mimic APA/IEEE/Dutch bibliographies + Kamerstuk, not the gitignored
  real corpus).
- `uv run ruff check` on changed files → clean.
- `uv run mypy` on the 4 new source modules → clean (the `mypy.ini`
  duplicate-option warning is pre-existing/unrelated, as in V.4).

### `# TODO(V-live)` — confirm against the real corpus (tomorrow's manual smoke)
- **Segmentation robustness** (`segmenter.py`): the numbered/blank-line/
  author-year strategy selection is validated on synthetic fixtures; confirm the
  author-year start-detection opener + continuation-cue behave on real wrapped,
  no-blank-line paper bibliographies (economics PDFs) and the Regio Deal
  convenanten.
- **Author parsing** (`reference_parser.py`): APA vs IEEE routing is heuristic;
  confirm coverage on real mixed-style bibliographies (particle surnames,
  "et al.", corporate authors).
- **Full-text region bounds** (`region_locator.py`): the fallback takes the LAST
  heading-only "References" line to end / next terminator; confirm on real
  `full_text` that no in-body "references" prose false-triggers and the terminator
  set covers the real tail sections.
- **Footnotes/endnotes**: v1 intentionally handles only bibliography-section
  references (Docling emits no footnote element type; layout/superscript recovery
  is deferred to a later phase — noted in the region_locator docstring).

---

## Phase V.4 — external work-resolution cascade (Backend) — READY FOR REVIEW

**Branch**: `track/v4-work-resolver` (off `main` @ `5735deb`)
**Date**: 2026-07-23
**Scope**: pure resolution only — no live DB, no U.3 / cites_materialization
wiring (a later phase), no change to `ParsedReference` or the `cites` schema.

### What was built
The outward-looking half of the Track V boundary. Given a `ParsedReference`
(the V → U.3 contract), resolve it to a canonical external `ResolvedWork`,
routed by reference *shape*, behind a single-high-confidence precision guard
(unresolved is always preferred over a wrong match). New workspace subpackage
`packages/shared/src/shared/references/`, sibling to `vocabulary/` and
`retrieval/`.

1. **Core** — `references/work_resolver.py`
   - `ResolvedWork`: DB-free dataclass (mirrors `vocabulary.VocabMatch`).
   - `WorkResolver` / `WorkEnricher` protocols (`name` + `async resolve`).
   - `ResolverCascade`: shape-first routing, stop at first confident hit, fall
     through on network-failure/below-guard, `None` if nothing clears the guard.
     Skips an `available == False` resolver silently (the RePEc gate seam).
   - Precision guard mirrors K.4 `crossref_provider`: token-Jaccard title
     overlap floor (`MIN_TITLE_OVERLAP = 0.5`) + author-surname confirmer +
     blended `MIN_MATCH_CONFIDENCE = 0.5`; DOI/id-exact = certain (1.0).
   - Pure shape detectors: `extract_arxiv_id`, `looks_like_nl_policy`,
     `looks_like_econ_wp`.
2. **Providers** (each ends in `Resolver`, DB-free, HTTP via the shared
   fail-soft `VocabularyHTTPClient`):
   - **ACTIVE** — `OpenAlexResolver` (DOI + title/author fallback, broadest
     recall), `CrossrefResolver` (DOI-exact + title+author query; the K.4
     entity `CrossrefProvider` is untouched), `DataCiteResolver` (DOI path for
     datasets/theses/preprints/software), `ArxivResolver` (Atom API, id-exact),
     `OverheidResolver` (overheid.nl KOOP SRU, dossier-id or title-overlap).
   - **GATED** — `RePEcResolver`: reads `REPEC_API_KEY`; unset → `available =
     False` → cascade skips silently; `resolve()` returns `None` without raising.
     Configured → real guarded title/author lookup. The ONLY inert leg, and it
     is a clean tested skip (not a stub).
   - **ENRICHMENT (opt-in)** — `ReferenceEnricher` attaches ORCID / ROR ids to
     an already-resolved work, best-effort + precision-guarded (ORCID only when
     the returned family name matches; ROR only when its matcher flags
     `chosen`). Never required.
3. **Infra** — added `VocabularyHTTPClient.get_text` (the text counterpart of
   `get_json`) so the XML authorities (arXiv Atom, KOOP SRU) keep the same
   caching / rate-limit / timeout / fail-soft discipline.

### Tests / checks
- `uv run pytest packages/shared/tests/` → **560 passed** (52 new for V.4; no
  regressions). New files: `test_work_resolver.py`, `test_openalex_resolver.py`,
  `test_crossref_resolver.py`, `test_datacite_resolver.py`,
  `test_arxiv_resolver.py`, `test_overheid_resolver.py`,
  `test_repec_resolver.py`, `test_reference_enrichment.py`. All HTTP mocked via
  `httpx.MockTransport` — no live network.
- `uv run ruff check` on all changed files → clean.
- `uv run mypy` on `references/` + `http_client.py` → clean (the `mypy.ini`
  duplicate-option warning is pre-existing, unrelated).

### `# TODO(V.4-live)` — assumptions to confirm on the live smoke
- `extract_arxiv_id` id-form coverage vs real bibliographies (work_resolver.py).
- arXiv Atom element/namespace mapping — title / author name / published year /
  `arxiv:doi` (arxiv_resolver.py).
- KOOP SRU record structure — dcterms nesting (title / identifier / type /
  issued) + enriched `preferredUrl`, and the dossier-identifier format
  (overheid_resolver.py).
- CitEc request contract (endpoint path, access-code param name) + JSON
  response shape, to confirm once the emailed RePEc code arrives
  (repec_resolver.py).
- ORCID public API needs `Accept: application/json` to return JSON; the shared
  client currently sends only `User-Agent`, so the live ORCID leg would need
  that header added or it degrades to XML → no-match (enrichment.py).
