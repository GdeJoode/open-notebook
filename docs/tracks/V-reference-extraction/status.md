# Track V — status

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
